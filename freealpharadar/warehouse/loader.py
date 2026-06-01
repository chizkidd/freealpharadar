"""Parse SEC Financial Statement Data Set ZIPs into a tidy fundamentals table.

Each quarterly ZIP contains tab-separated ``sub.txt`` (one row per filing) and
``num.txt`` (one row per numeric XBRL fact). This module filters them to annual
10-K/20-F fundamentals, coalesces the many tag variants into canonical fields,
pivots to one row per ``(cik, fiscal_year)``, maps CIK -> ticker, and writes a
partitioned Parquet store the screening layer queries with DuckDB.

The output is tiny (a few canonical columns × ~8k companies × ~15 years) even
though the raw ``num.txt`` files are large.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from freealpharadar.config import SEC_TICKER_MAP, SEC_USER_AGENT
from freealpharadar.utils import get_logger
from freealpharadar.warehouse.config import (
    ANNUAL_FORMS,
    CANONICAL_TAGS,
    CIK_TICKER_CACHE,
    FACTS_PARQUET,
    FLOW_FIELDS,
    INSTANT_FIELDS,
)
from freealpharadar.warehouse.downloader import download_all

logger = get_logger(__name__)

# Reverse map: raw XBRL tag -> canonical field.
_TAG_TO_FIELD: Dict[str, str] = {
    tag: field for field, tags in CANONICAL_TAGS.items() for tag in tags
}
_WANTED_TAGS = frozenset(_TAG_TO_FIELD)
_CANONICAL_FIELDS = list(CANONICAL_TAGS)


def load_cik_ticker_map(
    refresh: bool = False, cache_path: Optional[Path] = None
) -> Dict[int, str]:
    """Return a ``{cik:int -> ticker}`` map from SEC's company_tickers.json.

    Cached to disk; pass ``refresh=True`` to re-fetch. Network is only touched
    on a cache miss. Tests inject their own map instead of calling this.
    """
    import requests

    cache_path = Path(cache_path or CIK_TICKER_CACHE)
    if cache_path.exists() and not refresh:
        return {int(k): v for k, v in json.loads(cache_path.read_text()).items()}

    resp = requests.get(
        SEC_TICKER_MAP, headers={"User-Agent": SEC_USER_AGENT}, timeout=60
    )
    resp.raise_for_status()
    mapping = {
        int(row["cik_str"]): str(row["ticker"]).upper() for row in resp.json().values()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping))
    return mapping


def _read_member(zf: zipfile.ZipFile, name: str, usecols: List[str]) -> pd.DataFrame:
    """Read a TSV member of the ZIP as strings, keeping only ``usecols``."""
    with zf.open(name) as fh:
        return pd.read_csv(
            fh,
            sep="\t",
            dtype=str,
            usecols=lambda c: c in set(usecols),
            na_filter=False,
            on_bad_lines="skip",
        )


def _read_num(zf: zipfile.ZipFile, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream ``num.txt`` in chunks, keeping only wanted USD tags.

    ``num.txt`` can be 1-2 GB; chunked reading bounds memory to one chunk while
    the filtered result (a handful of tags) stays small.
    """
    cols = {"adsh", "tag", "ddate", "qtrs", "uom", "value"}
    frames: List[pd.DataFrame] = []
    with zf.open("num.txt") as fh:
        for chunk in pd.read_csv(
            fh,
            sep="\t",
            dtype=str,
            usecols=lambda c: c in cols,
            na_filter=False,
            on_bad_lines="skip",
            chunksize=chunksize,
        ):
            chunk = chunk[chunk["tag"].isin(_WANTED_TAGS) & (chunk["uom"] == "USD")]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=list(cols))
    return pd.concat(frames, ignore_index=True)


def load_quarter(
    zip_path: Path, cik_to_ticker: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """Load one quarter ZIP into a wide per-(cik, fy) fundamentals frame.

    Args:
        zip_path: Path to a ``YYYYqQ.zip`` Financial Statement Data Set.
        cik_to_ticker: Optional CIK->ticker map; rows without a known ticker are
            kept with an empty ticker (callers may drop them).

    Returns:
        DataFrame with columns ``cik, ticker, name, fy`` plus the canonical
        fields (``revenue, gross_profit, rnd, net_income, assets, liabilities``).
        Empty frame if the ZIP is unreadable.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            sub = _read_member(
                zf, "sub.txt", ["adsh", "cik", "name", "form", "fy", "fp", "period"]
            )
            num = _read_num(zf)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", zip_path, exc)
        return pd.DataFrame()

    sub = sub[sub["form"].isin(ANNUAL_FORMS)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["cik"] = pd.to_numeric(sub["cik"], errors="coerce")
    sub["fy"] = pd.to_numeric(sub["fy"], errors="coerce")
    # Fall back to the calendar year of the period end when fy is absent.
    period_year = pd.to_numeric(sub["period"].str.slice(0, 4), errors="coerce")
    sub["fy"] = sub["fy"].fillna(period_year)
    sub = sub.dropna(subset=["cik", "fy"])
    sub["cik"] = sub["cik"].astype(int)
    sub["fy"] = sub["fy"].astype(int)
    meta = sub.set_index("adsh")[["cik", "name", "fy"]]

    if num.empty:
        return pd.DataFrame()
    num = num.copy()
    num["field"] = num["tag"].map(_TAG_TO_FIELD)
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num.dropna(subset=["value"])

    # Flows are annual (qtrs == 4); balance-sheet items instantaneous (qtrs == 0).
    is_flow = num["field"].isin(FLOW_FIELDS) & (num["qtrs"] == "4")
    is_instant = num["field"].isin(INSTANT_FIELDS) & (num["qtrs"] == "0")
    num = num[is_flow | is_instant]

    joined = num.join(meta, on="adsh", how="inner")
    if joined.empty:
        return pd.DataFrame()

    # Latest period end wins within a (cik, fy, field).
    joined = joined.sort_values("ddate")
    joined = joined.drop_duplicates(["cik", "fy", "field"], keep="last")

    wide = joined.pivot_table(
        index=["cik", "fy"], columns="field", values="value", aggfunc="last"
    ).reset_index()
    names = meta.reset_index().drop_duplicates("cik")[["cik", "name"]]
    wide = wide.merge(names, on="cik", how="left")

    mapping = cik_to_ticker or {}
    wide["ticker"] = wide["cik"].map(mapping).fillna("")
    for field in _CANONICAL_FIELDS:
        if field not in wide.columns:
            wide[field] = pd.NA
    return wide[["cik", "ticker", "name", "fy", *_CANONICAL_FIELDS]]


def build_warehouse(
    since_year: int = 2009,
    cik_to_ticker: Optional[Dict[int, str]] = None,
    zip_paths: Optional[Iterable[Path]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Build the full fundamentals Parquet store from quarterly ZIPs.

    Args:
        since_year: Earliest year to ingest.
        cik_to_ticker: CIK->ticker map; fetched via :func:`load_cik_ticker_map`
            when omitted (requires network).
        zip_paths: Explicit ZIP paths (used by tests); otherwise quarters are
            downloaded/cached via the downloader.
        out_path: Output Parquet path; defaults to :data:`FACTS_PARQUET`.

    Returns:
        Path to the written Parquet file.
    """
    out_path = Path(out_path or FACTS_PARQUET)
    if cik_to_ticker is None:
        cik_to_ticker = load_cik_ticker_map()
    paths = list(zip_paths) if zip_paths is not None else download_all(since_year)

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = load_quarter(Path(p), cik_to_ticker)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No quarterly data loaded; nothing to write.")

    facts = pd.concat(frames, ignore_index=True)
    # Across re-filings of the same year in different quarters, keep the last.
    facts = facts.sort_values(["cik", "fy"]).drop_duplicates(["cik", "fy"], keep="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(out_path, index=False)
    logger.info(
        "Wrote warehouse: %d rows / %d companies -> %s",
        len(facts),
        facts["cik"].nunique(),
        out_path,
    )
    return out_path
