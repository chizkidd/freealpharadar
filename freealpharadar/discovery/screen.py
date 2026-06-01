"""Stage 1: cheap fundamentals-only screen over the whole warehouse.

Computes a handful of long-horizon fundamentals for every company in the bulk
store and applies "under-the-radar" gates that need *only* fundamentals (no
prices/ownership, which Stage 2 adds per-ticker): small/mid revenue scale and
sustained multi-year growth. Ranks survivors by a cheap composite and returns a
shortlist for the expensive full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from freealpharadar.utils import get_logger
from freealpharadar.warehouse.store import WarehouseStore

logger = get_logger(__name__)


@dataclass
class ScreenConfig:
    """Tunable gates and weights for the bulk screen.

    Attributes:
        min_years: Minimum annual revenue points required.
        min_revenue: Lower revenue bound (exclude pre-revenue shells), USD.
        max_revenue: Upper revenue bound (keep it small/mid, "under the
            radar"), USD.
        min_revenue_cagr: Minimum multi-year revenue CAGR to qualify.
        cagr_years: Horizon for the revenue CAGR.
        n_candidates: Size of the shortlist handed to Stage 2.
        w_cagr / w_margin / w_rnd: Composite weights.
    """

    min_years: int = 3
    min_revenue: float = 10e6
    max_revenue: float = 2e9
    min_revenue_cagr: float = 0.15
    cagr_years: int = 5
    n_candidates: int = 100
    w_cagr: float = 1.0
    w_margin: float = 0.6
    w_rnd: float = 0.6


def _cagr_over(series: List[Tuple[int, float]], years: int) -> Optional[float]:
    """Annualised growth over the span closest to ``years`` in a yearly series."""
    series = [(y, v) for y, v in series if v is not None]
    if len(series) < 2:
        return None
    latest_fy, latest_val = series[-1]
    if latest_val is None or latest_val <= 0:
        return None
    target = latest_fy - years
    old_fy, old_val = min(series, key=lambda p: abs(p[0] - target))
    span = latest_fy - old_fy
    if span <= 0 or old_val is None or old_val <= 0:
        return None
    return (latest_val / old_val) ** (1.0 / span) - 1.0


def _slope(points: List[Tuple[int, float]]) -> Optional[float]:
    """OLS slope of value vs year; ``None`` with < 3 points."""
    pts = [(x, y) for x, y in points if y is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    denom = sum((x - mx) ** 2 for x, _ in pts)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / denom


def _company_metrics(g: pd.DataFrame, cfg: ScreenConfig) -> Optional[dict]:
    """Compute screen metrics for one company's rows, or ``None`` if too sparse."""
    g = g.sort_values("fy")
    rev = [(int(r.fy), _f(r.revenue)) for r in g.itertuples()]
    rev = [(y, v) for y, v in rev if v is not None and v > 0]
    if len(rev) < cfg.min_years:
        return None

    latest_rev = rev[-1][1]
    cagr = _cagr_over(rev, cfg.cagr_years)

    margins = []
    for r in g.itertuples():
        rv, gp = _f(r.revenue), _f(r.gross_profit)
        if rv and gp is not None and rv > 0:
            margins.append((int(r.fy), gp / rv))
    margin_trend = _slope(margins)

    latest = g.iloc[-1]
    rnd, rev_l = _f(latest.rnd), _f(latest.revenue)
    rnd_intensity = (rnd / rev_l) if (rnd is not None and rev_l) else None

    return {
        "ticker": str(latest.ticker),
        "name": str(latest.get("name", "")),
        "latest_revenue": latest_rev,
        "revenue_cagr": cagr,
        "gross_margin_trend": margin_trend,
        "rnd_intensity": rnd_intensity,
        "years": len(rev),
    }


def _f(value) -> Optional[float]:
    """Coerce to float, returning None on NaN/failure."""
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def screen_candidates(
    store: WarehouseStore, cfg: Optional[ScreenConfig] = None
) -> pd.DataFrame:
    """Screen the whole warehouse and return a ranked candidate shortlist.

    Args:
        store: The warehouse to screen.
        cfg: Screen configuration; defaults to :class:`ScreenConfig`.

    Returns:
        DataFrame of up to ``cfg.n_candidates`` rows, ranked by composite score
        descending, with the per-company metrics used.
    """
    cfg = cfg or ScreenConfig()
    facts = store.query("SELECT * FROM facts WHERE ticker <> ''")
    if facts.empty:
        return pd.DataFrame()

    rows = [
        m
        for _, g in facts.groupby("ticker")
        if (m := _company_metrics(g, cfg)) is not None
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Gates.
    df = df[
        (df["latest_revenue"] >= cfg.min_revenue)
        & (df["latest_revenue"] <= cfg.max_revenue)
        & (df["revenue_cagr"].fillna(-1) >= cfg.min_revenue_cagr)
    ].copy()
    if df.empty:
        return df

    # Cheap composite from min-max normalised metrics (NaNs -> 0 contribution).
    def _norm(col: str) -> pd.Series:
        s = df[col].astype(float)
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or hi == lo:
            return pd.Series(0.0, index=df.index)
        return ((s - lo) / (hi - lo)).fillna(0.0)

    df["screen_score"] = (
        cfg.w_cagr * _norm("revenue_cagr")
        + cfg.w_margin * _norm("gross_margin_trend")
        + cfg.w_rnd * _norm("rnd_intensity")
    )
    df = df.sort_values("screen_score", ascending=False).head(cfg.n_candidates)
    logger.info("Screen produced %d candidates.", len(df))
    return df.reset_index(drop=True)
