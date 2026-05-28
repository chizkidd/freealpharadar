"""Manual CSV upload loader for employee / culture signals.

Some of the most valuable signals -- employee growth, Glassdoor-style culture
ratings, product-moat assessments -- cannot be obtained from a free, key-less,
no-scrape source. Rather than scrape (which would violate the project's
constraints), FreeAlphaRadar accepts an **optional** CSV upload.

The CSV is entirely optional: when absent, every factor it would feed is simply
omitted from scoring with no degradation to the rest of the pipeline. A
template lives at ``manual_upload_template.csv`` in the repo root.

Expected columns (all optional except ``ticker``)::

    ticker, employee_growth, culture_score, product_moat_score, glassdoor_rating
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Union

import pandas as pd

from freealpharadar.utils import get_logger

logger = get_logger(__name__)

_NUMERIC_COLUMNS = (
    "employee_growth",
    "culture_score",
    "product_moat_score",
    "glassdoor_rating",
)


def load_manual_csv(
    source: Union[str, io.BytesIO, io.StringIO, None],
) -> Dict[str, Dict[str, Any]]:
    """Load the optional manual-signals CSV into a per-ticker dict.

    Args:
        source: A filesystem path, a file-like object (e.g. a Streamlit upload),
            or ``None``. When ``None`` an empty mapping is returned so callers
            can treat "no upload" uniformly.

    Returns:
        Mapping of upper-cased ticker to a dict of numeric signal columns. Rows
        missing a ticker are skipped; non-numeric cells become ``None``.
    """
    if source is None:
        return {}

    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse manual CSV: %s", exc)
        return {}

    if "ticker" not in df.columns:
        logger.warning("Manual CSV missing required 'ticker' column; ignoring.")
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker == "NAN":
            continue
        entry: Dict[str, Any] = {}
        for col in _NUMERIC_COLUMNS:
            if col in df.columns:
                entry[col] = _to_float(row.get(col))
        result[ticker] = entry

    logger.info("Loaded manual signals for %d tickers", len(result))
    return result


def _to_float(value: Any) -> Optional[float]:
    """Coerce a CSV cell to float, returning ``None`` on failure/blank."""
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
