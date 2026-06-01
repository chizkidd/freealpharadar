"""Download SEC Financial Statement Data Set quarterly ZIPs.

Synchronous and dependency-light (``requests``), with on-disk caching and
resume: a quarter already present in the cache is never re-downloaded. Sends the
configured SEC ``User-Agent`` per the fair-access policy.
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import List, Optional

import requests

from freealpharadar.config import SEC_USER_AGENT
from freealpharadar.utils import get_logger
from freealpharadar.warehouse.config import (
    DATASET_URL_TEMPLATE,
    FIRST_YEAR,
    ZIP_CACHE_DIR,
)

logger = get_logger(__name__)

_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def available_quarters(since_year: int = FIRST_YEAR) -> List[str]:
    """List dataset quarter ids (e.g. ``"2023q1"``) from ``since_year`` to now.

    Args:
        since_year: First calendar year to include (clamped to the program
            start, 2009).

    Returns:
        Quarter ids in chronological order, up to the most recently completed
        quarter.
    """
    start = max(since_year, FIRST_YEAR)
    today = _dt.date.today()
    quarters: List[str] = []
    for year in range(start, today.year + 1):
        for q in range(1, 5):
            # Skip future/in-progress quarters (data lands ~1 quarter late).
            quarter_start_month = (q - 1) * 3 + 1
            if _dt.date(year, quarter_start_month, 1) > today:
                continue
            quarters.append(f"{year}q{q}")
    return quarters


def download_quarter(
    quarter: str, dest_dir: Optional[Path] = None, throttle: float = 0.5
) -> Optional[Path]:
    """Download one quarter's ZIP, skipping it if already cached.

    Args:
        quarter: Quarter id such as ``"2023q1"``.
        dest_dir: Cache directory; defaults to :data:`ZIP_CACHE_DIR`.
        throttle: Seconds to sleep after a network fetch (politeness).

    Returns:
        Path to the cached ZIP, or ``None`` if the download failed.
    """
    dest_dir = Path(dest_dir or ZIP_CACHE_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{quarter}.zip"
    if path.exists() and path.stat().st_size > 0:
        logger.debug("Quarter %s already cached.", quarter)
        return path

    url = DATASET_URL_TEMPLATE.format(quarter=quarter)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        logger.info("Downloaded %s (%.1f MB).", quarter, len(resp.content) / 1e6)
        time.sleep(throttle)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to download %s: %s", quarter, exc)
        return None


def download_all(
    since_year: int = FIRST_YEAR, dest_dir: Optional[Path] = None
) -> List[Path]:
    """Download every quarter from ``since_year`` to now (cached/resumable)."""
    paths: List[Path] = []
    for quarter in available_quarters(since_year):
        p = download_quarter(quarter, dest_dir=dest_dir)
        if p is not None:
            paths.append(p)
    return paths
