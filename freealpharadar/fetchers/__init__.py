"""Data ingestion layer.

Every fetcher follows the same contract:

* It is **asynchronous** (``async def fetch(...)``).
* It **caches** all results in SQLite with a per-source TTL.
* On any network failure -- or when running offline -- it transparently
  **falls back to the last cached payload** and records a non-fatal warning so
  the UI can surface it without breaking.

Public fetchers:

* :class:`~freealpharadar.fetchers.yfinance_fetcher.YFinanceFetcher`
* :class:`~freealpharadar.fetchers.sec_fetcher.SECFetcher`
* :class:`~freealpharadar.fetchers.patents_fetcher.PatentFetcher`
* :class:`~freealpharadar.fetchers.news_fetcher.NewsFetcher`
* :func:`~freealpharadar.fetchers.manual_csv.load_manual_csv`
"""

from __future__ import annotations

from freealpharadar.fetchers.base import BaseFetcher, FetchResult
from freealpharadar.fetchers.manual_csv import load_manual_csv
from freealpharadar.fetchers.news_fetcher import NewsFetcher
from freealpharadar.fetchers.patents_fetcher import (
    PatentFetcher,
    PatentsViewFetcher,
)
from freealpharadar.fetchers.sec_fetcher import SECFetcher
from freealpharadar.fetchers.yfinance_fetcher import YFinanceFetcher

__all__ = [
    "BaseFetcher",
    "FetchResult",
    "YFinanceFetcher",
    "SECFetcher",
    "PatentFetcher",
    "PatentsViewFetcher",
    "NewsFetcher",
    "load_manual_csv",
]
