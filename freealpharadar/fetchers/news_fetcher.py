"""Yahoo Finance news fetcher.

Replaces the GDELT DOC API (which hard-throttles cloud IPs with HTTP 429,
making live runs slow or empty). ``yfinance``'s ``Ticker.news`` is free,
key-less, already in our dependency stack, and costs one call per ticker with
no aggressive rate limiting.

yfinance has shipped two payload schemas for ``.news`` over time; we normalise
both into a single shape. Yahoo does not provide a sentiment/tone score, so
tone is computed downstream from the headlines by the FinBERT/lexicon analyzer
(see :mod:`freealpharadar.ml.enrich` and the deep-dive News tab).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional

from freealpharadar.config import settings
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


class NewsFetcher(BaseFetcher):
    """Fetch recent news headlines for a ticker via ``yfinance.Ticker.news``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="news", ttl=settings.ttl.news, **kwargs)

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Fetch and normalise Yahoo news for ``key`` in a worker thread.

        Args:
            key: Cache key / ticker symbol.
            company_name: Display name (kept for parity with other fetchers).
        """
        company_name: str = kwargs.get("company_name") or key
        items = await asyncio.to_thread(self._fetch_sync, key)
        return self._summarise(company_name, items)

    def _fetch_sync(self, ticker: str) -> List[Dict[str, Any]]:
        """Synchronous yfinance news pull, executed off the event loop."""
        import yfinance as yf  # imported lazily so the package imports offline

        tk = yf.Ticker(ticker)
        return list(tk.news or [])

    @staticmethod
    def _summarise(company_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalise raw yfinance news items into a stable article list."""
        articles: List[Dict[str, Any]] = []
        for item in items or []:
            art = _normalise_item(item)
            if art and art.get("title"):
                articles.append(art)
        return {
            "company_name": company_name,
            "article_count": len(articles),
            "articles": articles,
        }


def _normalise_item(item: Any) -> Optional[Dict[str, Any]]:
    """Map a yfinance news item (either schema) to ``{title,url,publisher,published}``."""
    if not isinstance(item, dict):
        return None
    # Newer schema (yfinance >= ~0.2.40) wraps the fields under "content".
    content = item.get("content")
    if isinstance(content, dict):
        provider = (content.get("provider") or {}).get("displayName")
        url = (content.get("canonicalUrl") or {}).get("url") or (
            content.get("clickThroughUrl") or {}
        ).get("url")
        return {
            "title": content.get("title"),
            "url": url,
            "publisher": provider,
            "published": content.get("pubDate") or content.get("displayTime"),
        }
    # Legacy flat schema.
    return {
        "title": item.get("title"),
        "url": item.get("link"),
        "publisher": item.get("publisher"),
        "published": _epoch_to_iso(item.get("providerPublishTime")),
    }


def _epoch_to_iso(ts: Any) -> Optional[str]:
    """Convert a Unix epoch (seconds) to an ISO-8601 string; ``None`` if absent."""
    if not ts:
        return None
    try:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None
