"""GDELT news & sentiment fetcher.

Uses the free, key-less GDELT 2.0 Doc API to retrieve recent news articles
mentioning a company, along with GDELT's "tone" sentiment metric. Articles and
per-article tone are cached so that the dashboard's news feed and the
"controversy score" remain available offline.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
import weakref
from typing import Any, Dict, List

from freealpharadar.config import GDELT_DOC_ENDPOINT, settings
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

# GDELT rate-limits aggressively (HTTP 429) when called concurrently. Serialise
# all GDELT requests across the process and space them out so a 32-name batch
# doesn't trip the limiter. Tunable via FAR_GDELT_INTERVAL (seconds).
_GDELT_MIN_INTERVAL = float(os.environ.get("FAR_GDELT_INTERVAL", "2.0"))
_gdelt_last_call = 0.0
# An asyncio.Lock binds to the event loop that is running when it is created.
# Streamlit starts a fresh loop on every rerun, so a module-level lock would
# raise "bound to a different event loop". Cache one lock per running loop, with
# weak keys so locks for finished loops are garbage-collected automatically.
_gdelt_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _get_gdelt_lock() -> asyncio.Lock:
    """Return a lock bound to the currently running event loop."""
    loop = asyncio.get_running_loop()
    lock = _gdelt_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _gdelt_locks[loop] = lock
    return lock


async def _gdelt_throttle() -> None:
    """Block until at least ``_GDELT_MIN_INTERVAL`` has passed since the last call."""
    global _gdelt_last_call
    async with _get_gdelt_lock():
        wait = _GDELT_MIN_INTERVAL - (time.monotonic() - _gdelt_last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _gdelt_last_call = time.monotonic()


class GDELTFetcher(BaseFetcher):
    """Fetch recent news articles and aggregate tone via the GDELT Doc API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="gdelt", ttl=settings.ttl.news, **kwargs)

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Query GDELT for articles about the company named ``key``.

        Args:
            key: Cache key (typically a ticker).
            company_name: Search phrase; falls back to ``key``.
        """
        company_name: str = kwargs.get("company_name") or key
        params = {
            "query": f'"{company_name}"',
            "mode": "ToneChart",
            "format": "json",
            "timespan": "3m",
        }
        # The ToneChart mode returns a tone histogram; ArtList returns articles.
        # Throttle both calls so concurrent tickers don't trip GDELT's limiter.
        await _gdelt_throttle()
        tone_data = await self._http_get_json(GDELT_DOC_ENDPOINT, params=params)

        art_params = {
            "query": f'"{company_name}"',
            "mode": "ArtList",
            "format": "json",
            "maxrecords": "50",
            "timespan": "3m",
            "sort": "datedesc",
        }
        await _gdelt_throttle()
        art_data = await self._http_get_json(GDELT_DOC_ENDPOINT, params=art_params)

        return self._summarise(company_name, tone_data, art_data)

    @staticmethod
    def _summarise(
        company_name: str,
        tone_data: Dict[str, Any],
        art_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate GDELT responses into articles + average tone."""
        articles: List[Dict[str, Any]] = []
        for art in art_data.get("articles", []) or []:
            articles.append(
                {
                    "title": art.get("title"),
                    "url": art.get("url"),
                    "domain": art.get("domain"),
                    "seendate": art.get("seendate"),
                    "tone": _safe_float(art.get("tone")),
                    "language": art.get("language"),
                }
            )

        # ToneChart returns bins of {bin: tone, count}. Compute a weighted mean.
        bins = tone_data.get("tonechart", []) or []
        weighted_sum = 0.0
        total = 0
        for b in bins:
            tone = _safe_float(b.get("bin"))
            count = b.get("count", 0) or 0
            if tone is not None:
                weighted_sum += tone * count
                total += count
        avg_tone = weighted_sum / total if total else _avg_article_tone(articles)

        return {
            "company_name": company_name,
            "article_count": len(articles),
            "avg_tone": avg_tone,
            "articles": articles,
        }


def _avg_article_tone(articles: List[Dict[str, Any]]) -> float:
    """Mean tone across articles that report one; ``0.0`` if none."""
    tones = [a["tone"] for a in articles if a.get("tone") is not None]
    return statistics.mean(tones) if tones else 0.0


def _safe_float(value: Any) -> Any:
    """Coerce to float or return ``None``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
