"""PatentsView patent data fetcher.

Collects, for a given company, the number of granted patents over time and a
sample of patent titles, which feed the "Disruption & Moat" factor group.

PatentsView's current Search API (``search.patentsview.org``) requires a
**free** API key (the legacy key-less endpoint was retired). The key is
optional here: set ``FAR_PATENTSVIEW_API_KEY`` to enable patents; without it the
fetcher returns an empty result and the app simply shows no patent data, so the
zero-config default is preserved. Results are cached aggressively (the free tier
allows ~45 requests/minute).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from freealpharadar.config import (
    PATENTSVIEW_API_KEY,
    PATENTSVIEW_ENDPOINT,
    settings,
)
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


class PatentsViewFetcher(BaseFetcher):
    """Fetch patent counts, assignees and titles for a company name."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="patentsview", ttl=settings.ttl.patents, **kwargs)

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Query PatentsView for patents assigned to the company named ``key``.

        The current PatentsView Search API requires a free API key. When it is
        not configured (``FAR_PATENTSVIEW_API_KEY``), we skip the call and return
        an empty result so the rest of the pipeline is unaffected.

        Args:
            key: Used here as a cache key (typically the ticker).
            company_name: The assignee organisation to search for. Falls back to
                ``key`` when not provided.
        """
        company_name: str = kwargs.get("company_name") or key
        if not PATENTSVIEW_API_KEY:
            logger.debug(
                "PatentsView API key not set (FAR_PATENTSVIEW_API_KEY); "
                "skipping patents for %s.",
                company_name,
            )
            return self._summarise(company_name, [])
        patents = await self._query_modern(company_name)
        return self._summarise(company_name, patents)

    async def _query_modern(self, company_name: str) -> List[Dict[str, Any]]:
        """Query the current ``search.patentsview.org`` endpoint (needs a key)."""
        query = {
            "q": {"_text_phrase": {"assignees.assignee_organization": company_name}},
            "f": ["patent_id", "patent_title", "patent_date"],
            "o": {"size": 100},
        }
        params = {
            "q": json.dumps(query["q"]),
            "f": json.dumps(query["f"]),
            "o": json.dumps(query["o"]),
        }
        headers = {"X-Api-Key": PATENTSVIEW_API_KEY}
        data = await self._http_get_json(
            PATENTSVIEW_ENDPOINT, params=params, headers=headers
        )
        return data.get("patents", []) or []

    @staticmethod
    def _summarise(company_name: str, patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate raw patents into counts-by-year and top assignees."""
        by_year: Counter = Counter()
        titles: List[str] = []
        for p in patents:
            date = p.get("patent_date") or ""
            year = date[:4] if isinstance(date, str) and len(date) >= 4 else None
            if year and year.isdigit():
                by_year[year] += 1
            title = p.get("patent_title")
            if title:
                titles.append(title)

        counts = [{"year": y, "count": by_year[y]} for y in sorted(by_year)]
        years_sorted = [c["count"] for c in counts]
        growth_rate = _trailing_growth(years_sorted)

        return {
            "company_name": company_name,
            "total_patents": len(patents),
            "counts_by_year": counts,
            "patent_growth_rate": growth_rate,
            "sample_titles": titles[:20],
        }


def _trailing_growth(yearly_counts: List[int]) -> float:
    """Compute a simple recent-vs-prior patent growth ratio.

    Compares the sum of the two most recent years against the prior two. A
    value of ``0`` means flat/unknown; ``1.0`` means a doubling.
    """
    if len(yearly_counts) < 4:
        return 0.0
    recent = sum(yearly_counts[-2:])
    prior = sum(yearly_counts[-4:-2])
    if prior == 0:
        return 1.0 if recent > 0 else 0.0
    return (recent - prior) / prior
