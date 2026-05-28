"""PatentsView patent data fetcher.

Uses the free, key-less PatentsView API to collect, for a given company, the
number of granted patents over time, the assignee organisations, and a sample
of patent titles. These feed the "Disruption & Moat" factor group.

PatentsView migrated to ``search.patentsview.org``; we target that endpoint and
fall back to the legacy endpoint if needed. The free tier permits roughly 45
requests/minute, so results are cached aggressively.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from freealpharadar.config import (
    PATENTSVIEW_ENDPOINT,
    PATENTSVIEW_LEGACY_ENDPOINT,
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

        Args:
            key: Used here as a cache key (typically the ticker).
            company_name: The assignee organisation to search for. Falls back to
                ``key`` when not provided.
        """
        company_name: str = kwargs.get("company_name") or key
        try:
            patents = await self._query_modern(company_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("modern PatentsView query failed: %s; trying legacy", exc)
            patents = await self._query_legacy(company_name)

        return self._summarise(company_name, patents)

    async def _query_modern(self, company_name: str) -> List[Dict[str, Any]]:
        """Query the current ``search.patentsview.org`` endpoint."""
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
        data = await self._http_get_json(PATENTSVIEW_ENDPOINT, params=params)
        return data.get("patents", []) or []

    async def _query_legacy(self, company_name: str) -> List[Dict[str, Any]]:
        """Query the legacy ``api.patentsview.org`` endpoint."""
        query = {
            "q": {"_text_phrase": {"assignee_organization": company_name}},
            "f": ["patent_number", "patent_title", "patent_date"],
            "o": {"per_page": 100},
        }
        params = {
            "q": json.dumps(query["q"]),
            "f": json.dumps(query["f"]),
            "o": json.dumps(query["o"]),
        }
        data = await self._http_get_json(PATENTSVIEW_LEGACY_ENDPOINT, params=params)
        patents = data.get("patents", []) or []
        # Normalise legacy field names.
        for p in patents:
            if "patent_number" in p and "patent_id" not in p:
                p["patent_id"] = p["patent_number"]
        return patents

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
