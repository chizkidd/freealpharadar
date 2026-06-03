"""Provider-agnostic patent data fetcher.

Collects, for a given company, the number of granted patents over time and a
sample of patent titles, which feed the "Disruption & Moat" factor group.

The fetcher supports two free providers and picks whichever is configured:

* **PatentsView** (``search.patentsview.org``) -- lowest-friction for a US
  universe. Set ``FAR_PATENTSVIEW_API_KEY`` (free key request at
  https://patentsview.org/apis/keyrequest).
* **Lens.org** (``api.lens.org``) -- global coverage. Set ``FAR_LENS_API_TOKEN``.

If **neither** token is set the fetcher returns an empty result and the app
simply shows no patent data, so the zero-config / zero-key default is preserved.
PatentsView takes precedence when both are set. Results are cached aggressively.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from freealpharadar.config import (
    LENS_API_TOKEN,
    LENS_ENDPOINT,
    PATENTSVIEW_API_KEY,
    PATENTSVIEW_ENDPOINT,
    settings,
)
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


class PatentFetcher(BaseFetcher):
    """Fetch patent counts and titles for a company from PatentsView or Lens."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="patents", ttl=settings.ttl.patents, **kwargs)

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Query the configured provider for patents assigned to ``company_name``.

        Selects PatentsView when ``FAR_PATENTSVIEW_API_KEY`` is set, else Lens
        when ``FAR_LENS_API_TOKEN`` is set, else returns an empty result so the
        rest of the pipeline is unaffected (zero-key default).

        Args:
            key: Used here as a cache key (typically the ticker).
            company_name: The assignee organisation to search for. Falls back to
                ``key`` when not provided.
        """
        company_name: str = kwargs.get("company_name") or key
        if PATENTSVIEW_API_KEY:
            patents = await self._query_patentsview(company_name)
        elif LENS_API_TOKEN:
            patents = await self._query_lens(company_name)
        else:
            logger.debug(
                "No patent provider configured (FAR_PATENTSVIEW_API_KEY / "
                "FAR_LENS_API_TOKEN); skipping patents for %s.",
                company_name,
            )
            return self._summarise(company_name, [])
        return self._summarise(company_name, patents)

    async def _query_patentsview(self, company_name: str) -> List[Dict[str, Any]]:
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

    async def _query_lens(self, company_name: str) -> List[Dict[str, Any]]:
        """Query the Lens.org patent search API (needs a Bearer token).

        Normalises each Lens record to the same ``{patent_title, patent_date}``
        shape that :meth:`_summarise` consumes, so downstream scoring is
        provider-agnostic.
        """
        body = {
            "query": {"match_phrase": {"applicant.name": company_name}},
            "size": 100,
            "include": ["biblio.invention_title.text", "date_published"],
        }
        headers = {
            "Authorization": f"Bearer {LENS_API_TOKEN}",
            "Content-Type": "application/json",
        }
        data = await self._http_post_json(
            LENS_ENDPOINT, json_body=body, headers=headers
        )
        out: List[Dict[str, Any]] = []
        for hit in data.get("data", []) or []:
            titles = (hit.get("biblio", {}) or {}).get("invention_title", []) or []
            title = titles[0].get("text") if titles else None
            out.append(
                {"patent_title": title, "patent_date": hit.get("date_published")}
            )
        return out

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


# Backwards-compatible alias: the class was provider-specific historically.
PatentsViewFetcher = PatentFetcher


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
