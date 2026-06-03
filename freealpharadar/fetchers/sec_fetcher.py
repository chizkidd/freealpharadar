"""SEC EDGAR fetcher.

Scrapes EDGAR directly -- no API key required -- to obtain:

* the ticker -> CIK mapping (``company_tickers.json``);
* recent filing metadata and the latest 10-K/10-Q document text, from which we
  extract **risk factors**, **MD&A** and the **business description**;
* **insider transactions** (Form 4) summarised from the submissions feed;
* selected XBRL company facts (revenue, R&D, etc.) via ``companyfacts``.

Everything is fetched from the public ``data.sec.gov`` JSON endpoints and the
EDGAR archives -- lightweight and equally key-less. All requests send a
descriptive User-Agent as the SEC fair-access policy requests.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from freealpharadar.config import (
    SEC_COMPANY_FACTS,
    SEC_SUBMISSIONS,
    SEC_TICKER_MAP,
    SEC_USER_AGENT,
    settings,
)
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Keywords used for lightweight qualitative flags extracted from filing text.
_FOUNDER_KEYWORDS = (
    "founder",
    "co-founder",
    "founded the company",
    "our founder",
)
_REGULATORY_KEYWORDS = (
    "regulation",
    "regulatory",
    "compliance",
    "fda",
    "sec investigation",
    "antitrust",
    "gdpr",
    "export control",
)
_KEY_PERSON_KEYWORDS = (
    "key person",
    "key personnel",
    "dependent on",
    "loss of",
    "chief executive officer",
)
_CONCENTRATION_KEYWORDS = (
    "customer concentration",
    "limited number of customers",
    "significant portion of our revenue",
    "one customer",
    "few customers",
)


class SECFetcher(BaseFetcher):
    """Fetch SEC EDGAR filings, facts and insider data for a ticker."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="sec", ttl=settings.ttl.sec, **kwargs)
        self._ticker_map: Optional[Dict[str, str]] = None

    async def _ensure_ticker_map(self) -> Dict[str, str]:
        """Load and cache the EDGAR ticker -> zero-padded CIK map."""
        if self._ticker_map is not None:
            return self._ticker_map
        cached = self.db.cached_payload("sec:tickermap", "all")
        if cached:
            self._ticker_map = cached
            return cached
        data = await self._http_get_json(SEC_TICKER_MAP, headers=_HEADERS)
        mapping: Dict[str, str] = {}
        for row in data.values():
            mapping[str(row["ticker"]).upper()] = str(row["cik_str"]).zfill(10)
        self.db.set_cache("sec:tickermap", "all", mapping)
        self._ticker_map = mapping
        return mapping

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Assemble an SEC data bundle for ``key`` (a ticker)."""
        ticker = key.upper()
        mapping = await self._ensure_ticker_map()
        cik = mapping.get(ticker)
        if cik is None:
            raise ValueError(f"No CIK found for ticker {ticker}")

        submissions = await self._http_get_json(
            SEC_SUBMISSIONS.format(cik=cik), headers=_HEADERS
        )
        facts = await self._fetch_company_facts(cik)
        insider = self._summarise_insider(submissions)
        text_sections = await self._fetch_filing_text(submissions, cik)

        business_text = text_sections.get("business", "")
        risk_text = text_sections.get("risk_factors", "")

        return {
            "ticker": ticker,
            "cik": cik,
            "company_name": submissions.get("name"),
            "sic_description": submissions.get("sicDescription"),
            "sections": text_sections,
            "facts": facts,
            "insider_transactions": insider,
            "flags": {
                "founder_led": _contains_any(business_text, _FOUNDER_KEYWORDS),
                "key_person_dependency": _contains_any(risk_text, _KEY_PERSON_KEYWORDS),
                "customer_concentration": _contains_any(
                    risk_text, _CONCENTRATION_KEYWORDS
                ),
                "regulatory_risk_count": _count_any(risk_text, _REGULATORY_KEYWORDS),
            },
        }

    async def _fetch_company_facts(self, cik: str) -> Dict[str, Any]:
        """Pull selected XBRL company facts (USD-GAAP concepts)."""
        try:
            data = await self._http_get_json(
                SEC_COMPANY_FACTS.format(cik=cik), headers=_HEADERS
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("companyfacts failed for CIK %s: %s", cik, exc)
            return {}

        facts = data.get("facts", {})
        gaap = facts.get("us-gaap", {})
        # Revenue is tagged under several us-gaap concepts depending on the
        # filer/era; capture each so the CAGR/margin factors can fall back in
        # priority order and mature names don't end up with an empty series.
        wanted = {
            "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "revenue_alt": "Revenues",
            "revenue_alt2": "SalesRevenueNet",
            "revenue_alt3": "RevenueFromContractWithCustomerIncludingAssessedTax",
            "rnd": "ResearchAndDevelopmentExpense",
            "gross_profit": "GrossProfit",
            "net_income": "NetIncomeLoss",
            "assets": "Assets",
            "liabilities": "Liabilities",
        }
        out: Dict[str, Any] = {}
        for label, concept in wanted.items():
            series = _annual_series(gaap.get(concept))
            if series:
                out[label] = series

        # Shares outstanding (for a market-cap fallback when yfinance is blocked):
        # the cover-page count lives under the ``dei`` taxonomy, with a us-gaap
        # backup. Used with a last price to approximate market cap.
        shares = _annual_series(
            facts.get("dei", {}).get("EntityCommonStockSharesOutstanding")
        ) or _annual_series(gaap.get("CommonStockSharesOutstanding"))
        if shares:
            out["shares"] = shares
        return out

    async def _fetch_filing_text(
        self, submissions: Dict[str, Any], cik: str
    ) -> Dict[str, str]:
        """Download the latest 10-K/10-Q and extract qualitative sections."""
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession = recent.get("accessionNumber", [])
        primary = recent.get("primaryDocument", [])

        target_idx: Optional[int] = None
        for i, form in enumerate(forms):
            if form in ("10-K", "10-Q"):
                target_idx = i
                break
        if target_idx is None:
            return {}

        acc_nodash = accession[target_idx].replace("-", "")
        doc = primary[target_idx]
        cik_int = int(cik)
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/" f"{acc_nodash}/{doc}"
        )
        try:
            html = await self._http_get_text(url, headers=_HEADERS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("filing fetch failed (%s): %s", url, exc)
            return {}

        text = _html_to_text(html)
        return {
            "form": forms[target_idx],
            "filing_url": url,
            "business": _extract_section(
                text, r"item\s*1\.?\s*business", (r"item\s*1a\b",), max_len=6000
            ),
            "risk_factors": _extract_section(
                text, r"item\s*1a\b", (r"item\s*1b\b", r"item\s*2\b"), max_len=8000
            ),
            "mdna": _extract_section(
                text,
                r"item\s*7\.?\s*management",
                (r"item\s*7a\b", r"item\s*8\b"),
                max_len=8000,
            ),
        }

    @staticmethod
    def _summarise_insider(submissions: Dict[str, Any]) -> Dict[str, Any]:
        """Count recent Form 4 (insider transaction) filings."""
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        form4 = [
            {"date": dates[i] if i < len(dates) else None}
            for i, f in enumerate(forms)
            if f == "4"
        ]
        return {"form4_count": len(form4), "recent": form4[:25]}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _html_to_text(html: str) -> str:
    """Strip HTML to plain text using BeautifulSoup when available."""
    import warnings

    try:
        from bs4 import BeautifulSoup

        # Modern 10-Ks are inline-XBRL (XHTML); parsing them with the HTML parser
        # is intentional and fine here, so silence the noisy XMLParsedAsHTMLWarning.
        with warnings.catch_warnings():
            try:
                from bs4 import XMLParsedAsHTMLWarning

                warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            except Exception:  # noqa: BLE001
                pass
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
    except Exception:  # noqa: BLE001
        text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _extract_section(
    text: str, start_pattern: str, end_patterns: tuple, max_len: int = 8000
) -> str:
    """Extract a filing section between regex item markers, robustly.

    Modern (inline-XBRL) 10-Ks repeat "Item 1A" in the table of contents, so a
    naive first-match grab returns a one-line TOC stub. Instead we consider
    *every* start-marker occurrence, take the text up to the next end marker for
    each, and keep the **longest** candidate -- which is the real section body
    rather than a TOC entry. All matching is case-insensitive and tolerant of
    whitespace/punctuation between "item" and the number.

    Args:
        text: Whitespace-normalised plain text of the filing.
        start_pattern: Regex marking the section start (e.g. ``r"item\\s*1a"``).
        end_patterns: Regexes marking possible section ends; the earliest match
            after the start wins.
        max_len: Hard cap on the returned section length.

    Returns:
        The longest matching section (trimmed/capped), or ``""`` if not found.
    """
    starts = [m.start() for m in re.finditer(start_pattern, text, re.IGNORECASE)]
    if not starts:
        return ""
    end_re = re.compile("|".join(end_patterns), re.IGNORECASE) if end_patterns else None

    best = ""
    for s in starts:
        if end_re is not None:
            m = end_re.search(text, s + 1)
            e = m.start() if m else min(s + max_len, len(text))
        else:
            e = min(s + max_len, len(text))
        candidate = text[s:e].strip()
        if len(candidate) > len(best):
            best = candidate
    return best[:max_len]


def _contains_any(text: str, keywords: tuple) -> bool:
    """Whether ``text`` (case-insensitively) contains any keyword."""
    low = text.lower()
    return any(kw in low for kw in keywords)


def _count_any(text: str, keywords: tuple) -> int:
    """Count total occurrences of any keyword in ``text``."""
    low = text.lower()
    return sum(low.count(kw) for kw in keywords)


def _annual_series(concept: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract an annual (FY) value series from an XBRL concept block."""
    if not concept:
        return []
    units = concept.get("units", {})
    points: List[Dict[str, Any]] = []
    for unit_vals in units.values():
        for item in unit_vals:
            if item.get("form") in ("10-K", "20-F") and item.get("fp") == "FY":
                points.append(
                    {
                        "fy": item.get("fy"),
                        "end": item.get("end"),
                        "val": item.get("val"),
                    }
                )
    # Deduplicate by fiscal year, keeping the latest filed value.
    by_fy: Dict[Any, Dict[str, Any]] = {}
    for p in points:
        by_fy[p["fy"]] = p
    return sorted(by_fy.values(), key=lambda p: p.get("fy") or 0)
