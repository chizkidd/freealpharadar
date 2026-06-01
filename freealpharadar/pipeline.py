"""Data orchestration pipeline.

Ties the individual fetchers together into a single, cached ``CompanyData``
bundle per ticker, fetching every free source concurrently. The scoring engine
and the Streamlit app both consume the output of :func:`gather_company` /
:func:`gather_universe`.

Because every fetcher already falls back to cached data on failure, the
pipeline itself is robust offline: it simply collects whatever each fetcher can
provide along with any non-fatal warnings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from freealpharadar.config import settings
from freealpharadar.fetchers import (
    GDELTFetcher,
    PatentsViewFetcher,
    SECFetcher,
    YFinanceFetcher,
)
from freealpharadar.fetchers.base import FetchResult
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CompanyData:
    """All ingested data for a single company.

    Attributes:
        ticker: Upper-cased ticker symbol.
        name: Best-available company name.
        sector: Sector label (from yfinance) or ``"Unknown"``.
        market_cap: Market capitalisation in USD, if known.
        yfinance: yfinance payload (prices, statements, key metrics).
        sec: SEC payload (sections, facts, insider, flags).
        patents: PatentsView payload (counts, growth, titles).
        gdelt: GDELT payload (articles, average tone).
        manual: Manual CSV signals for this ticker.
        derived: ML-enrichment outputs (FinBERT sentiment, controversy, cluster).
        warnings: Non-fatal warnings collected from fetchers.
    """

    ticker: str
    name: str = ""
    sector: str = "Unknown"
    market_cap: Optional[float] = None
    yfinance: Dict[str, Any] = field(default_factory=dict)
    sec: Dict[str, Any] = field(default_factory=dict)
    patents: Dict[str, Any] = field(default_factory=dict)
    gdelt: Dict[str, Any] = field(default_factory=dict)
    manual: Dict[str, Any] = field(default_factory=dict)
    derived: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _company_name_from(yf_payload: Dict[str, Any], ticker: str) -> str:
    """Derive a display name from a yfinance payload."""
    info = yf_payload.get("info", {}) if yf_payload else {}
    return info.get("longName") or info.get("shortName") or ticker


async def gather_company(
    ticker: str,
    manual_signals: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> CompanyData:
    """Fetch and assemble all data for a single ticker.

    Args:
        ticker: The ticker symbol to gather.
        manual_signals: Optional per-ticker manual CSV signals.
        force_refresh: Bypass cache freshness checks across all fetchers.

    Returns:
        A populated :class:`CompanyData` instance.
    """
    ticker = ticker.upper()
    yf = YFinanceFetcher()
    sec = SECFetcher()
    patents = PatentsViewFetcher()
    gdelt = GDELTFetcher()

    # yfinance first so we can derive the company name for name-based queries.
    yf_res: FetchResult = await yf.fetch(ticker, force_refresh=force_refresh)
    yf_payload = yf_res.payload or {}
    name = _company_name_from(yf_payload, ticker)

    sec_res, pat_res, gdelt_res = await asyncio.gather(
        sec.fetch(ticker, force_refresh=force_refresh),
        patents.fetch(ticker, force_refresh=force_refresh, company_name=name),
        gdelt.fetch(ticker, force_refresh=force_refresh, company_name=name),
    )

    warnings: List[str] = [
        r.warning for r in (yf_res, sec_res, pat_res, gdelt_res) if r.warning
    ]

    metrics = yf_payload.get("key_metrics", {}) if yf_payload else {}
    sec_payload = sec_res.payload or {}

    # Market-cap fallback: when yfinance is blocked (common in CI), approximate
    # market cap from SEC shares outstanding × the latest available close
    # (yfinance or the Stooq fallback). Better a rough cap than "n/a".
    market_cap = metrics.get("market_cap")
    if market_cap is None:
        market_cap = _approx_market_cap(yf_payload, sec_payload)
        if market_cap is not None:
            metrics["market_cap"] = market_cap
            metrics["market_cap_approx"] = True

    return CompanyData(
        ticker=ticker,
        name=name,
        sector=(metrics.get("sector") or "Unknown"),
        market_cap=market_cap,
        yfinance=yf_payload,
        sec=sec_payload,
        patents=pat_res.payload or {},
        gdelt=gdelt_res.payload or {},
        manual=manual_signals or {},
        warnings=warnings,
    )


def _approx_market_cap(
    yf_payload: Dict[str, Any], sec_payload: Dict[str, Any]
) -> Optional[float]:
    """Approximate market cap as latest SEC shares outstanding × latest close.

    Returns ``None`` unless both a positive share count and a positive recent
    close are available.
    """
    shares_series = (sec_payload.get("facts", {}) or {}).get("shares") or []
    shares = None
    for point in reversed(shares_series):  # latest fiscal year first
        val = point.get("val")
        if val:
            shares = float(val)
            break
    if not shares or shares <= 0:
        return None

    history = yf_payload.get("history", []) if yf_payload else []
    last_close = None
    for row in reversed(history):
        close = row.get("close")
        if close:
            last_close = float(close)
            break
    if not last_close or last_close <= 0:
        return None
    return shares * last_close


async def gather_universe(
    tickers: List[str],
    manual_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    force_refresh: bool = False,
    progress_cb: Optional[Any] = None,
) -> List[CompanyData]:
    """Gather data for a list of tickers with bounded concurrency.

    Args:
        tickers: Ticker symbols to gather.
        manual_signals: Mapping of ticker -> manual CSV signal dict.
        force_refresh: Bypass cache freshness checks.
        progress_cb: Optional callable ``(done, total, ticker)`` invoked after
            each company completes; used by the Streamlit progress bar.

    Returns:
        A list of :class:`CompanyData`, one per input ticker (order preserved
        as best-effort; failures still yield a stub entry).
    """
    manual_signals = manual_signals or {}
    semaphore = asyncio.Semaphore(settings.request_concurrency)
    total = len(tickers)
    done = 0
    results: Dict[str, CompanyData] = {}

    async def _one(tk: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                results[tk] = await gather_company(
                    tk,
                    manual_signals=manual_signals.get(tk.upper()),
                    force_refresh=force_refresh,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to gather %s: %s", tk, exc)
                results[tk] = CompanyData(ticker=tk.upper(), warnings=[str(exc)])
            finally:
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, tk)

    await asyncio.gather(*(_one(tk) for tk in tickers))
    return [results[tk] for tk in tickers if tk in results]
