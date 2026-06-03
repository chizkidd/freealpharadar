"""High-level orchestration service.

Provides the single entry point used by both the Streamlit app and the
``run_scorer`` batch job: gather data for a universe, enrich it with ML
signals, score it, and persist the results to SQLite. Also offers a synchronous
convenience wrapper so callers needn't manage an event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from freealpharadar.database import get_db
from freealpharadar.ml.enrich import enrich_companies
from freealpharadar.pipeline import CompanyData, gather_universe
from freealpharadar.scoring import ScoreResult, ScoringEngine
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    """Bundle returned by the scoring pipeline.

    Attributes:
        results: Scored companies, sorted by descending score.
        companies: The enriched raw company bundles (for deep dives).
        warnings: De-duplicated data warnings gathered across the universe.
    """

    results: List[ScoreResult] = field(default_factory=list)
    companies: List[CompanyData] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


async def run_pipeline_async(
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
    manual_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    force_refresh: bool = False,
    run_ml: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    persist: bool = True,
    fetch_news: bool = True,
) -> PipelineOutput:
    """Run the full ingest -> enrich -> score pipeline asynchronously.

    Args:
        tickers: Universe of ticker symbols.
        weights: Optional factor-weight overrides.
        manual_signals: Optional manual CSV signals keyed by ticker.
        force_refresh: Bypass cache freshness checks during ingestion.
        run_ml: Whether to run FinBERT/clustering enrichment.
        progress_cb: Progress callback forwarded to ingestion.
        persist: Whether to write score snapshots to SQLite.
        fetch_news: Fetch Yahoo Finance news for every name (default ``True``).
            It is a cheap, key-less call per ticker, so it runs inline; the
            discovery job passes ``False`` to skip it for speed.

    Returns:
        A :class:`PipelineOutput`.
    """
    companies = await gather_universe(
        tickers,
        manual_signals=manual_signals,
        force_refresh=force_refresh,
        progress_cb=progress_cb,
        fetch_news=fetch_news,
    )

    if run_ml:
        try:
            enrich_companies(companies)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML enrichment failed (continuing rule-based): %s", exc)

    engine = ScoringEngine(weights=weights)
    results = engine.score_universe(companies)

    if persist:
        db = get_db()
        for res in results:
            db.save_score(
                res.ticker,
                {
                    "score": res.score,
                    "raw_composite": res.raw_composite,
                    "sector": res.sector,
                    "market_cap": res.market_cap,
                    "group_scores": res.group_scores,
                    "top_factors": [
                        {"name": c.name, "contribution": c.contribution}
                        for c in res.contributions[:5]
                    ],
                },
            )

    warnings = sorted({w for c in companies for w in c.warnings})
    return PipelineOutput(results=results, companies=companies, warnings=warnings)


def fetch_company_news(
    ticker: str, company_name: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch Yahoo Finance news for a single company on demand.

    The universe run already fetches news inline, but this is a cheap fallback
    for the deep-dive News tab when a company's bundle has no news cached yet
    (e.g. seeded from an older snapshot). The result is cached in SQLite, so
    reopening the tab is instant. Returns an empty dict on failure (the caller
    shows a gentle "no news yet" note rather than a warning).
    """
    from freealpharadar.fetchers import NewsFetcher

    async def _go() -> Dict[str, Any]:
        res = await NewsFetcher().fetch(ticker, company_name=company_name)
        if res.warning:
            logger.debug("news (on-demand): %s", res.warning)
        return res.payload or {}

    try:
        return asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001
        logger.debug("on-demand news fetch failed for %s: %s", ticker, exc)
        return {}


def run_pipeline(
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
    manual_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    force_refresh: bool = False,
    run_ml: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    persist: bool = True,
    fetch_news: bool = True,
) -> PipelineOutput:
    """Synchronous wrapper around :func:`run_pipeline_async`.

    Safe to call from Streamlit (which runs in a thread without an event loop).
    """
    return asyncio.run(
        run_pipeline_async(
            tickers,
            weights=weights,
            manual_signals=manual_signals,
            force_refresh=force_refresh,
            run_ml=run_ml,
            progress_cb=progress_cb,
            persist=persist,
            fetch_news=fetch_news,
        )
    )
