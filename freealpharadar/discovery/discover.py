"""Stage 2 + promotion: full-score the shortlist and update the app universe.

Runs the existing 35-factor pipeline on the Stage-1 shortlist, applies a final
market-cap "under-the-radar" ceiling, ranks, and promotes the top-N into the
app's ``universe.txt`` and prewarm snapshot, writing a dated discovery report.

Building the warehouse and running Stage 2 require network access (sec.gov,
yfinance, ...), so this is an offline/CI job, not part of the live app.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from freealpharadar.config import UNIVERSE_FILE
from freealpharadar.discovery.screen import ScreenConfig, screen_candidates
from freealpharadar.utils import get_logger
from freealpharadar.warehouse.store import WarehouseStore

logger = get_logger(__name__)

DISCOVERIES_DIR: Path = Path(__file__).resolve().parents[2] / "discoveries"


@dataclass
class DiscoveredName:
    """One promoted name."""

    rank: int
    ticker: str
    name: str
    score: float
    market_cap: Optional[float]


@dataclass
class DiscoveryResult:
    """Outcome of a discovery run.

    Attributes:
        names: Ranked top-N discovered names (best first).
        candidates_screened: How many Stage-1 candidates were scored.
        universe_path: Where the refreshed ``universe.txt`` was written (or None).
        report_path: Where the markdown report was written (or None).
    """

    names: List[DiscoveredName] = field(default_factory=list)
    candidates_screened: int = 0
    universe_path: Optional[str] = None
    report_path: Optional[str] = None


def run_discovery(
    store: Optional[WarehouseStore] = None,
    screen_cfg: Optional[ScreenConfig] = None,
    top_n: int = 10,
    max_market_cap: float = 20e9,
    run_ml: bool = True,
    write_outputs: bool = True,
    min_promote_coverage: float = 0.5,
    universe_path: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    pipeline_fn: Optional[Callable[..., Any]] = None,
) -> DiscoveryResult:
    """Run the full discovery funnel and (optionally) promote the winners.

    Args:
        store: Warehouse to screen; defaults to the configured store.
        screen_cfg: Stage-1 screen configuration.
        top_n: Number of names to promote.
        max_market_cap: Final "under-the-radar" ceiling (USD); names above it
            are dropped. Names with unknown cap are kept.
        run_ml: Whether Stage 2 runs FinBERT/clustering enrichment.
        write_outputs: When ``True`` rewrite ``universe.txt``, regenerate the
            prewarm snapshot, and write the markdown report.
        min_promote_coverage: Quality gate -- the promoted names must have a
            market cap on at least this fraction, otherwise ``universe.txt`` and
            the snapshot are left untouched (the report is still written). This
            stops a Stage-2 run where the live data was blocked from replacing a
            good universe with empty rows.
        universe_path: Override for the universe file (tests).
        report_dir: Override for the report directory (tests).
        snapshot_path: Override for the prewarm snapshot path (tests).
        pipeline_fn: Injectable scorer (defaults to
            :func:`freealpharadar.service.run_pipeline`); tests pass a stub.

    Returns:
        A :class:`DiscoveryResult`.
    """
    store = store or WarehouseStore()
    screen_cfg = screen_cfg or ScreenConfig()

    candidates = screen_candidates(store, screen_cfg)
    if candidates.empty:
        logger.warning("Screen returned no candidates.")
        return DiscoveryResult()
    tickers = candidates["ticker"].tolist()

    if pipeline_fn is None:
        from freealpharadar.service import run_pipeline as pipeline_fn  # lazy

    output = pipeline_fn(tickers, run_ml=run_ml, force_refresh=True, persist=True)

    eligible = [
        r
        for r in output.results
        if r.market_cap is None or r.market_cap <= max_market_cap
    ]
    eligible.sort(key=lambda r: r.score, reverse=True)
    top = eligible[:top_n]

    names = [
        DiscoveredName(i + 1, r.ticker, r.name, r.score, r.market_cap)
        for i, r in enumerate(top)
    ]
    result = DiscoveryResult(names=names, candidates_screened=len(tickers))

    if write_outputs and names:
        # Always write the report (useful diagnostics even on a thin run)...
        result.report_path = str(_write_report(result, report_dir))
        # ...but only promote into the live universe/snapshot when the result
        # is actually populated, so a blocked-data run can't ship empty rows.
        coverage = sum(1 for n in names if n.market_cap) / len(names)
        if coverage >= min_promote_coverage:
            result.universe_path = str(_write_universe(names, universe_path))
            _export_snapshot(
                [n.ticker for n in names], snapshot_path, min_promote_coverage
            )
        else:
            logger.warning(
                "Discovered names have only %.0f%% market-cap coverage (< %.0f%%); "
                "keeping the existing universe.txt/snapshot. See the report.",
                coverage * 100,
                min_promote_coverage * 100,
            )

    return result


def _write_universe(names: List[DiscoveredName], path: Optional[Path] = None) -> Path:
    """Overwrite ``universe.txt`` with the auto-discovered tickers.

    Each line is ``TICKER  # Company Name`` so the file stays readable; the
    universe loader strips the inline ``#`` comment.
    """
    path = Path(path or UNIVERSE_FILE)
    today = _dt.date.today().isoformat()
    header = [
        "# FreeAlphaRadar — AUTO-DISCOVERED universe.",
        f"# Generated by `python -m freealpharadar.discovery run` on {today}.",
        "# Ranked best -> worst from a market-wide scan of SEC filers.",
        "# Edit freely; a future discovery run will overwrite this file.",
        "",
    ]
    lines = []
    for n in names:
        if n.name and n.name != n.ticker:
            lines.append(f"{n.ticker}  # {n.name}")
        else:
            lines.append(n.ticker)
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    logger.info("Wrote auto-discovered universe (%d names) -> %s", len(names), path)
    return path


def _write_report(result: DiscoveryResult, report_dir: Optional[Path] = None) -> Path:
    """Write a dated markdown report of the discovery run."""
    out_dir = Path(report_dir or DISCOVERIES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    path = out_dir / f"{today}.md"
    lines = [
        f"# FreeAlphaRadar discoveries — {today}",
        "",
        f"Scanned **{result.candidates_screened}** shortlisted candidates; "
        f"top **{len(result.names)}** under-the-radar names (best → worst):",
        "",
        "| Rank | Ticker | Name | Score | Market cap |",
        "|-----:|--------|------|------:|-----------:|",
    ]
    for n in result.names:
        cap = f"${n.market_cap / 1e9:.2f}B" if n.market_cap else "n/a"
        lines.append(f"| {n.rank} | {n.ticker} | {n.name} | {n.score:.1f} | {cap} |")
    lines += ["", "_Not investment advice. Generated from free public data._"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote discovery report -> %s", path)
    return path


def _export_snapshot(
    tickers: List[str], path: Optional[Path] = None, min_coverage: float = 0.0
) -> None:
    """Regenerate the prewarm snapshot for the freshly discovered tickers."""
    try:
        from freealpharadar.sample_data import export_cache_snapshot

        export_cache_snapshot(tickers, path=path, min_coverage=min_coverage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not export prewarm snapshot: %s", exc)
