"""Standalone batch scorer.

Refreshes all data sources for a universe and updates the SQLite cache and the
stored score snapshots. Designed to be scheduled (cron / GitHub Actions) so the
Streamlit app only ever has to *read* a warm cache.

Usage::

    python run_scorer.py                      # default universe, refresh caches
    python run_scorer.py --tickers PLTR BE    # explicit universe
    python run_scorer.py --no-refresh         # use cache where fresh
    python run_scorer.py --seed-sample        # (re)seed offline sample data
    python run_scorer.py --no-ml              # skip FinBERT/clustering

The app remains fully functional even if this is never run: the Streamlit app
self-seeds sample data on first launch.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from freealpharadar.config import settings
from freealpharadar.sample_data import seed_cache, write_sample_json
from freealpharadar.service import run_pipeline
from freealpharadar.utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FreeAlphaRadar batch scorer")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to score (defaults to the configured universe).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not bypass cache TTLs (use cached data where still fresh).",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Skip FinBERT/clustering enrichment.",
    )
    parser.add_argument(
        "--seed-sample",
        action="store_true",
        help="Seed/refresh the offline sample dataset before scoring.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    """Entry point. Returns a process exit code."""
    setup_logging()
    args = parse_args(argv)

    tickers = args.tickers or settings.default_universe

    if args.seed_sample:
        write_sample_json()
        seed_cache(force=True)
        logger.info("Sample dataset seeded.")

    logger.info(
        "Scoring %d tickers (refresh=%s, ml=%s)...",
        len(tickers),
        not args.no_refresh,
        not args.no_ml,
    )

    def _cb(done: int, total: int, ticker: str) -> None:
        logger.info("  [%d/%d] %s", done, total, ticker)

    output = run_pipeline(
        tickers,
        force_refresh=not args.no_refresh,
        run_ml=not args.no_ml,
        progress_cb=_cb,
        persist=True,
    )

    logger.info("Done. Top names:")
    for res in output.results[:10]:
        logger.info("  %-6s %5.1f  %s", res.ticker, res.score, res.name)

    if output.warnings:
        logger.warning("%d data warning(s) during run.", len(output.warnings))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
