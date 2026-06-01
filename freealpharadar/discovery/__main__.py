"""CLI for the market-wide discovery funnel.

Examples::

    # Build/refresh the warehouse (downloads SEC bulk data; needs network):
    python -m freealpharadar.warehouse build --since 2015

    # Screen all filers + full-score the shortlist + promote the top 10:
    python -m freealpharadar.discovery run --top 10
    python -m freealpharadar.discovery run --top 10 --no-ml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from freealpharadar.discovery.discover import run_discovery
from freealpharadar.discovery.screen import ScreenConfig
from freealpharadar.utils import get_logger, setup_logging

logger = get_logger(__name__)


def _parse(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="freealpharadar.discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Screen + full-score + promote top-N.")
    run.add_argument("--top", type=int, default=10, help="Names to promote.")
    run.add_argument(
        "--candidates", type=int, default=100, help="Stage-1 shortlist size."
    )
    run.add_argument(
        "--max-cap", type=float, default=20e9, help="Market-cap ceiling (USD)."
    )
    run.add_argument(
        "--min-cagr", type=float, default=0.15, help="Min revenue CAGR gate."
    )
    run.add_argument("--no-ml", action="store_true", help="Skip FinBERT/clustering.")
    run.add_argument(
        "--dry-run", action="store_true", help="Do not write universe/report."
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    """Entry point."""
    setup_logging()
    args = _parse(argv)
    if args.cmd == "run":
        cfg = ScreenConfig(n_candidates=args.candidates, min_revenue_cagr=args.min_cagr)
        result = run_discovery(
            screen_cfg=cfg,
            top_n=args.top,
            max_market_cap=args.max_cap,
            run_ml=not args.no_ml,
            write_outputs=not args.dry_run,
        )
        logger.info(
            "Discovered %d names from %d candidates:",
            len(result.names),
            result.candidates_screened,
        )
        for n in result.names:
            cap = f"${n.market_cap / 1e9:.1f}B" if n.market_cap else "n/a"
            logger.info(
                "  %2d. %-6s %5.1f  %-30s %s", n.rank, n.ticker, n.score, n.name, cap
            )
        if result.universe_path:
            logger.info(
                "Universe -> %s | report -> %s",
                result.universe_path,
                result.report_path,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
