"""CLI for building/querying the bulk-fundamentals warehouse.

Examples::

    python -m freealpharadar.warehouse build --since 2015
    python -m freealpharadar.warehouse query --ticker AAPL
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from freealpharadar.utils import get_logger, setup_logging

logger = get_logger(__name__)


def _parse(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="freealpharadar.warehouse")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Download + load SEC bulk data into Parquet.")
    b.add_argument("--since", type=int, default=2015, help="First year to ingest.")

    q = sub.add_parser("query", help="Show stored annual fundamentals for a ticker.")
    q.add_argument("--ticker", required=True)
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    """Entry point."""
    setup_logging()
    args = _parse(argv)
    if args.cmd == "build":
        from freealpharadar.warehouse.loader import build_warehouse

        path = build_warehouse(since_year=args.since)
        logger.info("Warehouse built at %s", path)
    elif args.cmd == "query":
        from freealpharadar.warehouse.store import WarehouseStore

        df = WarehouseStore().get_fundamentals(args.ticker)
        logger.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
