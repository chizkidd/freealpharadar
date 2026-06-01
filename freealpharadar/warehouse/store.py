"""DuckDB-backed read API over the fundamentals Parquet store."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from freealpharadar.utils import get_logger
from freealpharadar.warehouse.config import FACTS_PARQUET

logger = get_logger(__name__)


class WarehouseStore:
    """Read-only query layer over the warehouse Parquet file.

    Args:
        parquet_path: Path to the facts Parquet; defaults to
            :data:`FACTS_PARQUET`.
    """

    def __init__(self, parquet_path: Optional[Path] = None) -> None:
        self.path = Path(parquet_path or FACTS_PARQUET)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Warehouse not built at {self.path}. Run "
                f"`python -m freealpharadar.warehouse build` first."
            )

    def _connect(self):
        import duckdb

        con = duckdb.connect(database=":memory:")
        con.execute(
            f"CREATE VIEW facts AS SELECT * FROM read_parquet('{self.path.as_posix()}')"
        )
        return con

    def query(self, sql: str) -> pd.DataFrame:
        """Run arbitrary SQL against the ``facts`` view and return a DataFrame."""
        con = self._connect()
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def annual_series(self, ticker: str, field: str = "revenue") -> pd.DataFrame:
        """Return the ascending ``fy, value`` series for one ticker/field."""
        return self.query(
            f"SELECT fy, {field} AS value FROM facts "
            f"WHERE ticker = '{ticker.upper()}' AND {field} IS NOT NULL "
            f"ORDER BY fy"
        )

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Return all stored annual rows for one ticker."""
        return self.query(
            f"SELECT * FROM facts WHERE ticker = '{ticker.upper()}' ORDER BY fy"
        )

    def tickers(self) -> list:
        """Distinct non-empty tickers present in the store."""
        df = self.query("SELECT DISTINCT ticker FROM facts WHERE ticker <> ''")
        return sorted(df["ticker"].tolist())
