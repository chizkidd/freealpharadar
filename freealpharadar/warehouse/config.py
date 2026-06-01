"""Configuration for the bulk-fundamentals warehouse."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from freealpharadar.config import DATA_DIR

# Gitignored store location.
WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"
ZIP_CACHE_DIR: Path = WAREHOUSE_DIR / "zips"
FACTS_PARQUET: Path = WAREHOUSE_DIR / "facts.parquet"
CIK_TICKER_CACHE: Path = WAREHOUSE_DIR / "cik_ticker.json"

# SEC Financial Statement Data Sets: one ZIP per quarter, e.g. ".../2023q1.zip".
DATASET_URL_TEMPLATE = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
)
# Coverage of the program.
FIRST_YEAR = 2009

# Map canonical field -> the us-gaap XBRL tags that may carry it (first hit
# wins per filing). Companies tag the same concept several ways, so we coalesce.
CANONICAL_TAGS: Dict[str, Tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "gross_profit": ("GrossProfit",),
    "rnd": ("ResearchAndDevelopmentExpense",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
}

# Flow concepts are reported as an annual duration (qtrs == 4); balance-sheet
# concepts are instantaneous (qtrs == 0).
FLOW_FIELDS = frozenset({"revenue", "gross_profit", "rnd", "net_income"})
INSTANT_FIELDS = frozenset({"assets", "liabilities"})

# Forms whose annual XBRL we trust for the yearly series.
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A"})
