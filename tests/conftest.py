"""Pytest configuration and shared fixtures.

Critically, this module configures the environment for **fully offline**
testing *before* any FreeAlphaRadar module is imported:

* ``FAR_OFFLINE=1`` ensures no fetcher ever touches the network.
* ``FAR_DB_PATH`` points the SQLite cache at a throwaway temp file so tests
  never pollute the real cache.

All fixtures build :class:`CompanyData` from deterministic canned data, so the
scoring maths is exercised without any external dependency.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- Configure environment BEFORE importing the package. -------------------- #
_TMP_DB = Path(tempfile.gettempdir()) / "freealpharadar_test.sqlite"
os.environ.setdefault("FAR_OFFLINE", "1")
os.environ["FAR_DB_PATH"] = str(_TMP_DB)
os.environ.setdefault("FAR_LOG_LEVEL", "WARNING")

import pytest  # noqa: E402

from freealpharadar.pipeline import CompanyData  # noqa: E402
from freealpharadar.sample_data import build_sample_dataset  # noqa: E402


def _company_from_sample(ticker: str) -> CompanyData:
    """Construct a CompanyData for ``ticker`` from the sample dataset."""
    dataset = build_sample_dataset()
    bundle = dataset[ticker]
    yf = bundle["yfinance"]
    metrics = yf["key_metrics"]
    return CompanyData(
        ticker=ticker,
        name=yf["info"]["longName"],
        sector=metrics["sector"],
        market_cap=metrics["market_cap"],
        yfinance=yf,
        sec=bundle["sec"],
        patents=bundle["patentsview"],
        gdelt=bundle["gdelt"],
        manual={},
    )


@pytest.fixture(scope="session")
def sample_universe():
    """A small, deterministic universe of CompanyData objects."""
    tickers = ["PLTR", "BE", "IONQ", "RKLB", "CRWD"]
    return [_company_from_sample(t) for t in tickers]


@pytest.fixture
def single_company():
    """A single well-formed CompanyData (Palantir) for factor tests."""
    return _company_from_sample("PLTR")


@pytest.fixture
def empty_company():
    """A CompanyData with no data, to exercise None-safety."""
    return CompanyData(ticker="EMPTY")


@pytest.fixture
def handcrafted_company():
    """A CompanyData with hand-set, exactly-known numbers for precise asserts."""
    return CompanyData(
        ticker="TEST",
        name="Test Co",
        sector="Technology",
        market_cap=1_000_000_000.0,
        yfinance={
            "info": {"totalRevenue": 1000.0, "ebitda": 200.0, "freeCashflow": 50.0},
            "income_statement": [
                {
                    "period": "2024",
                    "Total Revenue": 1000.0,
                    "Gross Profit": 600.0,
                    "Research And Development": 200.0,
                    "Net Income": 100.0,
                },
                {
                    "period": "2023",
                    "Total Revenue": 800.0,
                    "Gross Profit": 440.0,
                    "Research And Development": 150.0,
                    "Net Income": 50.0,
                },
                {
                    "period": "2022",
                    "Total Revenue": 500.0,
                    "Gross Profit": 250.0,
                    "Research And Development": 100.0,
                    "Net Income": -10.0,
                },
                {
                    "period": "2021",
                    "Total Revenue": 250.0,
                    "Gross Profit": 100.0,
                    "Research And Development": 60.0,
                    "Net Income": -40.0,
                },
            ],
            "cash_flow": [
                {"period": "2024", "Operating Cash Flow": 150.0},
                {"period": "2023", "Operating Cash Flow": 90.0},
                {"period": "2022", "Operating Cash Flow": 10.0},
                {"period": "2021", "Operating Cash Flow": -20.0},
            ],
            "balance_sheet": [],
            "history": [{"date": f"m{i}", "close": 10.0 + i} for i in range(20)],
            "key_metrics": {
                "market_cap": 1_000_000_000.0,
                "enterprise_value": 1200.0,
                "trailing_pe": 25.0,
                "price_to_book": 5.0,
                "short_percent_of_float": 0.1,
                "held_percent_institutions": 0.5,
                "held_percent_insiders": 0.2,
                "gross_margins": 0.6,
                "profit_margins": 0.1,
                "revenue_growth": 0.25,
                "sector": "Technology",
            },
        },
        sec={
            "sections": {
                "risk_factors": "We face litigation risk and uncertainty.",
                "business": "Founded by our founder.",
            },
            "flags": {
                "founder_led": True,
                "key_person_dependency": True,
                "customer_concentration": False,
                "regulatory_risk_count": 3,
            },
            "insider_transactions": {"form4_count": 5},
        },
        patents={
            "total_patents": 100,
            "patent_growth_rate": 0.5,
            "counts_by_year": [
                {"year": "2021", "count": 10},
                {"year": "2022", "count": 15},
                {"year": "2023", "count": 25},
                {"year": "2024", "count": 30},
            ],
            "sample_titles": ["A", "B", "C"],
        },
        gdelt={
            "avg_tone": 2.5,
            "article_count": 12,
            "articles": [
                {"title": "Great growth and record profit", "tone": 4.0},
                {"title": "Lawsuit and investigation risk", "tone": -3.0},
            ],
        },
        manual={
            "product_moat_score": 8.0,
            "culture_score": 7.5,
            "employee_growth": 0.3,
            "glassdoor_rating": 4.2,
        },
    )
