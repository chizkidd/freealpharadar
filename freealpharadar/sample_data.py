"""Offline sample-data generation and cache seeding.

To honour the promise that FreeAlphaRadar runs **fully offline with zero
configuration**, the SQLite cache is seeded on first launch with deterministic,
plausible sample payloads for the default universe -- shaped exactly like the
real fetcher outputs. This lets the entire pipeline (ingest -> enrich -> score)
run with no network access, and gives the test-suite stable canned data.

The numbers are synthetic and clearly not investment advice; they exist purely
so the dashboard is populated and the scoring maths is exercised end to end.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from freealpharadar.config import SAMPLE_DATA_DIR, settings
from freealpharadar.database import Database, get_db
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

# Curated metadata so the sample universe looks realistic (sector, scale, theme).
_PROFILES: Dict[str, Dict[str, Any]] = {
    "PLTR": {
        "name": "Palantir Technologies Inc.",
        "sector": "Technology",
        "industry": "Software",
        "cap": 55e9,
        "founder": True,
        "theme": "AI/defense",
    },
    "BE": {
        "name": "Bloom Energy Corporation",
        "sector": "Industrials",
        "industry": "Electrical Equipment",
        "cap": 4e9,
        "founder": True,
        "theme": "fuel cells",
    },
    "SNDK": {
        "name": "Sandisk Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
        "cap": 9e9,
        "founder": False,
        "theme": "flash storage",
    },
    "IONQ": {
        "name": "IonQ, Inc.",
        "sector": "Technology",
        "industry": "Computer Hardware",
        "cap": 8e9,
        "founder": True,
        "theme": "quantum",
    },
    "RKLB": {
        "name": "Rocket Lab USA, Inc.",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "cap": 11e9,
        "founder": True,
        "theme": "launch",
    },
    "OKLO": {
        "name": "Oklo Inc.",
        "sector": "Utilities",
        "industry": "Utilities-Renewable",
        "cap": 6e9,
        "founder": True,
        "theme": "nuclear",
    },
    "SMR": {
        "name": "NuScale Power Corporation",
        "sector": "Utilities",
        "industry": "Utilities-Renewable",
        "cap": 5e9,
        "founder": False,
        "theme": "nuclear",
    },
    "ASTS": {
        "name": "AST SpaceMobile, Inc.",
        "sector": "Communication Services",
        "industry": "Telecom",
        "cap": 7e9,
        "founder": True,
        "theme": "satellite",
    },
    "TEM": {
        "name": "Tempus AI, Inc.",
        "sector": "Healthcare",
        "industry": "Health Information Services",
        "cap": 9e9,
        "founder": True,
        "theme": "AI diagnostics",
    },
    "RXRX": {
        "name": "Recursion Pharmaceuticals, Inc.",
        "sector": "Healthcare",
        "industry": "Biotechnology",
        "cap": 3e9,
        "founder": True,
        "theme": "AI drug discovery",
    },
    "PATH": {
        "name": "UiPath Inc.",
        "sector": "Technology",
        "industry": "Software",
        "cap": 7e9,
        "founder": True,
        "theme": "automation",
    },
    "CRWD": {
        "name": "CrowdStrike Holdings, Inc.",
        "sector": "Technology",
        "industry": "Software-Infrastructure",
        "cap": 90e9,
        "founder": True,
        "theme": "cybersecurity",
    },
}

_RISK_TEMPLATE = (
    "Item 1A. Risk Factors. We have a history of operating losses and may not "
    "achieve profitability. Our business depends on the continued growth of the "
    "{theme} market, which is uncertain and subject to regulation and regulatory "
    "scrutiny. We rely on a limited number of customers and a significant portion "
    "of our revenue is concentrated. The loss of our chief executive officer or "
    "other key personnel could harm our business. We face intense competition and "
    "rapid technological change. Compliance with evolving regulation may increase "
    "costs. Item 2. Properties."
)

_BUSINESS_TEMPLATE = (
    "Item 1. Business. {name} was founded by our founder to build {theme} "
    "technology. Our founder continues to lead the company. We develop and sell "
    "products that we believe create a durable competitive moat. Item 1A. Risk Factors."
)


def _rng(ticker: str) -> random.Random:
    """Deterministic per-ticker RNG so sample data is stable across runs."""
    return random.Random(hash(ticker) & 0xFFFFFFFF)


def _yf_payload(ticker: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build a yfinance-shaped sample payload."""
    rng = _rng(ticker)
    base_rev = profile["cap"] * rng.uniform(0.05, 0.3)
    growth = rng.uniform(0.1, 0.6)
    periods = ["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
    income, history = [], []
    rev = base_rev
    for p in periods:
        gm = rng.uniform(0.4, 0.8)
        gross = rev * gm
        rnd = rev * rng.uniform(0.1, 0.4)
        ni = gross - rev * rng.uniform(0.5, 0.9)
        income.append(
            {
                "period": p,
                "Total Revenue": round(rev, 2),
                "Gross Profit": round(gross, 2),
                "Research And Development": round(rnd, 2),
                "Net Income": round(ni, 2),
                "Operating Income": round(gross - rev * 0.4, 2),
            }
        )
        rev = rev / (1 + growth)  # older periods are smaller

    cashflow = [
        {
            "period": p,
            "Operating Cash Flow": round(
                income[i]["Net Income"] + income[i]["Total Revenue"] * 0.1, 2
            ),
        }
        for i, p in enumerate(periods)
    ]
    balance = [
        {
            "period": p,
            "Assets": round(profile["cap"] * 0.3, 2),
            "Liabilities": round(profile["cap"] * 0.1, 2),
        }
        for p in periods
    ]

    price = rng.uniform(10, 60)
    for m in range(60):
        price *= 1 + rng.uniform(-0.08, 0.12)
        history.append(
            {
                "date": f"2020-{(m % 12) + 1:02d}-01",
                "close": round(price, 2),
                "volume": rng.randint(1_000_000, 50_000_000),
            }
        )

    return {
        "ticker": ticker,
        "info": {
            "longName": profile["name"],
            "shortName": profile["name"].split()[0],
            "symbol": ticker,
            "sector": profile["sector"],
            "industry": profile["industry"],
            "longBusinessSummary": f"{profile['name']} operates in {profile['theme']}.",
            "fullTimeEmployees": rng.randint(500, 8000),
            "marketCap": profile["cap"],
            "enterpriseValue": profile["cap"] * rng.uniform(0.85, 1.1),
            "totalRevenue": base_rev,
            "ebitda": base_rev * rng.uniform(-0.1, 0.25),
            "freeCashflow": base_rev * rng.uniform(-0.1, 0.2),
        },
        "history": history,
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cashflow,
        "key_metrics": {
            "market_cap": profile["cap"],
            "enterprise_value": profile["cap"] * rng.uniform(0.85, 1.1),
            "short_ratio": rng.uniform(1, 8),
            "short_percent_of_float": rng.uniform(0.02, 0.25),
            "held_percent_institutions": rng.uniform(0.2, 0.9),
            "held_percent_insiders": rng.uniform(0.02, 0.4),
            "sector": profile["sector"],
            "industry": profile["industry"],
            "trailing_pe": rng.uniform(15, 120),
            "price_to_book": rng.uniform(2, 30),
            "gross_margins": rng.uniform(0.4, 0.8),
            "profit_margins": rng.uniform(-0.3, 0.25),
            "revenue_growth": growth,
        },
    }


def _sec_payload(ticker: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build an SEC-shaped sample payload."""
    rng = _rng(ticker + "sec")
    return {
        "ticker": ticker,
        "cik": str(rng.randint(1_000_000, 1_999_999)).zfill(10),
        "company_name": profile["name"],
        "sic_description": profile["industry"],
        "sections": {
            "form": "10-K",
            "filing_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
            "business": _BUSINESS_TEMPLATE.format(
                name=profile["name"], theme=profile["theme"]
            ),
            "risk_factors": _RISK_TEMPLATE.format(theme=profile["theme"]),
            "mdna": "Management's Discussion. Revenue grew driven by demand.",
        },
        "facts": {},
        "insider_transactions": {"form4_count": rng.randint(0, 30), "recent": []},
        "flags": {
            "founder_led": profile["founder"],
            "key_person_dependency": True,
            "customer_concentration": rng.random() > 0.4,
            "regulatory_risk_count": rng.randint(1, 9),
        },
    }


def _patents_payload(ticker: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build a PatentsView-shaped sample payload."""
    rng = _rng(ticker + "pat")
    counts = []
    base = rng.randint(2, 40)
    for y in range(2018, 2025):
        base = max(0, int(base * rng.uniform(0.9, 1.6)))
        counts.append({"year": str(y), "count": base})
    total = sum(c["count"] for c in counts)
    recent = sum(c["count"] for c in counts[-2:])
    prior = sum(c["count"] for c in counts[-4:-2]) or 1
    return {
        "company_name": profile["name"],
        "total_patents": total,
        "counts_by_year": counts,
        "patent_growth_rate": (recent - prior) / prior,
        "sample_titles": [
            f"System and method for {profile['theme']} #{i}"
            for i in range(1, rng.randint(5, 18))
        ],
    }


def _gdelt_payload(ticker: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build a GDELT-shaped sample payload."""
    rng = _rng(ticker + "news")
    n = rng.randint(5, 30)
    articles = []
    for i in range(n):
        tone = rng.uniform(-6, 6)
        articles.append(
            {
                "title": f"{profile['name']} advances in {profile['theme']} ({i})",
                "url": f"https://news.example.com/{ticker.lower()}/{i}",
                "domain": "news.example.com",
                "seendate": "20250115T120000Z",
                "tone": round(tone, 2),
                "language": "English",
            }
        )
    avg = sum(a["tone"] for a in articles) / len(articles)
    return {
        "company_name": profile["name"],
        "article_count": n,
        "avg_tone": round(avg, 2),
        "articles": articles,
    }


def build_sample_dataset() -> Dict[str, Dict[str, Any]]:
    """Build the full sample dataset for the default universe.

    Returns:
        Mapping of ticker -> ``{"yfinance":..., "sec":..., "patentsview":...,
        "gdelt":...}``.
    """
    dataset: Dict[str, Dict[str, Any]] = {}
    for ticker, profile in _PROFILES.items():
        dataset[ticker] = {
            "yfinance": _yf_payload(ticker, profile),
            "sec": _sec_payload(ticker, profile),
            "patentsview": _patents_payload(ticker, profile),
            "gdelt": _gdelt_payload(ticker, profile),
        }
    return dataset


def write_sample_json() -> None:
    """Persist the sample dataset to ``data/sample/sample_companies.json``."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DATA_DIR / "sample_companies.json"
    path.write_text(json.dumps(build_sample_dataset(), indent=2), encoding="utf-8")
    logger.info("Wrote sample dataset to %s", path)


def seed_cache(db: Database | None = None, force: bool = False) -> int:
    """Seed the SQLite cache with the sample dataset for offline operation.

    Args:
        db: Database to seed; defaults to the shared singleton.
        force: Re-seed even if cache entries already exist.

    Returns:
        Number of ticker bundles seeded.
    """
    db = db or get_db()
    dataset = build_sample_dataset()
    seeded = 0
    for ticker, sources in dataset.items():
        if not force and db.is_fresh("yfinance", ticker, settings.ttl.fundamentals):
            continue
        for source, payload in sources.items():
            db.set_cache(source, ticker, payload)
        seeded += 1
    if seeded:
        logger.info("Seeded sample cache for %d tickers.", seeded)
    return seeded


def cache_is_empty(db: Database | None = None) -> bool:
    """Whether the cache has no entries for the first default ticker."""
    db = db or get_db()
    first = settings.default_universe[0] if settings.default_universe else "PLTR"
    return db.get_cache("yfinance", first) is None
