"""Factor library for the scoring engine.

Each :class:`FactorSpec` declares a single financial / alternative-data signal
together with the function that extracts its *raw* value from a
:class:`~freealpharadar.pipeline.CompanyData` bundle. The engine is responsible
for normalising and weighting; factors only compute raw numbers.

Design principles:

* **Total None-safety** -- any missing input yields ``None`` (treated as a
  neutral score downstream) rather than raising.
* **Orientation** -- ``higher_is_better`` records whether a larger raw value is
  more attractive; normalisation flips the sign for "lower is better" factors.
* **Transparency** -- every factor carries a human-readable description that
  surfaces in the deep-dive drill-down.

The 32 factors are organised into the four groups of the "Moat + Momentum +
Misvaluation" framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence

from freealpharadar.pipeline import CompanyData


class FactorGroup(str, Enum):
    """The four conceptual groups of factors."""

    MOAT = "Disruption & Moat"
    MOMENTUM = "Growth & Momentum"
    VALUATION = "Valuation & Inefficiency"
    QUALITATIVE = "Qualitative Flags"


@dataclass(frozen=True)
class FactorSpec:
    """Specification for a single scoring factor.

    Attributes:
        name: Stable machine identifier.
        label: Human-readable label for the UI.
        group: Which :class:`FactorGroup` the factor belongs to.
        fn: Callable mapping a :class:`CompanyData` to a raw float or ``None``.
        higher_is_better: Orientation flag for normalisation.
        description: Explanation shown in drill-downs.
    """

    name: str
    label: str
    group: FactorGroup
    fn: Callable[[CompanyData], Optional[float]]
    higher_is_better: bool = True
    description: str = ""


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    """Best-effort float coercion, returning ``None`` on failure."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _line_series(
    statements: Sequence[dict], candidates: Sequence[str]
) -> List[Optional[float]]:
    """Extract a line-item series (most recent first) from statement records.

    Args:
        statements: List of period records (each a dict of line item -> value).
        candidates: Candidate line-item names; the first present wins.

    Returns:
        A list of values aligned with the statement periods.
    """
    if not statements:
        return []
    key: Optional[str] = None
    for cand in candidates:
        if any(cand in rec for rec in statements):
            key = cand
            break
    if key is None:
        return []
    return [_to_float(rec.get(key)) for rec in statements]


def _income(company: CompanyData) -> List[dict]:
    return company.yfinance.get("income_statement", []) or []


def _balance(company: CompanyData) -> List[dict]:
    return company.yfinance.get("balance_sheet", []) or []


def _cashflow(company: CompanyData) -> List[dict]:
    return company.yfinance.get("cash_flow", []) or []


def _metrics(company: CompanyData) -> dict:
    return company.yfinance.get("key_metrics", {}) or {}


def _revenue_series(company: CompanyData) -> List[Optional[float]]:
    return _line_series(_income(company), ["Total Revenue", "TotalRevenue", "Revenues"])


def _gross_profit_series(company: CompanyData) -> List[Optional[float]]:
    return _line_series(_income(company), ["Gross Profit", "GrossProfit"])


# --------------------------------------------------------------------------- #
# Disruption & Moat
# --------------------------------------------------------------------------- #
def f_patent_growth(c: CompanyData) -> Optional[float]:
    """Recent-vs-prior patent-grant growth rate (PatentsView)."""
    val = c.patents.get("patent_growth_rate")
    return _to_float(val)


def f_patent_count(c: CompanyData) -> Optional[float]:
    """Total granted patents on record (PatentsView)."""
    val = c.patents.get("total_patents")
    return _to_float(val)


def f_patent_breadth(c: CompanyData) -> Optional[float]:
    """Breadth of patent activity, proxied by sampled distinct titles."""
    titles = c.patents.get("sample_titles") or []
    return float(len(titles)) if titles else None


def f_rnd_intensity(c: CompanyData) -> Optional[float]:
    """R&D expense as a fraction of revenue (yfinance income statement)."""
    rnd = _line_series(
        _income(c),
        ["Research And Development", "ResearchAndDevelopment", "Research Development"],
    )
    rev = _revenue_series(c)
    if rnd and rev and rnd[0] is not None and rev[0]:
        return rnd[0] / rev[0]
    return None


def f_founder_led(c: CompanyData) -> Optional[float]:
    """Whether the SEC business description signals founder leadership."""
    flags = c.sec.get("flags", {})
    if "founder_led" not in flags:
        return None
    return 1.0 if flags.get("founder_led") else 0.0


def f_product_moat(c: CompanyData) -> Optional[float]:
    """Product-moat score supplied via the optional manual CSV."""
    return _to_float(c.manual.get("product_moat_score"))


def f_culture(c: CompanyData) -> Optional[float]:
    """Culture score supplied via the optional manual CSV."""
    return _to_float(c.manual.get("culture_score"))


# --------------------------------------------------------------------------- #
# Growth & Momentum
# --------------------------------------------------------------------------- #
def f_revenue_cagr(c: CompanyData) -> Optional[float]:
    """Three-year revenue CAGR computed from the income statement."""
    rev = _revenue_series(c)
    rev = [v for v in rev if v is not None]
    if len(rev) < 3:
        return None
    latest, oldest = rev[0], rev[min(3, len(rev) - 1)]
    years = min(3, len(rev) - 1)
    if oldest is None or oldest <= 0 or years <= 0:
        return None
    return (latest / oldest) ** (1.0 / years) - 1.0


def f_revenue_growth(c: CompanyData) -> Optional[float]:
    """Trailing revenue growth (yfinance key metric)."""
    return _to_float(_metrics(c).get("revenue_growth"))


def f_gross_margin(c: CompanyData) -> Optional[float]:
    """Latest gross margin (gross profit / revenue)."""
    gp = _gross_profit_series(c)
    rev = _revenue_series(c)
    if gp and rev and gp[0] is not None and rev[0]:
        return gp[0] / rev[0]
    return _to_float(_metrics(c).get("gross_margins"))


def f_gross_margin_expansion(c: CompanyData) -> Optional[float]:
    """Change in gross margin between the latest and the prior-2 period."""
    gp = _gross_profit_series(c)
    rev = _revenue_series(c)
    if len(gp) < 3 or len(rev) < 3:
        return None
    try:
        latest = gp[0] / rev[0] if rev[0] else None
        prior = gp[2] / rev[2] if rev[2] else None
    except (TypeError, ZeroDivisionError):
        return None
    if latest is None or prior is None:
        return None
    return latest - prior


def f_price_momentum(c: CompanyData) -> Optional[float]:
    """Trailing ~12-month price return from monthly history."""
    hist = c.yfinance.get("history", []) or []
    closes = [_to_float(h.get("close")) for h in hist]
    closes = [x for x in closes if x is not None and x > 0]
    if len(closes) < 13:
        return None
    return closes[-1] / closes[-13] - 1.0


def f_employee_growth(c: CompanyData) -> Optional[float]:
    """Employee growth supplied via the optional manual CSV."""
    return _to_float(c.manual.get("employee_growth"))


def f_customer_concentration(c: CompanyData) -> Optional[float]:
    """Customer-concentration risk flag parsed from SEC risk factors."""
    flags = c.sec.get("flags", {})
    if "customer_concentration" not in flags:
        return None
    return 1.0 if flags.get("customer_concentration") else 0.0


# --------------------------------------------------------------------------- #
# Valuation & Inefficiency
# --------------------------------------------------------------------------- #
def f_ev_gp(c: CompanyData) -> Optional[float]:
    """Enterprise value to gross profit (lower is cheaper)."""
    ev = _to_float(_metrics(c).get("enterprise_value"))
    gp = _gross_profit_series(c)
    if ev is not None and gp and gp[0] and gp[0] > 0:
        return ev / gp[0]
    return None


def f_trailing_pe(c: CompanyData) -> Optional[float]:
    """Trailing P/E (lower is cheaper)."""
    return _to_float(_metrics(c).get("trailing_pe"))


def f_price_to_book(c: CompanyData) -> Optional[float]:
    """Price-to-book ratio (lower is cheaper)."""
    return _to_float(_metrics(c).get("price_to_book"))


def f_short_interest(c: CompanyData) -> Optional[float]:
    """Short interest as a percent of float (lower = less bearish positioning).

    Note: heavily-shorted names can also be squeeze candidates; the orientation
    is configurable by adjusting the factor's weight in the UI.
    """
    val = _to_float(_metrics(c).get("short_percent_of_float"))
    if val is None:
        val = _to_float(_metrics(c).get("short_ratio"))
    return val


def f_institutional_ownership(c: CompanyData) -> Optional[float]:
    """Institutional ownership fraction.

    Lower is treated as *more* attractive: low institutional ownership is the
    hallmark of a genuinely under-the-radar name with room for re-rating.
    """
    return _to_float(_metrics(c).get("held_percent_institutions"))


def f_insider_ownership(c: CompanyData) -> Optional[float]:
    """Insider ownership fraction (higher = more skin in the game)."""
    return _to_float(_metrics(c).get("held_percent_insiders"))


def f_insider_activity(c: CompanyData) -> Optional[float]:
    """Recent Form 4 filing count.

    Treated as lower-is-better since a surge of Form 4s frequently reflects
    insider *selling*; the deep-dive shows the underlying filings.
    """
    insider = c.sec.get("insider_transactions", {})
    if "form4_count" not in insider:
        return None
    return _to_float(insider.get("form4_count"))


def f_piotroski(c: CompanyData) -> Optional[float]:
    """A simplified Piotroski F-style score (0-5) from the statements.

    Points awarded for: positive net income, positive operating cash flow,
    cash flow exceeding net income (earnings quality), gross-margin
    improvement, and positive revenue growth.
    """
    rev = _revenue_series(c)
    ni = _line_series(
        _income(c), ["Net Income", "NetIncome", "Net Income Common Stockholders"]
    )
    cfo = _line_series(
        _cashflow(c),
        [
            "Operating Cash Flow",
            "OperatingCashFlow",
            "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities",
        ],
    )
    if not rev or not ni:
        return None

    score = 0
    used = False
    if ni and ni[0] is not None:
        used = True
        score += 1 if ni[0] > 0 else 0
    if cfo and cfo[0] is not None:
        used = True
        score += 1 if cfo[0] > 0 else 0
        if ni and ni[0] is not None and cfo[0] > ni[0]:
            score += 1
    gm = f_gross_margin(c)
    gm_exp = f_gross_margin_expansion(c)
    if gm_exp is not None:
        used = True
        score += 1 if gm_exp > 0 else 0
    rg = f_revenue_cagr(c)
    if rg is not None:
        used = True
        score += 1 if rg > 0 else 0
    return float(score) if used else None


def f_profit_margin(c: CompanyData) -> Optional[float]:
    """Net profit margin (yfinance key metric)."""
    return _to_float(_metrics(c).get("profit_margins"))


def f_ebitda_margin(c: CompanyData) -> Optional[float]:
    """EBITDA margin computed from yfinance info."""
    info = c.yfinance.get("info", {}) or {}
    ebitda = _to_float(info.get("ebitda"))
    rev = _to_float(info.get("totalRevenue"))
    if ebitda is not None and rev:
        return ebitda / rev
    return None


def f_fcf_positive(c: CompanyData) -> Optional[float]:
    """Whether free cash flow is positive (yfinance info)."""
    info = c.yfinance.get("info", {}) or {}
    fcf = _to_float(info.get("freeCashflow"))
    if fcf is None:
        return None
    return 1.0 if fcf > 0 else 0.0


# --------------------------------------------------------------------------- #
# Qualitative Flags
# --------------------------------------------------------------------------- #
def f_news_tone(c: CompanyData) -> Optional[float]:
    """Average GDELT news tone (higher = more positive coverage)."""
    if not c.gdelt:
        return None
    return _to_float(c.gdelt.get("avg_tone"))


def f_controversy(c: CompanyData) -> Optional[float]:
    """Controversy score (lower is better).

    Prefers the FinBERT-derived score attached to ``derived`` when available;
    otherwise falls back to a heuristic combining negative news tone and the
    SEC regulatory-risk keyword count so the factor works fully offline.
    """
    derived = getattr(c, "derived", {}) or {}
    if "controversy_score" in derived and derived["controversy_score"] is not None:
        return _to_float(derived["controversy_score"])
    tone = c.gdelt.get("avg_tone") if c.gdelt else None
    reg = c.sec.get("flags", {}).get("regulatory_risk_count")
    if tone is None and reg is None:
        return None
    score = 0.0
    if tone is not None:
        score += max(0.0, -float(tone))
    if reg is not None:
        score += min(float(reg), 10.0) * 0.5
    return score


def f_finbert_sentiment(c: CompanyData) -> Optional[float]:
    """FinBERT sentiment of the SEC risk section (higher = more positive).

    Populated by the ML enrichment step; ``None`` when FinBERT has not run.
    """
    derived = getattr(c, "derived", {}) or {}
    return _to_float(derived.get("finbert_sentiment"))


def f_key_person(c: CompanyData) -> Optional[float]:
    """Key-person dependency flag from SEC risk factors (lower is better)."""
    flags = c.sec.get("flags", {})
    if "key_person_dependency" not in flags:
        return None
    return 1.0 if flags.get("key_person_dependency") else 0.0


def f_regulatory_risk(c: CompanyData) -> Optional[float]:
    """Count of regulatory-risk keyword hits in SEC text (lower is better)."""
    flags = c.sec.get("flags", {})
    val = flags.get("regulatory_risk_count")
    return _to_float(val)


def f_news_volume(c: CompanyData) -> Optional[float]:
    """Volume of recent news coverage (GDELT article count)."""
    if not c.gdelt:
        return None
    return _to_float(c.gdelt.get("article_count"))


def f_glassdoor(c: CompanyData) -> Optional[float]:
    """Glassdoor-style rating supplied via the optional manual CSV."""
    return _to_float(c.manual.get("glassdoor_rating"))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
FACTORS: List[FactorSpec] = [
    # Disruption & Moat
    FactorSpec(
        "patent_growth",
        "Patent growth rate",
        FactorGroup.MOAT,
        f_patent_growth,
        True,
        "Recent vs prior granted-patent growth (PatentsView).",
    ),
    FactorSpec(
        "patent_count",
        "Patent count",
        FactorGroup.MOAT,
        f_patent_count,
        True,
        "Total granted patents on record.",
    ),
    FactorSpec(
        "patent_breadth",
        "Patent breadth",
        FactorGroup.MOAT,
        f_patent_breadth,
        True,
        "Distinct patent titles sampled, a breadth proxy.",
    ),
    FactorSpec(
        "rnd_intensity",
        "R&D intensity",
        FactorGroup.MOAT,
        f_rnd_intensity,
        True,
        "R&D expense divided by revenue.",
    ),
    FactorSpec(
        "founder_led",
        "Founder-led",
        FactorGroup.MOAT,
        f_founder_led,
        True,
        "SEC business description signals founder leadership.",
    ),
    FactorSpec(
        "product_moat",
        "Product moat (manual)",
        FactorGroup.MOAT,
        f_product_moat,
        True,
        "Analyst product-moat score from the manual CSV.",
    ),
    FactorSpec(
        "culture",
        "Culture (manual)",
        FactorGroup.MOAT,
        f_culture,
        True,
        "Employee culture score from the manual CSV.",
    ),
    # Growth & Momentum
    FactorSpec(
        "revenue_cagr",
        "Revenue CAGR (3y)",
        FactorGroup.MOMENTUM,
        f_revenue_cagr,
        True,
        "Three-year revenue compound annual growth rate.",
    ),
    FactorSpec(
        "revenue_growth",
        "Revenue growth (TTM)",
        FactorGroup.MOMENTUM,
        f_revenue_growth,
        True,
        "Trailing revenue growth from yfinance.",
    ),
    FactorSpec(
        "gross_margin",
        "Gross margin",
        FactorGroup.MOMENTUM,
        f_gross_margin,
        True,
        "Latest gross profit divided by revenue.",
    ),
    FactorSpec(
        "gross_margin_expansion",
        "Gross margin expansion",
        FactorGroup.MOMENTUM,
        f_gross_margin_expansion,
        True,
        "Change in gross margin over 2 periods.",
    ),
    FactorSpec(
        "price_momentum",
        "Price momentum (12m)",
        FactorGroup.MOMENTUM,
        f_price_momentum,
        True,
        "Trailing ~12-month price return.",
    ),
    FactorSpec(
        "employee_growth",
        "Employee growth (manual)",
        FactorGroup.MOMENTUM,
        f_employee_growth,
        True,
        "Headcount growth from the manual CSV.",
    ),
    FactorSpec(
        "customer_concentration",
        "Customer concentration risk",
        FactorGroup.MOMENTUM,
        f_customer_concentration,
        False,
        "Customer-concentration risk flagged in SEC filings.",
    ),
    # Valuation & Inefficiency
    FactorSpec(
        "ev_gp",
        "EV / Gross Profit",
        FactorGroup.VALUATION,
        f_ev_gp,
        False,
        "Enterprise value over gross profit (lower is cheaper).",
    ),
    FactorSpec(
        "trailing_pe",
        "Trailing P/E",
        FactorGroup.VALUATION,
        f_trailing_pe,
        False,
        "Trailing price/earnings (lower is cheaper).",
    ),
    FactorSpec(
        "price_to_book",
        "Price / Book",
        FactorGroup.VALUATION,
        f_price_to_book,
        False,
        "Price-to-book ratio (lower is cheaper).",
    ),
    FactorSpec(
        "short_interest",
        "Short interest",
        FactorGroup.VALUATION,
        f_short_interest,
        False,
        "Short interest as a percent of float.",
    ),
    FactorSpec(
        "institutional_ownership",
        "Institutional ownership",
        FactorGroup.VALUATION,
        f_institutional_ownership,
        False,
        "Lower institutional ownership = more under-the-radar.",
    ),
    FactorSpec(
        "insider_ownership",
        "Insider ownership",
        FactorGroup.VALUATION,
        f_insider_ownership,
        True,
        "Insider ownership fraction (skin in game).",
    ),
    FactorSpec(
        "insider_activity",
        "Insider Form 4 activity",
        FactorGroup.VALUATION,
        f_insider_activity,
        False,
        "Recent Form 4 filing count.",
    ),
    FactorSpec(
        "piotroski",
        "Piotroski F-score",
        FactorGroup.VALUATION,
        f_piotroski,
        True,
        "Simplified Piotroski financial-strength score (0-5).",
    ),
    FactorSpec(
        "profit_margin",
        "Profit margin",
        FactorGroup.VALUATION,
        f_profit_margin,
        True,
        "Net profit margin.",
    ),
    FactorSpec(
        "ebitda_margin",
        "EBITDA margin",
        FactorGroup.VALUATION,
        f_ebitda_margin,
        True,
        "EBITDA divided by revenue.",
    ),
    FactorSpec(
        "fcf_positive",
        "Free cash flow positive",
        FactorGroup.VALUATION,
        f_fcf_positive,
        True,
        "Whether free cash flow is positive.",
    ),
    # Qualitative Flags
    FactorSpec(
        "news_tone",
        "News tone (GDELT)",
        FactorGroup.QUALITATIVE,
        f_news_tone,
        True,
        "Average GDELT news tone.",
    ),
    FactorSpec(
        "controversy",
        "Controversy score",
        FactorGroup.QUALITATIVE,
        f_controversy,
        False,
        "FinBERT/GDELT-derived controversy (lower better).",
    ),
    FactorSpec(
        "finbert_sentiment",
        "FinBERT risk sentiment",
        FactorGroup.QUALITATIVE,
        f_finbert_sentiment,
        True,
        "FinBERT sentiment on SEC risk section.",
    ),
    FactorSpec(
        "key_person",
        "Key-person dependency",
        FactorGroup.QUALITATIVE,
        f_key_person,
        False,
        "Key-person dependency flagged in SEC filings.",
    ),
    FactorSpec(
        "regulatory_risk",
        "Regulatory risk",
        FactorGroup.QUALITATIVE,
        f_regulatory_risk,
        False,
        "Regulatory-risk keyword count in SEC text.",
    ),
    FactorSpec(
        "news_volume",
        "News volume",
        FactorGroup.QUALITATIVE,
        f_news_volume,
        True,
        "Recent GDELT article count.",
    ),
    FactorSpec(
        "glassdoor",
        "Glassdoor rating (manual)",
        FactorGroup.QUALITATIVE,
        f_glassdoor,
        True,
        "Glassdoor-style rating from the manual CSV.",
    ),
]

# Quick lookup by name.
FACTORS_BY_NAME = {f.name: f for f in FACTORS}
