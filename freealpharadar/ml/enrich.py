"""ML enrichment glue.

Attaches ML-derived signals to :class:`~freealpharadar.pipeline.CompanyData`
objects *before* scoring, populating their ``derived`` dict with:

* ``finbert_sentiment`` -- FinBERT (or lexicon) sentiment of the SEC risk
  section, in ``[-1, 1]``.
* ``finbert_news_sentiment`` -- sentiment of recent GDELT headlines.
* ``controversy_score`` -- a blended controversy metric (higher = more
  controversial).
* ``cluster`` -- the PCA/K-Means peer-cluster label.

All steps degrade gracefully: if a model can't load, the lexicon fallback runs;
if data is missing, the field is simply left ``None``.
"""

from __future__ import annotations

from typing import List, Optional

from freealpharadar.ml.clustering import cluster_companies
from freealpharadar.ml.finbert import FinBERTSentiment
from freealpharadar.pipeline import CompanyData
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


def enrich_companies(
    companies: List[CompanyData],
    analyzer: Optional[FinBERTSentiment] = None,
    run_clustering: bool = True,
) -> List[CompanyData]:
    """Enrich companies in-place with FinBERT sentiment and cluster labels.

    Args:
        companies: The universe to enrich.
        analyzer: A shared :class:`FinBERTSentiment`; created if not supplied.
        run_clustering: Whether to run PCA/K-Means clustering.

    Returns:
        The same list, with each company's ``derived`` dict populated.
    """
    analyzer = analyzer or FinBERTSentiment()

    for company in companies:
        risk_text = company.sec.get("sections", {}).get("risk_factors", "")
        risk_sent = analyzer.analyze(risk_text) if risk_text else None

        headlines = [
            a.get("title", "")
            for a in (company.gdelt.get("articles", []) or [])
            if a.get("title")
        ][:15]
        news_sent = analyzer.analyze_many(headlines) if headlines else None

        finbert_score = risk_sent.score if risk_sent else None
        news_score = news_sent.score if news_sent else None

        # Controversy: negative sentiment + regulatory keyword pressure.
        controversy = _controversy(company, finbert_score, news_score)

        company.derived.update(
            {
                "finbert_sentiment": finbert_score,
                "finbert_news_sentiment": news_score,
                "finbert_backend": analyzer.backend,
                "controversy_score": controversy,
            }
        )

    if run_clustering:
        clusters = cluster_companies(companies)
        labels = clusters.get("labels", {})
        coords = clusters.get("coords", {})
        for company in companies:
            company.derived["cluster"] = labels.get(company.ticker)
            company.derived["cluster_coords"] = coords.get(company.ticker)

    return companies


def _controversy(
    company: CompanyData,
    finbert_score: Optional[float],
    news_score: Optional[float],
) -> Optional[float]:
    """Blend sentiment and regulatory signals into a controversy score."""
    components: List[float] = []
    if finbert_score is not None:
        components.append(max(0.0, -finbert_score))  # negative risk sentiment
    if news_score is not None:
        components.append(max(0.0, -news_score))  # negative news
    reg = company.sec.get("flags", {}).get("regulatory_risk_count")
    if reg is not None:
        components.append(min(float(reg), 10.0) / 10.0)
    if not components:
        return None
    return sum(components) / len(components)
