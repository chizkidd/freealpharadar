"""Zero-shot ML enhancements.

* :mod:`freealpharadar.ml.finbert` -- pre-trained FinBERT sentiment with a
  lexicon-based offline fallback.
* :mod:`freealpharadar.ml.clustering` -- PCA + K-Means peer clustering.
* :mod:`freealpharadar.ml.xgboost_model` -- optional supervised breakout model
  skeleton that degrades gracefully to rule-based scoring.
* :mod:`freealpharadar.ml.enrich` -- glue that attaches ML outputs to company
  records before scoring.
"""

from __future__ import annotations

from freealpharadar.ml.clustering import cluster_companies
from freealpharadar.ml.enrich import enrich_companies
from freealpharadar.ml.finbert import FinBERTSentiment, SentimentResult

__all__ = [
    "FinBERTSentiment",
    "SentimentResult",
    "cluster_companies",
    "enrich_companies",
]
