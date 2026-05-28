"""Peer clustering via PCA + K-Means.

Reduces a handful of financial ratios to two principal components and clusters
companies with K-Means, so the user can spot names that behave differently from
their nominal sector peers (a classic source of misvaluation). The function is
robust to tiny universes and missing data, and never raises into the UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from freealpharadar.pipeline import CompanyData
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

# Ratios used as clustering features (extracted from yfinance key metrics/info).
_FEATURES = (
    "gross_margins",
    "profit_margins",
    "revenue_growth",
    "trailing_pe",
    "price_to_book",
)


def _feature_vector(company: CompanyData) -> List[Optional[float]]:
    """Build a raw feature vector for one company."""
    metrics = company.yfinance.get("key_metrics", {}) or {}
    vec: List[Optional[float]] = []
    for feat in _FEATURES:
        val = metrics.get(feat)
        try:
            vec.append(float(val) if val is not None else None)
        except (TypeError, ValueError):
            vec.append(None)
    return vec


def cluster_companies(
    companies: List[CompanyData], n_clusters: int = 4
) -> Dict[str, Any]:
    """Cluster companies on financial ratios using PCA + K-Means.

    Args:
        companies: Universe to cluster.
        n_clusters: Desired number of clusters (auto-reduced for small
            universes).

    Returns:
        A dict with ``labels`` (ticker -> cluster id), ``coords`` (ticker ->
        ``[pc1, pc2]``) and ``backend`` describing what was used. On any
        failure an empty-but-valid structure is returned.
    """
    empty = {"labels": {}, "coords": {}, "backend": "none"}
    if len(companies) < 3:
        return empty

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        logger.warning("scikit-learn unavailable for clustering: %s", exc)
        return empty

    tickers = [c.ticker for c in companies]
    raw = np.array(
        [
            [v if v is not None else np.nan for v in _feature_vector(c)]
            for c in companies
        ],
        dtype=float,
    )

    # Drop all-NaN columns to keep the imputer/PCA well-conditioned.
    valid_cols = ~np.all(np.isnan(raw), axis=0)
    raw = raw[:, valid_cols]
    if raw.shape[1] == 0:
        return empty

    try:
        imputed = SimpleImputer(strategy="mean").fit_transform(raw)
        scaled = StandardScaler().fit_transform(imputed)
        n_components = min(2, scaled.shape[1])
        coords = PCA(n_components=n_components).fit_transform(scaled)
        if n_components == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        k = max(2, min(n_clusters, len(companies) - 1))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Clustering failed: %s", exc)
        return empty

    return {
        "labels": {t: int(lbl) for t, lbl in zip(tickers, labels)},
        "coords": {
            t: [float(coords[i][0]), float(coords[i][1])] for i, t in enumerate(tickers)
        },
        "backend": "pca+kmeans",
    }
