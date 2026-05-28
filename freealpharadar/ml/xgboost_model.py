"""Optional supervised breakout model (skeleton).

This is a deliberately thin scaffold. If the user supplies a CSV of historical
"breakout" labels (``ticker,label`` where ``label`` is 1 for a confirmed
breakout, 0 otherwise), an XGBoost classifier can be trained on the engine's
factor matrix to *re-rank* companies by predicted breakout probability.

When **no labels exist** -- the default -- the system uses rule-based scoring
with zero degradation: :func:`predict_breakout_probability` simply returns
``None`` and callers fall back to the composite score. This keeps the
zero-shot, no-training promise intact while leaving a clear extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from freealpharadar.pipeline import CompanyData
from freealpharadar.scoring.factors import FACTORS
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


def _factor_row(company: CompanyData) -> List[float]:
    """Build a dense factor-feature row, imputing missing values to 0."""
    row: List[float] = []
    for spec in FACTORS:
        try:
            val = spec.fn(company)
            row.append(float(val) if val is not None else 0.0)
        except Exception:  # noqa: BLE001
            row.append(0.0)
    return row


@dataclass
class BreakoutModel:
    """A thin wrapper around an optional XGBoost classifier.

    Attributes:
        model: The fitted estimator, or ``None`` when untrained.
        feature_names: Ordered factor names used as features.
    """

    model: object = None
    feature_names: List[str] = field(default_factory=lambda: [f.name for f in FACTORS])

    @property
    def is_trained(self) -> bool:
        """Whether a usable model has been fitted."""
        return self.model is not None

    def train(self, companies: List[CompanyData], labels: Dict[str, int]) -> bool:
        """Fit the classifier on labelled companies.

        Args:
            companies: Universe providing the feature rows.
            labels: Mapping of ticker -> 0/1 breakout label.

        Returns:
            ``True`` if training succeeded, ``False`` otherwise (e.g. XGBoost
            not installed, too few labels, or a single class).
        """
        labelled = [(c, labels[c.ticker]) for c in companies if c.ticker in labels]
        if len(labelled) < 8 or len({lbl for _, lbl in labelled}) < 2:
            logger.info("Insufficient/degenerate labels; skipping XGBoost training.")
            return False
        try:
            import numpy as np
            from xgboost import XGBClassifier

            X = np.array([_factor_row(c) for c, _ in labelled], dtype=float)
            y = np.array([lbl for _, lbl in labelled], dtype=int)
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                eval_metric="logloss",
            )
            clf.fit(X, y)
            self.model = clf
            logger.info("Trained XGBoost breakout model on %d examples.", len(labelled))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("XGBoost training unavailable/failed: %s", exc)
            return False

    def predict_breakout_probability(self, company: CompanyData) -> Optional[float]:
        """Return P(breakout) for a company, or ``None`` if untrained."""
        if not self.is_trained:
            return None
        try:
            import numpy as np

            X = np.array([_factor_row(company)], dtype=float)
            proba = self.model.predict_proba(X)[0][1]  # type: ignore[attr-defined]
            return float(proba)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Breakout prediction failed: %s", exc)
            return None
