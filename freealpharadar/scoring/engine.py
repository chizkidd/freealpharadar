"""The scoring engine.

Combines the factor library, cross-sectional normalisation and user-supplied
weights into a single, transparent composite score per company. Every factor's
**raw value**, **normalised z-score** and **weighted contribution** is retained
so the dashboard can drill all the way down -- nothing is a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from freealpharadar.pipeline import CompanyData
from freealpharadar.scoring.factors import FACTORS, FactorGroup, FactorSpec
from freealpharadar.scoring.normalize import normalize_matrix, to_0_100
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

# Equal-weight default: every factor contributes equally until the user tunes
# the sliders in the UI.
DEFAULT_WEIGHTS: Dict[str, float] = {f.name: 1.0 for f in FACTORS}


@dataclass
class FactorContribution:
    """The per-factor breakdown for one company.

    Attributes:
        name: Factor machine name.
        label: Human-readable label.
        group: Factor group.
        raw: Raw extracted value (``None`` if missing).
        zscore: Normalised, orientation-adjusted z-score.
        weight: Weight applied to this factor.
        contribution: ``weight * zscore`` -- the factor's signed push on the
            composite score (in z-units, pre-rescale).
    """

    name: str
    label: str
    group: str
    raw: Optional[float]
    zscore: float
    weight: float
    contribution: float


@dataclass
class ScoreResult:
    """Full scoring result for a single company.

    Attributes:
        ticker: Ticker symbol.
        name: Company name.
        sector: Sector label.
        market_cap: Market capitalisation in USD.
        score: Final composite score on a 0-100 scale.
        raw_composite: The pre-rescale weighted-average z-score.
        contributions: Per-factor breakdown, ordered by absolute contribution.
        group_scores: Average z-score within each factor group.
        cluster: Optional K-Means cluster label.
        warnings: Data warnings carried from ingestion.
    """

    ticker: str
    name: str
    sector: str
    market_cap: Optional[float]
    score: float
    raw_composite: float
    contributions: List[FactorContribution] = field(default_factory=list)
    group_scores: Dict[str, float] = field(default_factory=dict)
    cluster: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


class ScoringEngine:
    """Compute composite scores for a universe of companies.

    Args:
        weights: Mapping of factor name -> weight. Missing factors default to
            their :data:`DEFAULT_WEIGHTS` value. Negative weights are permitted
            (e.g. to invert a factor's orientation).
        factors: Factor specifications to use; defaults to the full registry.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        factors: Optional[List[FactorSpec]] = None,
    ) -> None:
        self.factors = factors or FACTORS
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            merged.update(weights)
        self.weights = merged

    # ------------------------------------------------------------------ #
    def score_universe(self, companies: List[CompanyData]) -> List[ScoreResult]:
        """Score every company *relative to the current universe*.

        Normalisation is cross-sectional, so scores are only meaningful within
        the universe supplied here.

        Args:
            companies: The universe to score.

        Returns:
            A list of :class:`ScoreResult`, sorted by descending score.
        """
        if not companies:
            return []

        # 1. Extract the raw factor matrix: factor name -> [value per company].
        raw_matrix: Dict[str, List[Optional[float]]] = {}
        orientation: Dict[str, bool] = {}
        for spec in self.factors:
            orientation[spec.name] = spec.higher_is_better
            raw_matrix[spec.name] = [self._safe_eval(spec, c) for c in companies]

        # 2. Normalise each factor column across the universe.
        z_matrix = normalize_matrix(raw_matrix, orientation)

        # 3. Build per-company contributions and the weighted composite.
        composites: List[float] = []
        per_company_contribs: List[List[FactorContribution]] = []
        per_company_groups: List[Dict[str, float]] = []

        for idx, company in enumerate(companies):
            contribs: List[FactorContribution] = []
            total_weight = 0.0
            weighted_sum = 0.0
            group_acc: Dict[str, List[float]] = {}

            for spec in self.factors:
                weight = self.weights.get(spec.name, 1.0)
                z = z_matrix[spec.name][idx]
                contribution = weight * z
                contribs.append(
                    FactorContribution(
                        name=spec.name,
                        label=spec.label,
                        group=spec.group.value,
                        raw=raw_matrix[spec.name][idx],
                        zscore=z,
                        weight=weight,
                        contribution=contribution,
                    )
                )
                weighted_sum += contribution
                total_weight += abs(weight)
                group_acc.setdefault(spec.group.value, []).append(z)

            composite = weighted_sum / total_weight if total_weight else 0.0
            composites.append(composite)
            contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
            per_company_contribs.append(contribs)
            per_company_groups.append(
                {g: sum(v) / len(v) for g, v in group_acc.items()}
            )

        # 4. Rescale composites to a friendly 0-100 scale across the universe.
        final_scores = to_0_100(composites)

        results: List[ScoreResult] = []
        for idx, company in enumerate(companies):
            results.append(
                ScoreResult(
                    ticker=company.ticker,
                    name=company.name,
                    sector=company.sector,
                    market_cap=company.market_cap,
                    score=round(final_scores[idx], 2),
                    raw_composite=round(composites[idx], 4),
                    contributions=per_company_contribs[idx],
                    group_scores=per_company_groups[idx],
                    cluster=company.derived.get("cluster"),
                    warnings=company.warnings,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _safe_eval(spec: FactorSpec, company: CompanyData) -> Optional[float]:
        """Evaluate a factor, swallowing errors into ``None``."""
        try:
            val = spec.fn(company)
            if val is None:
                return None
            return float(val)
        except Exception as exc:  # noqa: BLE001
            logger.debug("factor %s failed for %s: %s", spec.name, company.ticker, exc)
            return None

    # ------------------------------------------------------------------ #
    def group_weight_summary(self) -> Dict[str, float]:
        """Return the total weight assigned to each factor group."""
        summary: Dict[str, float] = {g.value: 0.0 for g in FactorGroup}
        for spec in self.factors:
            summary[spec.group.value] += self.weights.get(spec.name, 1.0)
        return summary
