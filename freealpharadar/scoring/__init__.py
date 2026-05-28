"""Multi-dimensional scoring framework ("Moat + Momentum + Misvaluation").

The scoring engine computes 30+ raw factors per company, z-normalises each
factor within the current universe, applies user-configurable weights, and
produces a transparent, fully drillable final score.
"""

from __future__ import annotations

from freealpharadar.scoring.engine import (
    DEFAULT_WEIGHTS,
    ScoreResult,
    ScoringEngine,
)
from freealpharadar.scoring.factors import FACTORS, FactorGroup, FactorSpec

__all__ = [
    "ScoringEngine",
    "ScoreResult",
    "DEFAULT_WEIGHTS",
    "FACTORS",
    "FactorSpec",
    "FactorGroup",
]
