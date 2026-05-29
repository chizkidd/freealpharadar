"""Cross-sectional normalisation utilities.

Factors are measured in wildly different units (ratios, counts, booleans), so
before combining them we z-normalise each factor *within the current universe*.
A z-score expresses how many standard deviations a company sits above or below
the universe mean for that factor, which is exactly the "relative to peers"
lens an investor wants.

Missing values are imputed to the universe mean (i.e. a neutral z-score of 0)
so that a company is never unduly penalised or rewarded for absent data.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional


def zscore_column(
    values: List[Optional[float]], higher_is_better: bool = True
) -> List[float]:
    """Z-normalise a single factor across the universe.

    Args:
        values: Raw factor values, one per company. ``None`` means "missing".
        higher_is_better: When ``False`` the sign is flipped so that, after
            normalisation, larger scores always mean "more attractive".

    Returns:
        A list of z-scores aligned with ``values``. Missing inputs map to
        ``0.0`` (neutral). When every present value is identical (zero
        variance) all present entries map to ``0.0``.
    """
    present = [v for v in values if v is not None]
    if not present:
        return [0.0 for _ in values]

    mean = statistics.mean(present)
    stdev = statistics.pstdev(present) if len(present) > 1 else 0.0

    out: List[float] = []
    for v in values:
        if v is None or stdev == 0.0:
            out.append(0.0)
        else:
            z = (v - mean) / stdev
            out.append(z if higher_is_better else -z)
    return out


def clamp(value: float, low: float = -3.0, high: float = 3.0) -> float:
    """Clamp ``value`` to ``[low, high]`` to limit outlier influence."""
    return max(low, min(high, value))


def to_0_100(zscores: List[float]) -> List[float]:
    """Map clamped z-scores (roughly [-3, 3]) onto a friendly 0-100 scale.

    This is used only for the *final* composite score so the dashboard can show
    intuitive 0-100 values; intermediate factor scores keep their z units.
    """
    scaled = []
    for z in zscores:
        c = clamp(z)
        scaled.append((c + 3.0) / 6.0 * 100.0)
    return scaled


def normalize_matrix(
    raw: Dict[str, List[Optional[float]]],
    higher_is_better: Dict[str, bool],
) -> Dict[str, List[float]]:
    """Z-normalise every factor column in a raw factor matrix.

    Args:
        raw: Mapping of factor name -> list of raw values (one per company).
        higher_is_better: Mapping of factor name -> orientation flag.

    Returns:
        Mapping of factor name -> list of z-scores.
    """
    return {
        name: zscore_column(values, higher_is_better.get(name, True))
        for name, values in raw.items()
    }
