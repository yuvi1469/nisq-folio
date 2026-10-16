"""Choosing QAOA settings on tuning instances by a pre-registered rule (DECISIONS.md ADR-020).

Grid: depth p x penalty scale c (P = c * provable bound of ADR-017) x ramp Delta. For each
configuration and size N, take the median over tuning instances of the amplification over
random feasible guessing, A = p_opt / (1 / C(N, K)) (ADR-019). Then

    score(config) = mean over N of ln( median_N A(config) )

and select, among configurations with score >= best score - 0.1 (within ~10% in A):
    1. the smallest depth p (cheaper, and less noise, in the noisy experiments);
    2. then the largest penalty scale c (closest to the provable guarantee);
    3. then the smallest Delta.

The rule was fixed and committed before the tuning study ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

SCORE_TOLERANCE = 0.1
_FLOOR = 1e-12  # avoids log(0) when a configuration never puts mass on the optimum


@dataclass(frozen=True, order=True)
class TuningConfiguration:
    """One point of the tuning grid."""

    layers: int
    penalty_scale: float
    ramp_delta: float

    def label(self) -> str:
        """Short human-readable label, e.g. ``p=2 c=0.25 delta=1.5``."""
        return f"p={self.layers} c={self.penalty_scale:g} delta={self.ramp_delta:g}"


def configuration_score(amplifications_by_size: Mapping[int, Sequence[float]]) -> float:
    """Return mean over N of ln(median amplification) for one configuration.

    Args:
        amplifications_by_size: For each N, the amplification of every tuning instance.

    Returns:
        The score (higher is better).
    """
    if not amplifications_by_size or any(len(v) == 0 for v in amplifications_by_size.values()):
        raise ValueError("need at least one amplification for every size")
    logs = [np.log(max(float(np.median(v)), _FLOOR)) for v in amplifications_by_size.values()]
    return float(np.mean(logs))


def select_configuration(scores: Mapping[TuningConfiguration, float]) -> TuningConfiguration:
    """Apply the pre-registered selection rule (module docstring).

    Args:
        scores: Score of every configuration in the grid.

    Returns:
        The selected configuration.
    """
    if not scores:
        raise ValueError("no configurations to select from")
    best = max(scores.values())
    candidates = [c for c, s in scores.items() if s >= best - SCORE_TOLERANCE]
    return min(candidates, key=lambda c: (c.layers, -c.penalty_scale, c.ramp_delta))
