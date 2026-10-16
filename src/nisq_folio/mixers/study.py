"""Mixer study grid and its pre-registered per-mixer selection rule (DECISIONS.md ADR-021).

Each mixer gets its own best configuration, chosen exactly as in ADR-020: for each candidate,

    score = mean over N of ln( median over tuning instances of A )        (A from ADR-019)

and among candidates within ``SCORE_TOLERANCE`` of that mixer's best score pick, in order:
    1. the smallest depth p;
    2. the largest penalty scale c (only varies for the X mixer);
    3. COBYLA before SPSA (the ADR-020 baseline optimizer, unless SPSA is clearly better).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nisq_folio.qaoa.tuning import SCORE_TOLERANCE

_OPTIMIZER_ORDER = {"cobyla": 0, "spsa": 1}


@dataclass(frozen=True, order=True)
class MixerConfiguration:
    """One point of the mixer-study grid."""

    mixer: str
    optimizer: str
    layers: int
    penalty_scale: float

    def label(self) -> str:
        """Short label, e.g. ``xy_ring spsa p=2`` (penalty shown for the X mixer only)."""
        base = f"{self.mixer} {self.optimizer} p={self.layers}"
        return f"{base} c={self.penalty_scale:g}" if self.mixer == "x" else base


def select_per_mixer(
    scores: Mapping[MixerConfiguration, float],
) -> dict[str, MixerConfiguration]:
    """Apply the selection rule separately for each mixer (module docstring).

    Args:
        scores: Score of every configuration in the grid.

    Returns:
        Mixer name -> selected configuration.
    """
    if not scores:
        raise ValueError("no configurations to select from")
    selected: dict[str, MixerConfiguration] = {}
    for mixer in sorted({c.mixer for c in scores}):
        own = {c: s for c, s in scores.items() if c.mixer == mixer}
        best = max(own.values())
        candidates = [c for c, s in own.items() if s >= best - SCORE_TOLERANCE]
        selected[mixer] = min(
            candidates,
            key=lambda c: (c.layers, -c.penalty_scale, _OPTIMIZER_ORDER[c.optimizer]),
        )
    return selected
