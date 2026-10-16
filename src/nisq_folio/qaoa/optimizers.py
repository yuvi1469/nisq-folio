"""Classical optimizers for QAOA angles under a fixed evaluation budget (DECISIONS.md ADR-021).

Both minimize a noisy objective ``fun`` using at most ``budget`` calls and return the final
parameters; the caller counts the calls it actually received.

SPSA (simultaneous perturbation stochastic approximation, Spall 1992) estimates the whole
gradient from two evaluations per iteration, which makes it robust to shot noise:

    g_k = [ f(x_k + c_k D_k) - f(x_k - c_k D_k) ] / (2 c_k) * D_k,   D_k in {-1, +1}^d (Rademacher)
    x_{k+1} = x_k - a_k g_k,   a_k = a / (k + 1 + A)^0.602,   c_k = c / (k + 1)^0.101,  A = 0.1 K

The step size a is calibrated so that the first step has magnitude ``target_step``, from
``calibration_samples`` random perturbations at x_0 (2 evaluations each, counted in the budget):

    a = target_step * (A + 1)^0.602 / mean_m | f(x_0 + c D_m) - f(x_0 - c D_m) | / (2 c)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from nisq_folio.problem import FloatArray

Objective = Callable[[FloatArray], float]
OPTIMIZERS = ("cobyla", "spsa")


@dataclass(frozen=True)
class SpsaSettings:
    """SPSA hyperparameters (module docstring)."""

    target_step: float = 0.2
    perturbation: float = 0.1
    calibration_samples: int = 5

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not (self.target_step > 0 and self.perturbation > 0 and self.calibration_samples >= 1):
            raise ValueError("SPSA settings must be positive")


def minimum_budget(optimizer: str, dimension: int, spsa: SpsaSettings | None = None) -> int:
    """Smallest evaluation budget each optimizer can run with for ``dimension`` parameters."""
    if optimizer == "cobyla":
        return dimension + 2
    if optimizer == "spsa":
        return 2 * (spsa or SpsaSettings()).calibration_samples + 2
    raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")


def run_cobyla(fun: Objective, x0: FloatArray, budget: int, rhobeg: float) -> FloatArray:
    """Minimize with SciPy COBYLA using at most ``budget`` evaluations (may stop earlier)."""
    if budget < minimum_budget("cobyla", len(x0)):
        raise ValueError(f"COBYLA needs a budget of at least {len(x0) + 2}")
    result = minimize(fun, x0, method="COBYLA", options={"maxiter": budget, "rhobeg": rhobeg})
    return np.asarray(result.x, dtype=np.float64)


def run_spsa(
    fun: Objective,
    x0: FloatArray,
    budget: int,
    rng: np.random.Generator,
    settings: SpsaSettings | None = None,
) -> FloatArray:
    """Minimize with calibrated SPSA using exactly 2*m + 2*K <= ``budget`` evaluations.

    Args:
        fun: Noisy objective.
        x0: Starting parameters.
        budget: Maximum number of evaluations.
        rng: Seeded generator for the perturbation directions (CLAUDE.md rule 1).
        settings: SPSA hyperparameters.

    Returns:
        The final iterate x_K.
    """
    settings = settings or SpsaSettings()
    if budget < minimum_budget("spsa", len(x0), settings):
        raise ValueError(f"SPSA needs a budget of at least {2 * settings.calibration_samples + 2}")
    x = np.array(x0, dtype=np.float64)
    d, c = x.size, settings.perturbation
    iterations = (budget - 2 * settings.calibration_samples) // 2
    stability = 0.1 * iterations

    magnitudes = []
    for _ in range(settings.calibration_samples):
        delta = rng.choice((-1.0, 1.0), size=d)
        magnitudes.append(abs(fun(x + c * delta) - fun(x - c * delta)) / (2 * c))
    mean_magnitude = max(float(np.mean(magnitudes)), np.finfo(float).tiny)
    a = settings.target_step * (stability + 1) ** 0.602 / mean_magnitude

    for k in range(iterations):
        a_k = a / (k + 1 + stability) ** 0.602
        c_k = c / (k + 1) ** 0.101
        delta = rng.choice((-1.0, 1.0), size=d)
        slope = (fun(x + c_k * delta) - fun(x - c_k * delta)) / (2 * c_k)
        x = x - a_k * slope * delta
    return x
