"""QAOA angle optimization with shot-based energy estimates (DECISIONS.md ADR-020).

The optimizer only sees what hardware would give it: each evaluation draws S shots from the
QAOA output distribution and uses the sample mean of the QUBO energy,

    E_hat(gamma, beta) = (1/S) * sum_{s=1..S} E(x_s),   x_s ~ p_{gamma,beta}

and COBYLA minimizes E_hat within a fixed evaluation budget. The total sampling cost is
(evaluations x S) shots, which is what budget-matched comparisons count. The final distribution
is then scored exactly with :mod:`nisq_folio.qaoa.metrics`, so reported quality has no shot noise.

Initial angles follow a linear ramp (a common QAOA heuristic, chosen here, not tuned per
instance): for l = 0..p-1, gamma_l = Delta * (l + 1/2) / p and beta_l = Delta * (1 - (l + 1/2) / p).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from nisq_folio.problem import FloatArray, PortfolioProblem
from nisq_folio.qaoa.metrics import DistributionMetrics, SolutionLandscape, distribution_metrics
from nisq_folio.qaoa.statevector import NoiselessQaoa, hamiltonian_scale
from nisq_folio.qubo.encoding import build_qubo, qubo_to_ising


@dataclass(frozen=True)
class QaoaSettings:
    """Settings for :func:`run_noiseless_qaoa`.

    Attributes:
        layers: Depth p.
        shots: Shots S per energy evaluation.
        max_evaluations: COBYLA function-evaluation budget.
        rhobeg: COBYLA initial trust-region radius (radians).
        ramp_delta: Delta of the linear-ramp initial angles.
    """

    layers: int
    shots: int = 1024
    max_evaluations: int = 100
    rhobeg: float = 0.25
    ramp_delta: float = 0.75

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.layers < 1 or self.shots < 1:
            raise ValueError("layers and shots must be >= 1")
        # COBYLA needs at least (number of parameters + 2) evaluations and silently raises a
        # smaller budget, which would break budget accounting; reject it instead.
        if self.max_evaluations < 2 * self.layers + 2:
            raise ValueError(
                f"max_evaluations must be >= 2 * layers + 2 = {2 * self.layers + 2} for COBYLA"
            )
        if not (self.rhobeg > 0 and self.ramp_delta > 0):
            raise ValueError("rhobeg and ramp_delta must be > 0")


@dataclass(frozen=True)
class QaoaRun:
    """Outcome of one QAOA optimization.

    Attributes:
        gammas: Final cost angles.
        betas: Final mixer angles.
        n_evaluations: Energy evaluations actually used.
        total_shots: n_evaluations x shots.
        energy_trace: E_hat at every evaluation, in order.
        best_sampled: Best feasible objective seen in any sample (``None`` if none feasible).
        metrics: Exact metrics of the final output distribution.
    """

    gammas: FloatArray
    betas: FloatArray
    n_evaluations: int
    total_shots: int
    energy_trace: list[float] = field(repr=False)
    best_sampled: float | None
    metrics: DistributionMetrics

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (without the full trace)."""
        return {
            "gammas": [float(g) for g in self.gammas],
            "betas": [float(b) for b in self.betas],
            "n_evaluations": self.n_evaluations,
            "total_shots": self.total_shots,
            "final_energy_estimate": self.energy_trace[-1],
            "best_sampled": self.best_sampled,
            "metrics": self.metrics.as_dict(),
        }


def ramp_angles(layers: int, delta: float) -> tuple[FloatArray, FloatArray]:
    """Return linear-ramp initial angles (module docstring)."""
    fractions = (np.arange(layers) + 0.5) / layers
    return delta * fractions, delta * (1.0 - fractions)


def run_noiseless_qaoa(
    problem: PortfolioProblem,
    penalty: float,
    settings: QaoaSettings,
    rng: np.random.Generator,
    landscape: SolutionLandscape | None = None,
) -> QaoaRun:
    """Optimize X-mixer QAOA angles on the exact noiseless simulator with shot sampling.

    Args:
        problem: The portfolio problem.
        penalty: QUBO penalty weight P (ADR-017).
        settings: QAOA settings.
        rng: Seeded generator for shot sampling (CLAUDE.md rule 1).
        landscape: Precomputed landscape for scoring (computed if omitted).

    Returns:
        The run, scored exactly on its final distribution.
    """
    landscape = landscape or SolutionLandscape.from_problem(problem)
    qubo = build_qubo(problem, penalty)
    energies = qubo.all_energies()
    simulator = NoiselessQaoa(energies, hamiltonian_scale(qubo_to_ising(qubo)))
    p = settings.layers
    trace: list[float] = []
    best = np.inf

    def estimate(theta: FloatArray) -> float:
        nonlocal best
        probs = simulator.probabilities(theta[:p], theta[p:])
        counts = rng.multinomial(settings.shots, probs)
        seen = counts > 0
        hit = seen & landscape.feasible
        if hit.any():
            best = min(best, float(landscape.objective[hit].min()))
        value = float(counts @ energies / settings.shots)
        trace.append(value)
        return value

    gammas0, betas0 = ramp_angles(p, settings.ramp_delta)
    result = minimize(
        estimate,
        np.concatenate([gammas0, betas0]),
        method="COBYLA",
        options={"maxiter": settings.max_evaluations, "rhobeg": settings.rhobeg},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    final = simulator.probabilities(theta[:p], theta[p:])
    return QaoaRun(
        gammas=theta[:p],
        betas=theta[p:],
        n_evaluations=len(trace),
        total_shots=len(trace) * settings.shots,
        energy_trace=trace,
        best_sampled=None if not np.isfinite(best) else best,
        metrics=distribution_metrics(final, landscape),
    )
