"""Noisy QAOA runs with the same protocol as the noiseless variants (DECISIONS.md ADR-023).

Identical to :func:`nisq_folio.mixers.variants.run_variant` in everything except where the
output distribution comes from: here, :class:`NoisyEvaluator` (exact density matrix with the
scaled noise model, plus readout error) instead of the NumPy state vector. Shot sampling,
optimizers, ramp initialization, ADAPT operator selection and exact final scoring are the same.

Cost bookkeeping adds ``circuit_runs``: the number of noisy circuit simulations, which is what
dominates wall time (ADAPT needs 2 * |pool| extra runs per layer for operator selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nisq_folio.mixers.variants import VariantSettings, adapt_pool
from nisq_folio.noise.circuits import objective_ising
from nisq_folio.noise.density import NoisyEvaluator, ReferenceRates
from nisq_folio.problem import FloatArray, PortfolioProblem
from nisq_folio.qaoa.metrics import DistributionMetrics, SolutionLandscape, distribution_metrics
from nisq_folio.qaoa.optimize import ramp_angles
from nisq_folio.qaoa.optimizers import run_cobyla, run_spsa
from nisq_folio.qubo.encoding import build_qubo, qubo_to_ising


@dataclass(frozen=True)
class NoisyRun:
    """Outcome of one noisy run (fields as in ``VariantRun``, plus noise bookkeeping)."""

    noise_level: float
    gammas: FloatArray
    betas: FloatArray
    n_evaluations: int
    optimization_shots: int
    selection_shots: int
    circuit_runs: int
    two_qubit_gates: int
    mixer_sequence: tuple[str, ...]
    metrics: DistributionMetrics

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "noise_level": self.noise_level,
            "gammas": [float(g) for g in self.gammas],
            "betas": [float(b) for b in self.betas],
            "n_evaluations": self.n_evaluations,
            "optimization_shots": self.optimization_shots,
            "selection_shots": self.selection_shots,
            "total_shots": self.optimization_shots + self.selection_shots,
            "circuit_runs": self.circuit_runs,
            "two_qubit_gates": self.two_qubit_gates,
            "mixer_sequence": list(self.mixer_sequence),
            "metrics": self.metrics.as_dict(),
        }


def make_evaluator(
    problem: PortfolioProblem,
    penalty_bound: float,
    settings: VariantSettings,
    level: float,
    rates: ReferenceRates,
    backend: str = "FakeKingston",
    seed_transpiler: int = 0,
) -> tuple[NoisyEvaluator, FloatArray]:
    """Build the noisy evaluator and the cost energies for a problem and mixer.

    The cost Hamiltonian and scale match the noiseless variant exactly: the penalized QUBO for
    the X mixer, the objective alone for XY/ADAPT, and s = std of the energies.
    """
    landscape_energies: FloatArray
    if settings.mixer == "x":
        qubo = build_qubo(problem, settings.penalty_scale * penalty_bound)
        ising, landscape_energies, initial = qubo_to_ising(qubo), qubo.all_energies(), "plus"
    else:
        ising = objective_ising(
            problem.sigma, problem.mu, problem.quadratic_coefficient, problem.linear_coefficient
        )
        landscape_energies = SolutionLandscape.from_problem(problem).objective
        initial = "dicke"
    evaluator = NoisyEvaluator(
        ising, float(landscape_energies.std()), initial, problem.k, level, rates,
        backend=backend, seed_transpiler=seed_transpiler,
    )  # fmt: skip
    return evaluator, landscape_energies


def run_noisy_variant(
    problem: PortfolioProblem,
    penalty_bound: float,
    settings: VariantSettings,
    level: float,
    rates: ReferenceRates,
    rng: np.random.Generator,
    landscape: SolutionLandscape | None = None,
    backend: str = "FakeKingston",
    seed_transpiler: int = 0,
) -> NoisyRun:
    """Run one QAOA variant under noise level m and score its final distribution exactly.

    Args:
        problem: The portfolio problem.
        penalty_bound: Provable penalty of ADR-017 (scaled by c for the X mixer).
        settings: Mixer, optimizer and protocol settings (as for the noiseless run).
        level: Noise level m.
        rates: Reference error rates.
        rng: Seeded generator for shots and SPSA directions.
        landscape: Precomputed scoring landscape.
        backend: Fake backend supplying the coupling map.
        seed_transpiler: Transpiler seed.

    Returns:
        The noisy run.
    """
    landscape = landscape or SolutionLandscape.from_problem(problem)
    evaluator, energies = make_evaluator(
        problem, penalty_bound, settings, level, rates, backend, seed_transpiler
    )
    p = settings.layers
    evaluations = 0

    def estimate(probs: FloatArray) -> float:
        counts = rng.multinomial(settings.shots, probs)
        return float(counts @ energies / settings.shots)

    def optimize(theta: FloatArray, names: list[str], budget: int) -> FloatArray:
        layers = len(names)

        def fun(t: FloatArray) -> float:
            nonlocal evaluations
            evaluations += 1
            return estimate(evaluator.distribution(names, t[:layers], t[layers:]))

        if settings.optimizer == "cobyla":
            return run_cobyla(fun, theta, budget, settings.rhobeg)
        return run_spsa(fun, theta, budget, rng, settings.spsa)

    gammas0, betas0 = ramp_angles(p, settings.ramp_delta)
    selection_shots = 0
    if settings.mixer != "adapt":
        names = [settings.mixer] * p
        theta = optimize(np.concatenate([gammas0, betas0]), names, settings.max_evaluations)
    else:
        pool = list(adapt_pool(problem.n))
        names = []
        gammas: list[float] = []
        betas: list[float] = []
        delta = settings.adapt_gradient_shift
        for layer in range(p):
            trial_g = np.array([*gammas, gammas0[layer]])
            gradients = {}
            for op in pool:
                plus = evaluator.distribution([*names, op], trial_g, np.array([*betas, delta]))
                minus = evaluator.distribution([*names, op], trial_g, np.array([*betas, -delta]))
                gradients[op] = (estimate(plus) - estimate(minus)) / (2 * delta)
            selection_shots += 2 * len(pool) * settings.shots
            chosen = max(gradients, key=lambda name: abs(gradients[name]))  # first on ties
            names.append(chosen)
            gammas.append(float(gammas0[layer]))
            betas.append(float(-np.sign(gradients[chosen]) * betas0[layer]))
            budget = (settings.max_evaluations - evaluations) // (p - layer)
            theta = optimize(np.array(gammas + betas), names, budget)
            gammas, betas = list(theta[: layer + 1]), list(theta[layer + 1 :])
        theta = np.array(gammas + betas)

    final = evaluator.distribution(names, theta[:p], theta[p:])
    return NoisyRun(
        noise_level=level,
        gammas=np.asarray(theta[:p]),
        betas=np.asarray(theta[p:]),
        n_evaluations=evaluations,
        optimization_shots=evaluations * settings.shots,
        selection_shots=selection_shots,
        circuit_runs=evaluator.circuit_runs,
        two_qubit_gates=evaluator.two_qubit_gates(names),
        mixer_sequence=tuple(names),
        metrics=distribution_metrics(final, landscape),
    )
