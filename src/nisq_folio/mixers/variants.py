"""Noiseless QAOA with a choice of mixer and optimizer (DECISIONS.md ADR-021).

Mixers:
    ``x``        transverse field, initial |+>^N, cost = QUBO with penalty P = c * bound (ADR-017)
    ``xy_ring``  parity-ordered XY ring, initial Dicke state, cost = objective f only (no penalty)
    ``adapt``    ADAPT-QAOA: initial Dicke state, cost = f only; layer l's mixer is chosen from
                 a pool of weight-preserving operators by the largest estimated energy gradient

ADAPT-QAOA follows the idea of Zhu et al., "Adaptive quantum approximate optimization algorithm
for solving combinatorial problems on a quantum computer" (Phys. Rev. Research, 2022; verify the
reference before citing). Here the pool is {XY ring} U {XY_ij : i < j}, restricted to
Hamming-weight-preserving operators so the state stays feasible. For layer l, with the cost
layer at its ramp angle gamma_l^0, each pool operator A gets a shot-based central difference

    g_A = [ E_hat(beta = +delta) - E_hat(beta = -delta) ] / (2 delta),

the operator with the largest |g_A| is appended with beta_l^0 = -sign(g_A) * ramp beta_l, and all
parameters are re-optimized with an equal share of the remaining evaluation budget. The
selection shots (2 * |pool| * S per layer) are counted in ``total_shots``.

Every energy estimate uses S shots from the exact distribution; the final distribution is scored
exactly (ADR-019). Hamiltonian scale s = std of the cost energies under uniform bitstrings (equal
to sqrt(sum J^2 + sum h^2), see ``qaoa.statevector``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nisq_folio.mixers.xy import apply_xy_pair, apply_xy_ring, dicke_state
from nisq_folio.problem import FloatArray, PortfolioProblem
from nisq_folio.qaoa.metrics import DistributionMetrics, SolutionLandscape, distribution_metrics
from nisq_folio.qaoa.optimize import ramp_angles
from nisq_folio.qaoa.optimizers import OPTIMIZERS, SpsaSettings, run_cobyla, run_spsa
from nisq_folio.qaoa.statevector import ComplexArray, apply_x_mixer
from nisq_folio.qubo.encoding import build_qubo

MIXERS = ("x", "xy_ring", "adapt")
Mixer = Callable[[ComplexArray, float], ComplexArray]


@dataclass(frozen=True)
class VariantSettings:
    """Settings for :func:`run_variant`.

    Attributes:
        mixer: One of :data:`MIXERS`.
        optimizer: One of ``OPTIMIZERS``.
        layers: Depth p.
        shots: Shots per energy estimate.
        max_evaluations: Optimizer evaluation budget (ADAPT selection shots are extra).
        rhobeg: COBYLA initial trust radius.
        ramp_delta: Linear-ramp scale for initial angles.
        penalty_scale: c in P = c * provable bound (X mixer only; ignored otherwise).
        spsa: SPSA hyperparameters.
        adapt_gradient_shift: delta for ADAPT's central-difference gradient.
    """

    mixer: str
    optimizer: str
    layers: int
    shots: int = 1024
    max_evaluations: int = 100
    rhobeg: float = 0.25
    ramp_delta: float = 0.75
    penalty_scale: float = 0.1
    spsa: SpsaSettings = field(default_factory=SpsaSettings)
    adapt_gradient_shift: float = np.pi / 8

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}, got {self.mixer!r}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.layers < 1 or self.shots < 1 or self.max_evaluations < 1:
            raise ValueError("layers, shots and max_evaluations must be >= 1")
        if not (self.penalty_scale > 0 and self.adapt_gradient_shift > 0):
            raise ValueError("penalty_scale and adapt_gradient_shift must be > 0")


@dataclass(frozen=True)
class VariantRun:
    """Outcome of one run; see :class:`nisq_folio.qaoa.optimize.QaoaRun` for shared fields.

    Attributes:
        gammas: Final cost angles.
        betas: Final mixer angles.
        n_evaluations: Optimizer evaluations used.
        optimization_shots: n_evaluations * shots.
        selection_shots: Shots spent choosing ADAPT operators (0 for fixed mixers).
        mixer_sequence: Name of the mixer used in each layer.
        metrics: Exact metrics of the final distribution.
    """

    gammas: FloatArray
    betas: FloatArray
    n_evaluations: int
    optimization_shots: int
    selection_shots: int
    mixer_sequence: tuple[str, ...]
    metrics: DistributionMetrics

    @property
    def total_shots(self) -> int:
        """All shots used: optimization plus ADAPT operator selection."""
        return self.optimization_shots + self.selection_shots

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "gammas": [float(g) for g in self.gammas],
            "betas": [float(b) for b in self.betas],
            "n_evaluations": self.n_evaluations,
            "optimization_shots": self.optimization_shots,
            "selection_shots": self.selection_shots,
            "total_shots": self.total_shots,
            "mixer_sequence": list(self.mixer_sequence),
            "metrics": self.metrics.as_dict(),
        }


def adapt_pool(n: int) -> dict[str, Mixer]:
    """Return ADAPT's operator pool: the XY ring and every single-pair XY rotation."""
    pool: dict[str, Mixer] = {"xy_ring": lambda s, b: apply_xy_ring(s, b, n)}
    for i in range(n):
        for j in range(i + 1, n):
            pool[f"xy_{i}_{j}"] = lambda s, b, i=i, j=j: apply_xy_pair(s, i, j, b, n)
    return pool


def evolve(
    initial: ComplexArray,
    phases: FloatArray,
    gammas: FloatArray,
    betas: FloatArray,
    mixers: list[Mixer],
) -> ComplexArray:
    """Apply prod_l M_l(beta_l) exp(-i gamma_l * phases) to ``initial`` (phases = E / s)."""
    psi = initial
    for gamma, beta, mix in zip(gammas, betas, mixers, strict=True):
        psi = mix(psi * np.exp(-1j * gamma * phases), float(beta))
    return psi


def run_variant(
    problem: PortfolioProblem,
    penalty_bound: float,
    settings: VariantSettings,
    rng: np.random.Generator,
    landscape: SolutionLandscape | None = None,
) -> VariantRun:
    """Run one noiseless QAOA variant and score its final distribution exactly.

    Args:
        problem: The portfolio problem.
        penalty_bound: The provable penalty of ADR-017 (scaled by c for the X mixer only).
        settings: Mixer, optimizer and protocol settings.
        rng: Seeded generator for shot sampling and SPSA directions (CLAUDE.md rule 1).
        landscape: Precomputed scoring landscape (computed if omitted).

    Returns:
        The run.
    """
    landscape = landscape or SolutionLandscape.from_problem(problem)
    n, k, p = problem.n, problem.k, settings.layers
    if settings.mixer == "x":
        energies = build_qubo(problem, settings.penalty_scale * penalty_bound).all_energies()
        initial = np.full(2**n, 2 ** (-n / 2), dtype=np.complex128)
    else:
        energies = landscape.objective  # feasible subspace: the penalty is identically zero
        initial = dicke_state(n, k)
    phases = energies / energies.std()
    evaluations = 0

    def estimate(psi: ComplexArray) -> float:
        probs = np.abs(psi) ** 2
        counts = rng.multinomial(settings.shots, probs / probs.sum())
        return float(counts @ energies / settings.shots)

    def optimize(theta: FloatArray, mixers: list[Mixer], budget: int) -> FloatArray:
        nonlocal evaluations
        layers = len(mixers)

        def fun(t: FloatArray) -> float:
            nonlocal evaluations
            evaluations += 1
            return estimate(evolve(initial, phases, t[:layers], t[layers:], mixers))

        if settings.optimizer == "cobyla":
            return run_cobyla(fun, theta, budget, settings.rhobeg)
        return run_spsa(fun, theta, budget, rng, settings.spsa)

    gammas0, betas0 = ramp_angles(p, settings.ramp_delta)
    selection_shots = 0
    if settings.mixer != "adapt":
        fixed: Mixer = (
            (lambda s, b: apply_x_mixer(s, b, n))
            if settings.mixer == "x"
            else (lambda s, b: apply_xy_ring(s, b, n))
        )
        mixers = [fixed] * p
        names = [settings.mixer] * p
        theta = optimize(np.concatenate([gammas0, betas0]), mixers, settings.max_evaluations)
    else:
        pool = adapt_pool(n)
        mixers, names = [], []
        gammas: list[float] = []
        betas: list[float] = []
        delta = settings.adapt_gradient_shift
        for layer in range(p):
            psi = evolve(initial, phases, np.array(gammas), np.array(betas), mixers)
            psi = psi * np.exp(-1j * gammas0[layer] * phases)
            gradients = {
                name: (estimate(op(psi, delta)) - estimate(op(psi, -delta))) / (2 * delta)
                for name, op in pool.items()
            }
            selection_shots += 2 * len(pool) * settings.shots
            chosen = max(gradients, key=lambda name: abs(gradients[name]))  # first on ties
            mixers.append(pool[chosen])
            names.append(chosen)
            gammas.append(float(gammas0[layer]))
            betas.append(float(-np.sign(gradients[chosen]) * betas0[layer]))
            budget = (settings.max_evaluations - evaluations) // (p - layer)
            theta = optimize(np.array(gammas + betas), mixers, budget)
            gammas, betas = list(theta[: layer + 1]), list(theta[layer + 1 :])
        theta = np.array(gammas + betas)

    final = evolve(initial, phases, theta[:p], theta[p:], mixers)
    probs = np.abs(final) ** 2
    return VariantRun(
        gammas=np.asarray(theta[:p]),
        betas=np.asarray(theta[p:]),
        n_evaluations=evaluations,
        optimization_shots=evaluations * settings.shots,
        selection_shots=selection_shots,
        mixer_sequence=tuple(names),
        metrics=distribution_metrics(probs / probs.sum(), landscape),
    )
