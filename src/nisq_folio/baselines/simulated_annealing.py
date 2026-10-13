"""Simulated annealing over feasible selections using swap moves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nisq_folio.problem import FloatArray, PortfolioProblem, Solution, make_solution


@dataclass(frozen=True)
class AnnealingSettings:
    """Settings for :func:`simulated_annealing`.

    Attributes:
        n_steps: Number of proposed swap moves.
        initial_acceptance: Target probability of accepting an average uphill move at the
            start; sets the initial temperature T_0.
        final_temperature_ratio: T_final / T_0 for the geometric cooling schedule.
        n_calibration_moves: Random swaps sampled to estimate the average uphill delta.
    """

    n_steps: int = 2000
    initial_acceptance: float = 0.8
    final_temperature_ratio: float = 1e-3
    n_calibration_moves: int = 100

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.n_steps < 1 or self.n_calibration_moves < 1:
            raise ValueError("n_steps and n_calibration_moves must be >= 1")
        if not 0.0 < self.initial_acceptance < 1.0:
            raise ValueError("initial_acceptance must be in (0, 1)")
        if not 0.0 < self.final_temperature_ratio <= 1.0:
            raise ValueError("final_temperature_ratio must be in (0, 1]")


def swap_delta(
    problem: PortfolioProblem, sigma_x: FloatArray, out_index: int, in_index: int
) -> float:
    """Change in f when asset ``out_index`` is removed and ``in_index`` is added.

    For x' = x - e_i + e_j, with a = lambda / K^2, b = 1 / K and s = Sigma x:

        f(x') - f(x) = a * (2 s_j - 2 s_i + Sigma_ii + Sigma_jj - 2 Sigma_ij) - b * (mu_j - mu_i)

    Args:
        problem: The problem instance.
        sigma_x: s = Sigma x for the current selection x.
        out_index: i, currently selected.
        in_index: j, currently not selected.

    Returns:
        The objective change, computed in O(1).
    """
    s, m = problem.sigma, problem.mu
    i, j = out_index, in_index
    quad = 2.0 * sigma_x[j] - 2.0 * sigma_x[i] + s[i, i] + s[j, j] - 2.0 * s[i, j]
    return float(problem.quadratic_coefficient * quad - problem.linear_coefficient * (m[j] - m[i]))


def simulated_annealing(
    problem: PortfolioProblem,
    rng: np.random.Generator,
    settings: AnnealingSettings | None = None,
) -> Solution:
    """Minimize f(x) by simulated annealing restricted to feasible selections.

    Starts from a uniformly random K-subset. Each step proposes swapping one selected asset
    for one unselected asset (both uniform), which keeps sum(x) = K. A move with change
    delta is accepted with the Metropolis probability min(1, exp(-delta / T_t)), where

        T_t = T_0 * r^(t / (n_steps - 1)),   r = final_temperature_ratio

    and T_0 = -mean(positive deltas of random calibration swaps) / ln(initial_acceptance).
    The best selection seen at any step is returned.

    Args:
        problem: The problem instance.
        rng: Source of randomness; pass a seeded Generator (CLAUDE.md rule 1).
        settings: Annealing settings; defaults to :class:`AnnealingSettings`.

    Returns:
        The best solution found. ``n_evaluations`` counts calibration and step deltas.
    """
    settings = settings or AnnealingSettings()
    n, k = problem.n, problem.k
    selected = np.zeros(n, dtype=bool)
    selected[rng.choice(n, size=k, replace=False)] = True
    sigma_x = problem.sigma @ selected.astype(np.float64)
    current = problem.objective(selected.astype(np.int64))

    def propose() -> tuple[int, int]:
        return int(rng.choice(np.flatnonzero(selected))), int(rng.choice(np.flatnonzero(~selected)))

    uphill = [
        d
        for d in (
            swap_delta(problem, sigma_x, *propose()) for _ in range(settings.n_calibration_moves)
        )
        if d > 0
    ]
    # If no sampled move is uphill (e.g. a flat objective), any positive T_0 works; use a tiny one.
    mean_uphill = float(np.mean(uphill)) if uphill else 1e-12
    t0 = -mean_uphill / np.log(settings.initial_acceptance)
    evaluations = settings.n_calibration_moves

    best_value, best_selection = current, selected.copy()
    denominator = max(settings.n_steps - 1, 1)
    for step in range(settings.n_steps):
        temperature = t0 * settings.final_temperature_ratio ** (step / denominator)
        i, j = propose()
        delta = swap_delta(problem, sigma_x, i, j)
        evaluations += 1
        if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
            selected[i], selected[j] = False, True
            sigma_x += problem.sigma[:, j] - problem.sigma[:, i]
            current += delta
            if current < best_value:
                best_value, best_selection = current, selected.copy()
    return make_solution(
        problem, np.flatnonzero(best_selection), "simulated_annealing", evaluations
    )
