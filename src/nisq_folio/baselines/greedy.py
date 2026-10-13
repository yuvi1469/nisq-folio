"""Greedy forward selection."""

from __future__ import annotations

import numpy as np

from nisq_folio.problem import PortfolioProblem, Solution, make_solution


def greedy(problem: PortfolioProblem) -> Solution:
    """Add assets one at a time, each time choosing the one that lowers f(x) the most.

    With a = lambda / K^2 and b = 1 / K (fixed at the target K throughout), adding asset j to
    the current selection x changes the objective by

        delta_j = a * (2 * (Sigma x)_j + Sigma_jj) - b * mu_j

    The asset with the smallest delta_j is added; ties go to the lowest index. After K steps
    the selection is feasible. This is deterministic and needs no seed.

    Args:
        problem: The problem instance.

    Returns:
        The greedy solution. ``n_evaluations`` counts every delta_j computed.
    """
    a, b = problem.quadratic_coefficient, problem.linear_coefficient
    sigma, mu = problem.sigma, problem.mu
    selected = np.zeros(problem.n, dtype=bool)
    sigma_x = np.zeros(problem.n)
    evaluations = 0
    for _ in range(problem.k):
        deltas = a * (2.0 * sigma_x + np.diag(sigma)) - b * mu
        deltas[selected] = np.inf
        evaluations += int((~selected).sum())
        j = int(np.argmin(deltas))
        selected[j] = True
        sigma_x += sigma[:, j]
    return make_solution(problem, np.flatnonzero(selected), "greedy", evaluations)
