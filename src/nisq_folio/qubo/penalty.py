"""Choosing the penalty weight P for the cardinality constraint (DECISIONS.md ADR-017).

Guarantee used: let f_ub >= f* be the objective of any feasible selection (greedy's), and for
each cardinality m let L_m <= f(x) for every x with exactly m ones. If

    P > max_{m != K} (f_ub - L_m) / (m - K)^2,

then every infeasible x (with m = 1^T x != K) has E(x) >= L_m + P (m - K)^2 > f_ub >= f* = E(x*),
so the QUBO's global minimum is the constrained optimum. The per-cardinality lower bound is

    L_m = a * lambda_min(Sigma) * m - b * (sum of the m largest mu_i)

because x^T Sigma x >= lambda_min * ||x||^2 = lambda_min * m and mu^T x <= the top-m sum.
Only cheap quantities are used; the optimum f* itself is never needed.
"""

from __future__ import annotations

import numpy as np

from nisq_folio.baselines.brute_force import enumerate_objectives
from nisq_folio.baselines.greedy import greedy
from nisq_folio.problem import FloatArray, PortfolioProblem
from nisq_folio.qubo.encoding import all_bitstrings


def cardinality_lower_bounds(problem: PortfolioProblem) -> FloatArray:
    """Return L_m <= min{f(x) : 1^T x = m} for m = 0..N (module docstring).

    Args:
        problem: The portfolio problem.

    Returns:
        Array of length N + 1 with L_m = a * lambda_min(Sigma) * m - b * top_m(mu).
    """
    lambda_min = float(np.linalg.eigvalsh(problem.sigma).min())
    m = np.arange(problem.n + 1)
    top_sums = np.concatenate([[0.0], np.cumsum(np.sort(problem.mu)[::-1])])
    return problem.quadratic_coefficient * lambda_min * m - problem.linear_coefficient * top_sums


def sufficient_penalty(problem: PortfolioProblem, margin: float = 0.01) -> float:
    """Return P = (1 + margin) * max_{m != K} (f_greedy - L_m) / (m - K)^2.

    This P provably makes the QUBO's global minimum the constrained optimum (module docstring).

    Args:
        problem: The portfolio problem.
        margin: Relative safety margin > 0 that makes the required inequality strict.

    Returns:
        The penalty weight P > 0.
    """
    if not margin > 0:
        raise ValueError(f"margin must be > 0, got {margin}")
    f_ub = greedy(problem).objective
    bounds = cardinality_lower_bounds(problem)
    m = np.arange(problem.n + 1)
    infeasible = m != problem.k
    required = (f_ub - bounds[infeasible]) / (m[infeasible] - problem.k) ** 2
    # If every requirement is <= 0, any positive P is exact; return the smallest positive float.
    return float((1.0 + margin) * max(required.max(), np.finfo(float).tiny))


def minimal_exact_penalty(problem: PortfolioProblem) -> float:
    """Return the smallest P such that no infeasible x has E(x) < f* (by brute force).

        P_exact = max over infeasible x of (f* - f(x)) / (1^T x - K)^2,   floored at 0

    Enumerates all 2^N bitstrings, so this is for analysis and validation only (N <= ~20).
    It is NOT used to build QUBOs, because it requires knowing the optimum f*.

    Args:
        problem: The portfolio problem.

    Returns:
        P_exact >= 0. Any P > P_exact makes the QUBO minimum equal the constrained optimum.
    """
    f_star = float(enumerate_objectives(problem)[1].min())
    bits = all_bitstrings(problem.n)
    a, b = problem.quadratic_coefficient, problem.linear_coefficient
    f_all = a * np.einsum("ki,ij,kj->k", bits, problem.sigma, bits) - b * bits @ problem.mu
    violation = (bits.sum(axis=1) - problem.k) ** 2
    infeasible = violation > 0
    ratios = (f_star - f_all[infeasible]) / violation[infeasible]
    return float(max(ratios.max(), 0.0))
