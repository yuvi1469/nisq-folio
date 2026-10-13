"""Exact solution by enumerating every K-subset of N assets."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from nisq_folio.problem import FloatArray, IntArray, PortfolioProblem, Solution, make_solution

MAX_SUBSETS = 5_000_000
_CHUNK = 100_000


def all_subsets(n: int, k: int) -> IntArray:
    """Return every K-subset of range(N) in lexicographic order, shape (C(N, K), K)."""
    return np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(-1, k)


def enumerate_objectives(
    problem: PortfolioProblem, max_subsets: int = MAX_SUBSETS
) -> tuple[IntArray, FloatArray]:
    """Evaluate f(x) for all C(N, K) feasible selections.

    Args:
        problem: The problem instance.
        max_subsets: Refuse to enumerate more subsets than this (guards memory and time).

    Returns:
        ``(subsets, values)``: subsets in lexicographic order, shape (C(N, K), K), and their
        objective values, shape (C(N, K),).

    Raises:
        ValueError: If C(N, K) exceeds ``max_subsets``.
    """
    count = comb(problem.n, problem.k)
    if count > max_subsets:
        raise ValueError(f"C({problem.n}, {problem.k}) = {count} exceeds max_subsets={max_subsets}")
    subsets = all_subsets(problem.n, problem.k)
    values = np.concatenate(
        [
            problem.objective_of_subsets(subsets[start : start + _CHUNK])
            for start in range(0, count, _CHUNK)
        ]
    )
    return subsets, values


def brute_force(problem: PortfolioProblem, max_subsets: int = MAX_SUBSETS) -> Solution:
    """Return the exact minimizer of f(x) over all K-subsets.

    Ties are broken by lexicographic order of the subsets (``np.argmin`` returns the first).

    Args:
        problem: The problem instance.
        max_subsets: Enumeration limit; see :func:`enumerate_objectives`.

    Returns:
        The optimal solution. ``n_evaluations`` is C(N, K).
    """
    subsets, values = enumerate_objectives(problem, max_subsets)
    best = int(np.argmin(values))
    return make_solution(problem, subsets[best], "brute_force", n_evaluations=len(values))
