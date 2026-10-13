"""Tests for the classical baselines, validated against an independent exhaustive search."""

from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.baselines.brute_force import all_subsets, brute_force, enumerate_objectives
from nisq_folio.baselines.greedy import greedy
from nisq_folio.baselines.markowitz import markowitz_rounding, markowitz_weights
from nisq_folio.baselines.simulated_annealing import (
    AnnealingSettings,
    simulated_annealing,
    swap_delta,
)
from nisq_folio.problem import PortfolioProblem


def reference_optimum(problem: PortfolioProblem) -> float:
    """Independent check: loop over subsets with the scalar objective (no vectorization)."""
    best = np.inf
    for subset in combinations(range(problem.n), problem.k):
        x = np.zeros(problem.n, dtype=int)
        x[list(subset)] = 1
        best = min(best, problem.objective(x))
    return best


def assert_valid_solution(problem: PortfolioProblem, objective: float, selection: tuple) -> None:
    assert len(selection) == problem.k == len(set(selection))
    assert list(selection) == sorted(selection)
    x = np.zeros(problem.n, dtype=int)
    x[list(selection)] = 1
    assert objective == pytest.approx(problem.objective(x), rel=1e-12, abs=1e-14)
    assert objective >= reference_optimum(problem) - 1e-12


# ---------------------------------------------------------------- brute force


@given(problems())
def test_brute_force_matches_independent_exhaustive_search(problem: PortfolioProblem) -> None:
    sol = brute_force(problem)
    assert sol.objective == pytest.approx(reference_optimum(problem), rel=1e-12, abs=1e-14)
    assert sol.n_evaluations == comb(problem.n, problem.k)
    assert_valid_solution(problem, sol.objective, sol.selection)


def test_all_subsets_is_lexicographic_and_complete() -> None:
    subsets = all_subsets(5, 2)
    assert subsets.shape == (10, 2)
    assert [tuple(r) for r in subsets] == list(combinations(range(5), 2))


def test_enumeration_refuses_oversized_problems() -> None:
    with pytest.raises(ValueError, match="exceeds max_subsets"):
        enumerate_objectives(random_problem(10, 5, 1.0, seed=0), max_subsets=100)


def test_enumeration_is_chunk_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    from nisq_folio.baselines import brute_force as module

    problem = random_problem(9, 4, 2.0, seed=5)
    _, whole = enumerate_objectives(problem)
    monkeypatch.setattr(module, "_CHUNK", 7)
    _, chunked = enumerate_objectives(problem)
    np.testing.assert_array_equal(whole, chunked)


# ---------------------------------------------------------------- greedy


def reference_greedy(problem: PortfolioProblem) -> tuple[int, ...]:
    """Independent greedy: add the asset minimizing the full objective (with fixed K)."""
    x = np.zeros(problem.n, dtype=int)
    for _ in range(problem.k):
        candidates = []
        for j in np.flatnonzero(x == 0):
            trial = x.copy()
            trial[j] = 1
            candidates.append((problem.objective(trial), j))
        _, j = min(candidates, key=lambda c: (c[0], c[1]))
        x[j] = 1
    return tuple(int(i) for i in np.flatnonzero(x))


@given(problems())
def test_greedy_matches_reference_and_is_feasible(problem: PortfolioProblem) -> None:
    sol = greedy(problem)
    assert sol.selection == reference_greedy(problem)
    assert_valid_solution(problem, sol.objective, sol.selection)


@given(st.integers(3, 10), st.data())
def test_greedy_with_zero_risk_aversion_picks_top_k_returns(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(1, n - 1))
    base = random_problem(n, k, 0.0, seed=data.draw(st.integers(0, 10_000)))
    sol = greedy(base)
    top_k = tuple(sorted(np.argsort(-base.mu, kind="stable")[:k].tolist()))
    assert sol.selection == top_k
    assert sol.objective == pytest.approx(brute_force(base).objective)


# ---------------------------------------------------------------- simulated annealing


@given(problems(), st.data())
def test_swap_delta_equals_objective_difference(
    problem: PortfolioProblem, data: st.DataObject
) -> None:
    """f(x - e_i + e_j) - f(x) computed in O(1) equals the full recomputation."""
    chosen = data.draw(
        st.sets(st.integers(0, problem.n - 1), min_size=problem.k, max_size=problem.k)
    )
    x = np.zeros(problem.n, dtype=int)
    x[list(chosen)] = 1
    i = data.draw(st.sampled_from(sorted(chosen)))
    j = data.draw(st.sampled_from([a for a in range(problem.n) if a not in chosen]))
    moved = x.copy()
    moved[i], moved[j] = 0, 1
    expected = problem.objective(moved) - problem.objective(x)
    got = swap_delta(problem, problem.sigma @ x.astype(float), i, j)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


@given(problems(), st.integers(0, 2**32 - 1))
def test_annealing_returns_valid_solution(problem: PortfolioProblem, seed: int) -> None:
    sol = simulated_annealing(problem, np.random.default_rng(seed), AnnealingSettings(n_steps=50))
    assert_valid_solution(problem, sol.objective, sol.selection)
    assert sol.n_evaluations == 50 + AnnealingSettings().n_calibration_moves


def test_annealing_is_deterministic_for_a_given_seed() -> None:
    problem = random_problem(12, 6, 4.5, seed=11)
    settings = AnnealingSettings(n_steps=300)
    first = simulated_annealing(problem, np.random.default_rng(42), settings)
    second = simulated_annealing(problem, np.random.default_rng(42), settings)
    assert first == second


@pytest.mark.parametrize("instance_seed", range(10))
def test_annealing_finds_optimum_on_small_instances(instance_seed: int) -> None:
    problem = random_problem(8, 4, 4.5, seed=instance_seed)
    sol = simulated_annealing(problem, np.random.default_rng(instance_seed), AnnealingSettings())
    assert sol.objective == pytest.approx(brute_force(problem).objective, rel=1e-12, abs=1e-14)


def test_annealing_handles_a_flat_objective() -> None:
    """With mu = 0 and Sigma = 0 every move has delta 0; the run must still be valid."""
    problem = PortfolioProblem(np.zeros(5), np.zeros((5, 5)), 2, 1.0, tuple("abcde"))
    sol = simulated_annealing(problem, np.random.default_rng(0), AnnealingSettings(n_steps=20))
    assert sol.objective == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_calibration_moves": 0},
        {"initial_acceptance": 1.0},
        {"initial_acceptance": 0.0},
        {"final_temperature_ratio": 0.0},
        {"final_temperature_ratio": 1.5},
    ],
)
def test_invalid_annealing_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AnnealingSettings(**kwargs)


# ---------------------------------------------------------------- Markowitz + rounding


@given(problems(min_n=3, max_n=10).filter(lambda p: p.risk_aversion > 0))
def test_markowitz_weights_satisfy_kkt_conditions(problem: PortfolioProblem) -> None:
    """The continuous Markowitz solution satisfies the KKT conditions.

    At the optimum of min lambda w^T Sigma w - mu^T w on the simplex, the gradient
    g = 2 lambda Sigma w - mu equals a constant nu on assets held (w_i > 0) and is >= nu on
    assets not held.
    """
    w = markowitz_weights(problem)
    assert w.min() >= 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    g = 2.0 * problem.risk_aversion * problem.sigma @ w - problem.mu
    held = w > 1e-6
    nu = g[held].mean()
    scale = max(1.0, np.abs(g).max())
    assert np.abs(g[held] - nu).max() <= 1e-5 * scale
    assert (g[~held] >= nu - 1e-5 * scale).all()


def test_markowitz_matches_closed_form_interior_solution() -> None:
    """Sigma = s I, all weights interior: w_i = 1/N + (mu_i - mean(mu)) / (2 lambda s)."""
    mu = np.array([0.10, 0.11, 0.12, 0.13])
    s, lam = 0.04, 5.0
    problem = PortfolioProblem(mu, s * np.eye(4), 2, lam, tuple("abcd"))
    expected = 0.25 + (mu - mu.mean()) / (2 * lam * s)
    np.testing.assert_allclose(markowitz_weights(problem), expected, atol=1e-7)
    assert markowitz_rounding(problem).selection == (2, 3)


@given(problems().filter(lambda p: p.risk_aversion > 0))
def test_markowitz_rounding_keeps_k_largest_weights(problem: PortfolioProblem) -> None:
    sol = markowitz_rounding(problem)
    w = markowitz_weights(problem)
    assert_valid_solution(problem, sol.objective, sol.selection)
    kept = w[list(sol.selection)].min()
    dropped = np.delete(w, list(sol.selection))
    assert (dropped <= kept + 1e-12).all()


def test_markowitz_reports_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from nisq_folio.baselines import markowitz as module

    monkeypatch.setattr(
        module, "minimize", lambda *a, **k: SimpleNamespace(success=False, message="boom")
    )
    with pytest.raises(RuntimeError, match="boom"):
        markowitz_weights(random_problem(4, 2, 1.0, seed=0))
