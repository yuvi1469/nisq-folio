"""Tests for nisq_folio.problem."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.problem import PortfolioProblem, Solution, make_solution


@given(problems(), st.data())
def test_objective_equals_equal_weight_mean_variance(
    problem: PortfolioProblem, data: st.DataObject
) -> None:
    """f(x) = lambda w^T Sigma w - mu^T w with w = x / K, for any feasible x."""
    chosen = data.draw(
        st.sets(st.integers(0, problem.n - 1), min_size=problem.k, max_size=problem.k)
    )
    x = np.zeros(problem.n, dtype=int)
    x[list(chosen)] = 1
    w = x / problem.k
    expected = problem.risk_aversion * w @ problem.sigma @ w - problem.mu @ w
    assert problem.objective(x) == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert problem.is_feasible(x)


@given(problems(max_n=7))
def test_batch_objective_matches_single_objective(problem: PortfolioProblem) -> None:
    subsets = np.array(list(combinations(range(problem.n), problem.k)))
    batch = problem.objective_of_subsets(subsets)
    for row, value in zip(subsets, batch, strict=True):
        x = np.zeros(problem.n, dtype=int)
        x[row] = 1
        assert value == pytest.approx(problem.objective(x), rel=1e-12, abs=1e-14)


def test_from_moments_aligns_sigma_to_mu_order() -> None:
    base = random_problem(4, 2, 1.0, seed=3)
    names = list(base.asset_names)
    mu = pd.Series(base.mu, index=names)
    sigma = pd.DataFrame(base.sigma, index=names, columns=names)
    shuffled = sigma.loc[names[::-1], names[::-1]]  # different label order on purpose
    rebuilt = PortfolioProblem.from_moments(mu, shuffled, 2, 1.0)
    np.testing.assert_array_equal(rebuilt.sigma, base.sigma)
    assert rebuilt.asset_names == base.asset_names


def test_arrays_are_read_only_copies() -> None:
    mu = np.array([0.1, 0.2, 0.3])
    problem = PortfolioProblem(mu, np.eye(3), 1, 1.0, ("a", "b", "c"))
    mu[0] = 99.0
    assert problem.mu[0] == 0.1
    with pytest.raises(ValueError):
        problem.mu[0] = 1.0


def bad(**changes: object) -> dict:
    kwargs = {
        "mu": np.zeros(3),
        "sigma": np.eye(3),
        "k": 1,
        "risk_aversion": 1.0,
        "asset_names": ("a", "b", "c"),
    }
    return {**kwargs, **changes}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (bad(mu=np.zeros(1)), "at least two"),
        (bad(sigma=np.eye(2)), "shape"),
        (bad(sigma=np.array([[1.0, 0.5, 0], [0, 1, 0], [0, 0, 1]])), "symmetric"),
        (bad(mu=np.array([0.0, np.nan, 0.0])), "finite"),
        (bad(k=0), "k must be"),
        (bad(k=3), "k must be"),
        (bad(k=True), "k must be"),
        (bad(risk_aversion=-1.0), "risk_aversion"),
        (bad(asset_names=("a", "a", "c")), "unique"),
    ],
)
def test_invalid_problems_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PortfolioProblem(**kwargs)


@pytest.mark.parametrize(
    ("x", "message"), [(np.array([1, 0]), "shape"), (np.array([1, 2, 0]), "0/1")]
)
def test_objective_rejects_non_binary_or_wrong_length(x: np.ndarray, message: str) -> None:
    problem = PortfolioProblem(**bad())
    with pytest.raises(ValueError, match=message):
        problem.objective(x)


def test_infeasible_vector_is_evaluated_but_flagged() -> None:
    problem = PortfolioProblem(**bad())
    x = np.array([1, 1, 0])
    assert not problem.is_feasible(x)
    assert np.isfinite(problem.objective(x))


def test_make_solution_sorts_recomputes_and_round_trips() -> None:
    problem = random_problem(5, 2, 2.0, seed=1)
    sol = make_solution(problem, np.array([4, 1]), "test", n_evaluations=7)
    assert sol.selection == (1, 4)
    x = sol.to_vector(problem.n)
    np.testing.assert_array_equal(x, [0, 1, 0, 0, 1])
    assert sol.objective == problem.objective(x)
    assert sol == Solution((1, 4), sol.objective, "test", 7)


@pytest.mark.parametrize(
    ("selected", "message"),
    [([0], "not a set of K"), ([1, 1], "not a set of K"), ([0, 9], "out-of-range")],
)
def test_make_solution_rejects_infeasible_selections(selected: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_solution(random_problem(5, 2, 1.0, seed=0), np.array(selected), "test", 0)
