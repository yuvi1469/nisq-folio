"""Tests for QAOA metrics (ADR-019), the shot-based optimizer, and the tuning rule (ADR-020)."""

import json
import math
from math import comb

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.problem import PortfolioProblem
from nisq_folio.qaoa.metrics import SolutionLandscape, distribution_metrics, samples_to_probability
from nisq_folio.qaoa.optimize import QaoaSettings, ramp_angles, run_noiseless_qaoa
from nisq_folio.qaoa.tuning import (
    SCORE_TOLERANCE,
    TuningConfiguration,
    configuration_score,
    select_configuration,
)
from nisq_folio.qubo.penalty import sufficient_penalty

# ---------------------------------------------------------------- metrics


@given(problems(min_n=3, max_n=7))
def test_uniform_distribution_has_amplification_one_vs_uniform(problem: PortfolioProblem) -> None:
    land = SolutionLandscape.from_problem(problem)
    size = 2**problem.n
    m = distribution_metrics(np.full(size, 1 / size), land)
    assert m.amplification_uniform == pytest.approx(1.0)
    assert m.amplification_feasible == pytest.approx(comb(problem.n, problem.k) / size)
    assert m.p_feasible == pytest.approx(comb(problem.n, problem.k) / size)


@given(problems(min_n=3, max_n=7))
def test_point_mass_on_optimum_is_perfect(problem: PortfolioProblem) -> None:
    land = SolutionLandscape.from_problem(problem)
    probs = np.zeros(2**problem.n)
    probs[np.flatnonzero(land.optimal)[0]] = 1.0
    m = distribution_metrics(probs, land)
    assert m.p_optimal == 1.0
    assert m.expected_gap == 0.0
    assert m.samples_to_90 == 1.0
    assert m.amplification_feasible == pytest.approx(
        comb(problem.n, problem.k) / land.optimal.sum()
    )


@given(problems(min_n=3, max_n=7))
def test_gaps_are_normalized_and_infeasible_is_worst(problem: PortfolioProblem) -> None:
    land = SolutionLandscape.from_problem(problem)
    gaps = land.gaps()
    assert gaps.min() >= 0.0 and gaps.max() <= 1.0
    assert (gaps[~land.feasible] == 1.0).all()
    assert (gaps[land.optimal] == 0.0).all()
    assert land.optimal.sum() >= 1
    assert land.f_star <= land.f_worst


def test_flat_problem_counts_every_feasible_selection_as_optimal() -> None:
    problem = PortfolioProblem(np.zeros(4), np.zeros((4, 4)), 2, 1.0, tuple("abcd"))
    land = SolutionLandscape.from_problem(problem)
    assert land.optimal.sum() == comb(4, 2)
    probs = np.where(land.feasible, 1.0 / comb(4, 2), 0.0)
    m = distribution_metrics(probs, land)
    assert m.amplification_feasible == pytest.approx(1.0)
    assert m.expected_gap == 0.0


@pytest.mark.parametrize(
    ("p", "target", "expected"),
    [(1.0, 0.9, 1.0), (0.0, 0.9, math.inf), (0.5, 0.9, 4.0), (0.1, 0.9, 22.0)],
)
def test_samples_to_probability(p: float, target: float, expected: float) -> None:
    assert samples_to_probability(p, target) == expected


@given(st.floats(1e-6, 0.999), st.floats(0.01, 0.99))
def test_samples_to_probability_reaches_target(p: float, target: float) -> None:
    k = samples_to_probability(p, target)
    assert 1 - (1 - p) ** k >= target - 1e-12
    if k > 1:
        assert 1 - (1 - p) ** (k - 1) < target + 1e-12


def test_metric_inputs_are_validated() -> None:
    land = SolutionLandscape.from_problem(random_problem(3, 1, 1.0, 0))
    with pytest.raises(ValueError, match="lengths"):
        distribution_metrics(np.ones(4) / 4, land)
    with pytest.raises(ValueError, match="sum to 1"):
        distribution_metrics(np.ones(8), land)
    with pytest.raises(ValueError, match="target"):
        samples_to_probability(0.5, 1.0)


# ---------------------------------------------------------------- optimizer


def test_ramp_angles() -> None:
    gammas, betas = ramp_angles(4, 1.0)
    np.testing.assert_allclose(gammas, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(betas, [0.875, 0.625, 0.375, 0.125])


def test_run_is_reproducible_accounts_for_shots_and_is_serializable() -> None:
    problem = random_problem(6, 3, 4.5, seed=2)
    settings = QaoaSettings(layers=2, shots=64, max_evaluations=30)
    penalty = sufficient_penalty(problem)
    first = run_noiseless_qaoa(problem, penalty, settings, np.random.default_rng(5))
    second = run_noiseless_qaoa(problem, penalty, settings, np.random.default_rng(5))
    assert first.as_dict() == second.as_dict()
    assert 1 <= first.n_evaluations <= 30
    assert first.total_shots == first.n_evaluations * 64
    assert len(first.energy_trace) == first.n_evaluations
    json.dumps(first.as_dict())
    land = SolutionLandscape.from_problem(problem)
    if first.best_sampled is not None:
        assert first.best_sampled >= land.f_star - 1e-12
    assert 0.0 <= first.metrics.p_optimal <= 1.0


def test_different_shot_seeds_give_different_runs() -> None:
    problem = random_problem(6, 3, 4.5, seed=2)
    settings = QaoaSettings(layers=1, shots=32, max_evaluations=20)
    penalty = sufficient_penalty(problem)
    a = run_noiseless_qaoa(problem, penalty, settings, np.random.default_rng(1))
    b = run_noiseless_qaoa(problem, penalty, settings, np.random.default_rng(2))
    assert a.energy_trace != b.energy_trace


class AllZerosSampler:
    """Fake RNG whose every shot lands on bitstring 0 (no asset selected: infeasible)."""

    def multinomial(self, shots: int, probs: np.ndarray) -> np.ndarray:
        """Put all ``shots`` on index 0."""
        counts = np.zeros(len(probs), dtype=np.int64)
        counts[0] = shots
        return counts


def test_best_sampled_is_none_when_nothing_feasible_is_seen() -> None:
    problem = random_problem(4, 2, 4.5, seed=0)
    settings = QaoaSettings(layers=1, shots=4, max_evaluations=4)
    run = run_noiseless_qaoa(problem, 1.0, settings, AllZerosSampler())  # type: ignore[arg-type]
    assert run.best_sampled is None


def test_inconsistent_landscapes_are_rejected() -> None:
    land = SolutionLandscape.from_problem(random_problem(4, 2, 4.5, seed=0))
    none_feasible = np.zeros_like(land.feasible)
    with pytest.raises(ValueError, match="no feasible"):
        SolutionLandscape(land.objective, none_feasible, land.optimal, land.f_star, land.f_worst)
    infeasible_optimum = ~land.feasible
    with pytest.raises(ValueError, match="subset"):
        SolutionLandscape(
            land.objective, land.feasible, infeasible_optimum, land.f_star, land.f_worst
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"layers": 0}, {"layers": 1, "shots": 0}, {"layers": 1, "max_evaluations": 0},
     {"layers": 1, "rhobeg": 0.0}, {"layers": 1, "ramp_delta": -1.0}],
)  # fmt: skip
def test_invalid_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        QaoaSettings(**kwargs)


# ---------------------------------------------------------------- tuning rule


def test_score_is_mean_log_median() -> None:
    score = configuration_score({8: [1.0, 2.0, 4.0], 10: [8.0]})
    assert score == pytest.approx((math.log(2.0) + math.log(8.0)) / 2)
    assert np.isfinite(configuration_score({8: [0.0]}))  # floored, not -inf
    with pytest.raises(ValueError):
        configuration_score({})
    with pytest.raises(ValueError):
        configuration_score({8: []})


def test_selection_prefers_smaller_depth_within_tolerance() -> None:
    a = TuningConfiguration(1, 0.5, 0.75)
    b = TuningConfiguration(3, 0.5, 0.75)
    assert select_configuration({a: 1.0, b: 1.0 + SCORE_TOLERANCE / 2}) == a
    assert select_configuration({a: 1.0, b: 1.0 + 2 * SCORE_TOLERANCE}) == b


def test_selection_tie_breaks_on_penalty_then_delta() -> None:
    low_c = TuningConfiguration(2, 0.1, 0.75)
    high_c = TuningConfiguration(2, 0.5, 1.5)
    high_c_small_delta = TuningConfiguration(2, 0.5, 0.75)
    scores = {low_c: 2.0, high_c: 2.0, high_c_small_delta: 1.95}
    assert select_configuration(scores) == high_c_small_delta
    with pytest.raises(ValueError):
        select_configuration({})


@given(st.dictionaries(
    st.builds(TuningConfiguration, st.integers(1, 4), st.sampled_from([0.1, 0.5, 1.0]),
              st.sampled_from([0.75, 1.5])),
    st.floats(-5, 5), min_size=1,
))  # fmt: skip
def test_selected_configuration_is_always_within_tolerance(scores: dict) -> None:
    chosen = select_configuration(scores)
    best = max(scores.values())
    assert scores[chosen] >= best - SCORE_TOLERANCE
    eligible = [c for c, s in scores.items() if s >= best - SCORE_TOLERANCE]
    assert chosen.layers == min(c.layers for c in eligible)
    assert TuningConfiguration(2, 0.25, 1.5).label() == "p=2 c=0.25 delta=1.5"


@pytest.mark.parametrize(("layers", "budget"), [(1, 3), (2, 5), (4, 9)])
def test_budget_below_cobyla_minimum_is_rejected(layers: int, budget: int) -> None:
    with pytest.raises(ValueError, match=r"2 \* layers \+ 2"):
        QaoaSettings(layers=layers, max_evaluations=budget)
    QaoaSettings(layers=layers, max_evaluations=budget + 1)  # the minimum itself is fine
