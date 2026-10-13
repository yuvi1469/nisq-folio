"""Tests for seeding, instance sampling, the in/out-of-sample split, calibration, and records."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.baselines.brute_force import all_subsets
from nisq_folio.calibration import balanced_risk_aversion
from nisq_folio.config import load_config
from nisq_folio.data.prices import PriceMetadata
from nisq_folio.data.returns import annualized_moments, split_returns
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.experiments.records import write_run_record
from nisq_folio.problem import PortfolioProblem
from nisq_folio.provenance import GitState
from nisq_folio.seeding import child_rng

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = tuple(f"T{i:02d}" for i in range(30))

# ---------------------------------------------------------------- seeding


@given(st.integers(0, 2**32 - 1), st.lists(st.integers(0, 1000), max_size=4))
def test_same_keys_give_same_stream(root: int, keys: list[int]) -> None:
    a = child_rng(root, *keys).random(5)
    b = child_rng(root, *keys).random(5)
    np.testing.assert_array_equal(a, b)


def test_different_keys_give_different_streams() -> None:
    draws = {tuple(child_rng(7, *keys).random(3)) for keys in [(0,), (1,), (0, 1), (1, 0)]}
    assert len(draws) == 4


def test_negative_seed_or_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        child_rng(-1)
    with pytest.raises(ValueError):
        child_rng(1, -2)


# ---------------------------------------------------------------- instances


@given(st.integers(2, 30), st.integers(1, 20), st.integers(0, 2**32 - 1))
def test_instances_are_distinct_ordered_and_reproducible(n: int, count: int, seed: int) -> None:
    from math import comb

    count = min(count, comb(30, n))
    first = sample_instance_tickers(UNIVERSE, n, count, seed)
    assert first == sample_instance_tickers(UNIVERSE, n, count, seed)
    assert len(first) == count == len(set(first))
    for instance in first:
        assert len(instance) == n
        assert list(instance) == sorted(instance, key=UNIVERSE.index)  # universe order kept


def test_asking_for_more_instances_does_not_change_earlier_ones() -> None:
    assert sample_instance_tickers(UNIVERSE, 8, 10, 1)[:5] == sample_instance_tickers(
        UNIVERSE, 8, 5, 1
    )


@pytest.mark.parametrize(("n", "count"), [(1, 1), (31, 1), (29, 31)])
def test_impossible_instance_requests_are_rejected(n: int, count: int) -> None:
    with pytest.raises(ValueError):
        sample_instance_tickers(UNIVERSE, n, count, 0)


def test_build_problem_uses_the_instance_columns(make_prices) -> None:  # noqa: ANN001
    prices = make_prices(["A", "B", "C", "D"], date(2020, 1, 1), 60, seed=2)
    returns = prices.pct_change().iloc[1:]
    problem = build_problem(returns, ("B", "D"), 1, 3.0)
    mu, sigma = annualized_moments(returns[["B", "D"]])
    assert problem.asset_names == ("B", "D")
    np.testing.assert_allclose(problem.mu, mu.to_numpy())
    np.testing.assert_allclose(problem.sigma, sigma.to_numpy())
    assert problem.risk_aversion == 3.0


# ---------------------------------------------------------------- split


@given(st.integers(2, 60), st.integers(1, 59))
def test_split_is_a_chronological_partition(n_days: int, cut: int) -> None:
    index = pd.bdate_range("2020-01-01", periods=n_days)
    returns = pd.DataFrame({"A": np.arange(n_days, dtype=float)}, index=index)
    cut = min(cut, n_days - 1)
    end = index[cut - 1].date()
    ins, oos = split_returns(returns, end)
    assert len(ins) + len(oos) == n_days
    assert ins.index.max() <= pd.Timestamp(end) < oos.index.min()


def test_split_rejects_empty_windows() -> None:
    returns = pd.DataFrame({"A": [0.1, 0.2]}, index=pd.bdate_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="empty"):
        split_returns(returns, date(2019, 1, 1))
    with pytest.raises(ValueError, match="empty"):
        split_returns(returns, date(2021, 1, 1))


# ---------------------------------------------------------------- calibration


@given(problems(min_n=4, max_n=8))
def test_balanced_lambda_equalizes_spread_of_both_terms(problem: PortfolioProblem) -> None:
    lam = balanced_risk_aversion(problem)
    subsets = all_subsets(problem.n, problem.k)
    k = problem.k
    returns = problem.mu[subsets].sum(axis=1) / k
    variances = problem.sigma[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2)) / k**2
    if np.std(variances) > 0:
        assert lam * np.std(variances) == pytest.approx(np.std(returns), rel=1e-9)


@given(problems(min_n=4, max_n=8), st.floats(0.1, 10.0))
def test_balanced_lambda_scales_with_returns(problem: PortfolioProblem, c: float) -> None:
    """Multiplying mu by c multiplies lambda_bal by c (the trade-off is unit-consistent)."""
    scaled = PortfolioProblem(problem.mu * c, problem.sigma, problem.k, 0.0, problem.asset_names)
    assert balanced_risk_aversion(scaled) == pytest.approx(
        c * balanced_risk_aversion(problem), rel=1e-9
    )


def test_balanced_lambda_is_infinite_when_variance_never_changes() -> None:
    problem = PortfolioProblem(np.array([0.1, 0.2, 0.3]), np.zeros((3, 3)), 1, 0.0, tuple("abc"))
    assert balanced_risk_aversion(problem) == float("inf")


def test_balanced_lambda_on_known_instance() -> None:
    problem = random_problem(6, 3, 0.0, seed=4)
    assert np.isfinite(balanced_risk_aversion(problem))


# ---------------------------------------------------------------- run records


def test_run_record_contains_config_git_data_and_results(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    text = (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
    text = text.replace("../results", str((tmp_path / "out").as_posix()))
    (cfg_dir / "c.yaml").write_text(text, encoding="utf-8")
    config = load_config(cfg_dir / "c.yaml")
    meta = PriceMetadata(
        ["A"], "2020-01-01", "2020-12-31", "2020-01-02", "2020-12-31", 1, "src", "v",
        "2026-01-01T00:00:00+00:00", "abc", False, "f" * 64,
    )  # fmt: skip
    git = GitState(commit="1234567890abcdef", dirty=True)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    path = write_run_record(config, "demo", git, meta, {"answer": 42}, now=now)

    assert path.name == "20260102T030405Z_1234567-dirty.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["git"] == {"commit": "1234567890abcdef", "dirty": True}
    assert record["config_text"] == text
    assert record["data"]["sha256"] == "f" * 64
    assert record["results"] == {"answer": 42}
