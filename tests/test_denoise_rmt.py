"""Tests for RMT cleaning (ADR-022), the estimator evaluation, its config section and CLI."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from nisq_folio.config import DENOISE_METHODS, ConfigError, load_config
from nisq_folio.data.returns import annualized_moments
from nisq_folio.denoise.__main__ import main
from nisq_folio.denoise.evaluation import (
    ESTIMATORS,
    correlation_distance,
    evaluate_instance,
    paired_comparison,
    universe_covariances,
)
from nisq_folio.denoise.rmt import clip_correlation, denoised_covariance, marchenko_pastur_edges
from nisq_folio.experiments.instances import build_problem

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))


def returns_frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2015-01-01", periods=values.shape[0])
    return pd.DataFrame(values, index=index, columns=[f"A{i:02d}" for i in range(values.shape[1])])


def one_factor(n: int, t: int, seed: int, flat_bulk: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns from r = b f + e and the exact population correlation of that model.

    With ``flat_bulk`` every asset has the same loading and idiosyncratic volatility, so all
    non-market population eigenvalues are equal: the regime in which clipping is exact.
    """
    rng = np.random.default_rng(seed)
    b = np.ones(n) if flat_bulk else rng.uniform(0.5, 1.5, n)
    idio = np.ones(n) if flat_bulk else rng.uniform(0.8, 1.2, n)
    r = np.outer(rng.normal(size=t), b) + rng.normal(size=(t, n)) * idio
    cov = np.outer(b, b) + np.diag(idio**2)
    s = 1 / np.sqrt(np.diag(cov))
    return r, cov * np.outer(s, s)


# ---------------------------------------------------------------- Marchenko-Pastur cleaning


def test_marchenko_pastur_edges() -> None:
    lo, hi = marchenko_pastur_edges(0.25)
    assert (lo, hi) == pytest.approx((0.25, 2.25))
    assert marchenko_pastur_edges(0.25, sigma2=2.0) == pytest.approx((0.5, 4.5))
    for q, s2 in ((0.0, 1.0), (1.5, 1.0), (0.5, 0.0)):
        with pytest.raises(ValueError):
            marchenko_pastur_edges(q, s2)


@settings(max_examples=30)
@given(st.integers(2, 25), st.integers(0, 10_000), st.sampled_from(["clip", "band_only"]))
def test_cleaned_correlation_is_a_valid_correlation_matrix(n: int, seed: int, rule: str) -> None:
    t = n * 4
    corr = np.corrcoef(np.random.default_rng(seed).normal(size=(t, n)), rowvar=False)
    clean, report = clip_correlation(corr, t, rule)
    np.testing.assert_allclose(clean, clean.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(clean), 1.0)
    assert np.linalg.eigvalsh(clean).min() > -1e-10
    assert report.n_kept + report.n_replaced == n
    assert list(report.eigenvalues) == sorted(report.eigenvalues, reverse=True)


def test_nothing_changes_when_every_eigenvalue_is_signal() -> None:
    corr = np.array([[1.0, 0.9], [0.9, 1.0]])  # eigenvalues 1.9 and 0.1, band ~[0.91, 1.09]
    clean, report = clip_correlation(corr, 1000, "band_only")
    np.testing.assert_allclose(clean, corr, atol=1e-12)
    assert report.n_replaced == 0


def test_pure_noise_is_cleaned_towards_the_identity() -> None:
    """For uncorrelated returns the population correlation is I; clipping must get closer."""
    n, t = 30, 600
    corr = np.corrcoef(np.random.default_rng(11).normal(size=(t, n)), rowvar=False)
    clean, report = clip_correlation(corr, t)
    assert np.linalg.norm(clean - np.eye(n)) < 0.5 * np.linalg.norm(corr - np.eye(n))
    assert report.n_kept <= 1


@pytest.mark.parametrize("seed", range(5))
def test_clipping_helps_when_the_population_bulk_is_flat(seed: int) -> None:
    """In its own regime (one factor, equal non-market eigenvalues) clipping beats the sample."""
    returns, population = one_factor(30, 200, seed, flat_bulk=True)
    sample = np.corrcoef(returns, rowvar=False)
    clean, report = clip_correlation(sample, 200)
    assert report.n_kept >= 1  # the market factor is kept
    # Strictly better; measured error ratios on seeds 0-9 were 0.31-0.85 (2026-10-16).
    assert np.linalg.norm(clean - population) < np.linalg.norm(sample - population)


def test_clipping_can_hurt_when_true_structure_lies_below_the_band() -> None:
    """Documented counter-example (ADR-022): clipping can be worse than the sample.

    With heterogeneous loadings the population has genuine eigenvalues below lambda_-, which
    textbook clipping flattens. Measured on seeds 0-9 (2026-10-16): clipping was worse than the
    sample on 9 of 10, band-only better on 8 of 10. This pins that behaviour so a future change
    to the method cannot silently alter it.
    """
    worse = 0
    for seed in range(10):
        returns, population = one_factor(30, 200, seed)
        sample = np.corrcoef(returns, rowvar=False)
        clean, _ = clip_correlation(sample, 200)
        worse += np.linalg.norm(clean - population) > np.linalg.norm(sample - population)
    assert worse == 9


def test_invalid_correlations_are_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        clip_correlation(np.array([[1.0, 0.2], [0.3, 1.0]]), 10)
    with pytest.raises(ValueError, match="unit diagonal"):
        clip_correlation(np.array([[2.0, 0.0], [0.0, 1.0]]), 10)
    with pytest.raises(ValueError, match="T >= N"):
        clip_correlation(np.eye(3), 2)
    with pytest.raises(ValueError, match="rule"):
        clip_correlation(np.eye(2), 10, "shrink")


def test_denoised_covariance_keeps_each_assets_variance() -> None:
    returns = returns_frame(one_factor(10, 300, 3)[0] * 0.01)
    sigma, _ = denoised_covariance(returns)
    _, sample = annualized_moments(returns)
    np.testing.assert_allclose(np.diag(sigma), np.diag(sample), rtol=1e-12)
    assert list(sigma.index) == list(returns.columns) == list(sigma.columns)
    assert np.linalg.eigvalsh(sigma.to_numpy()).min() > -1e-12


def test_denoised_covariance_rejects_bad_returns() -> None:
    bad = returns_frame(np.ones((20, 3)))
    with pytest.raises(ValueError, match="non-zero"):
        denoised_covariance(bad)
    nan = returns_frame(np.random.default_rng(0).normal(size=(20, 3)))
    nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        denoised_covariance(nan)


def test_build_problem_uses_the_external_covariance_block() -> None:
    returns = returns_frame(np.random.default_rng(2).normal(size=(100, 5)) * 0.01)
    sigma, _ = denoised_covariance(returns)
    problem = build_problem(returns, ("A01", "A03"), 1, 2.0, covariance=sigma)
    np.testing.assert_allclose(problem.sigma, sigma.loc[["A01", "A03"], ["A01", "A03"]])
    mu, _ = annualized_moments(returns[["A01", "A03"]])
    np.testing.assert_allclose(problem.mu, mu)  # mu is never denoised


# ---------------------------------------------------------------- evaluation


def test_universe_covariances_are_all_valid() -> None:
    returns = returns_frame(one_factor(12, 400, 5)[0] * 0.01)
    covariances, reports = universe_covariances(returns)
    assert set(covariances) == set(ESTIMATORS)
    for cov in covariances.values():
        assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-12
    np.testing.assert_allclose(np.diag(covariances["rmt_clip"]), np.diag(covariances["sample"]))
    assert 0.0 <= reports["ledoit_wolf"]["shrinkage"] <= 1.0
    assert correlation_distance(covariances["sample"], covariances["sample"]) == pytest.approx(0)


def test_evaluate_instance_scores_each_estimator() -> None:
    values = one_factor(8, 500, 6)[0] * 0.01
    returns = returns_frame(values)
    ins, oos = returns.iloc[:300], returns.iloc[300:]
    covariances, _ = universe_covariances(ins)
    tickers = tuple(returns.columns[:6])
    out = evaluate_instance(ins, oos, tickers, 3, 4.5, covariances)
    assert set(out) == set(ESTIMATORS)
    for r in out.values():
        assert len(r["selection"]) == 3 and set(r["selection"]) <= set(tickers)
        assert r["risk_ratio"] == pytest.approx(r["realized_variance"] / r["predicted_variance"])


def test_paired_comparison() -> None:
    out = paired_comparison(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 4.0, 5.0]))
    assert out["median_difference"] == pytest.approx(-1.0)
    assert out["share_lower"] == 0.75
    assert out["share_identical"] == 0.25
    assert 0.0 <= out["wilcoxon_p"] <= 1.0
    assert np.isnan(paired_comparison(np.ones(3), np.ones(3))["wilcoxon_p"])


# ---------------------------------------------------------------- config + CLI


def test_config_denoise_section() -> None:
    assert set(DENOISE_METHODS) <= set(ESTIMATORS)
    config = load_config(REPO_ROOT / "configs" / "base.yaml").require_denoise()
    assert config.experiment_method == "rmt_clip"
    assert config.evaluation_instances_per_size == 50


@pytest.mark.parametrize(
    ("section", "message"),
    [({"experiment_method": "pca", "evaluation_instances_per_size": 5}, "experiment_method"),
     ({"experiment_method": "rmt_clip", "evaluation_instances_per_size": 0}, ">= 1"),
     (None, "'denoise' section")],
)  # fmt: skip
def test_invalid_denoise_sections(tmp_path: Path, section: dict | None, message: str) -> None:
    content = {k: v for k, v in BASE.items() if k != "denoise"}
    if section is not None:
        content["denoise"] = section
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path).require_denoise()


def test_cli_writes_record(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    tickers = [f"T{i}" for i in range(8)]
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 1,
        "universe": {"start_date": "2020-01-01", "end_date": "2020-12-31", "n_stocks": 8,
                     "n_sectors": 1, "sectors": {"S": tickers}},
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4], "risk_aversion": 4.5},
        "denoise": {"experiment_method": "rmt_clip", "evaluation_instances_per_size": 4},
    }  # fmt: skip
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=8)

    assert main(["--config", str(path)], fetcher=fetcher) == 0
    assert "MP band" in capsys.readouterr().out
    record = json.loads(
        next((tmp_path / "results" / "denoise_evaluation").glob("*.json")).read_text()
    )
    size4 = record["results"]["sizes"]["4"]
    assert len(size4["instances"]) == 4
    assert set(size4["comparisons"]) == set(ESTIMATORS[1:])
