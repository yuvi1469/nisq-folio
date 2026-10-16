"""Out-of-sample evaluation of covariance estimators for portfolio selection (ADR-022).

Every estimator is fitted on the 30-asset universe's in-sample returns (2015-2020), and each
instance uses its block. For each instance and estimator, the exact (brute-force) optimal
selection x_hat is found, with w = x_hat / K, and scored on out-of-sample data (2021-2023)
that no estimator saw:

    realized variance   v_oos   = w^T Sigma_oos w
    predicted variance  v_pred  = w^T Sigma_est w
    risk ratio                  = v_oos / v_pred      (> 1: risk was underestimated)
    realized objective  f_oos   = lambda * v_oos - mu_oos^T w

Estimators:
    ``sample``         sample covariance (the "none" level of the denoising factor)
    ``rmt_clip``       Marchenko-Pastur clipping, keep lambda > lambda_+ (the "RMT" level)
    ``rmt_band_only``  average only eigenvalues inside the MP band (reference)
    ``ledoit_wolf``    Ledoit-Wolf linear shrinkage, scikit-learn (non-RMT reference)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.covariance import LedoitWolf

from nisq_folio.baselines.brute_force import brute_force
from nisq_folio.data.returns import TRADING_DAYS_PER_YEAR, annualized_moments
from nisq_folio.denoise.rmt import denoised_covariance
from nisq_folio.experiments.instances import build_problem

ESTIMATORS = ("sample", "rmt_clip", "rmt_band_only", "ledoit_wolf")


def universe_covariances(
    in_sample: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit every estimator on the universe's in-sample returns (annualized).

    Args:
        in_sample: Daily in-sample returns of the whole universe.

    Returns:
        ``(covariances, reports)``: estimator -> covariance, and JSON-serializable fit details.
    """
    _, sample = annualized_moments(in_sample)
    clip, clip_report = denoised_covariance(in_sample, rule="clip")
    band, band_report = denoised_covariance(in_sample, rule="band_only")
    lw = LedoitWolf().fit(in_sample.to_numpy())
    ledoit = pd.DataFrame(
        TRADING_DAYS_PER_YEAR * lw.covariance_, index=in_sample.columns, columns=in_sample.columns
    )
    covariances = {"sample": sample, "rmt_clip": clip, "rmt_band_only": band, "ledoit_wolf": ledoit}
    reports = {
        "rmt_clip": clip_report.__dict__,
        "rmt_band_only": band_report.__dict__,
        "ledoit_wolf": {"shrinkage": float(lw.shrinkage_)},
    }
    return covariances, reports


def correlation_distance(sigma: pd.DataFrame, target: pd.DataFrame) -> float:
    """Frobenius distance between the correlation matrices implied by two covariances."""

    def corr(m: pd.DataFrame) -> np.ndarray:
        s = 1.0 / np.sqrt(np.diag(m.to_numpy()))
        return m.to_numpy() * np.outer(s, s)

    return float(np.linalg.norm(corr(sigma) - corr(target.loc[sigma.index, sigma.columns])))


def evaluate_instance(
    in_sample: pd.DataFrame,
    out_of_sample: pd.DataFrame,
    tickers: tuple[str, ...],
    k: int,
    risk_aversion: float,
    covariances: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Score every estimator's optimal selection for one instance out of sample (module doc).

    Args:
        in_sample: Daily in-sample returns (universe).
        out_of_sample: Daily out-of-sample returns (universe).
        tickers: The instance's assets.
        k: Assets to select.
        risk_aversion: lambda.
        covariances: Estimator -> universe-level annualized covariance.

    Returns:
        Estimator -> {selection, predicted_variance, realized_variance, risk_ratio,
        realized_objective}.
    """
    mu_oos, sigma_oos = annualized_moments(out_of_sample.loc[:, list(tickers)])
    out: dict[str, dict[str, Any]] = {}
    for name, cov in covariances.items():
        problem = build_problem(in_sample, tickers, k, risk_aversion, covariance=cov)
        selection = brute_force(problem).selection
        w = np.zeros(problem.n)
        w[list(selection)] = 1.0 / k
        predicted = float(w @ problem.sigma @ w)
        realized = float(w @ sigma_oos.to_numpy() @ w)
        out[name] = {
            "selection": [tickers[i] for i in selection],
            "predicted_variance": predicted,
            "realized_variance": realized,
            "risk_ratio": realized / predicted,
            "realized_objective": risk_aversion * realized - float(mu_oos.to_numpy() @ w),
        }
    return out


def paired_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Summarize candidate - reference over paired instances (lower is better).

    Returns the median difference, the share of instances where the candidate is lower, the
    share with an identical value (same selection), and a two-sided Wilcoxon signed-rank
    p-value over the non-zero differences (``nan`` if every difference is zero). Instances share
    assets, so they are not independent; treat the p-value as descriptive.
    """
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    diff = candidate - reference
    nonzero = diff[diff != 0.0]
    p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size else float("nan")
    return {
        "median_difference": float(np.median(diff)),
        "share_lower": float(np.mean(diff < 0)),
        "share_identical": float(np.mean(diff == 0)),
        "wilcoxon_p": p_value,
    }
