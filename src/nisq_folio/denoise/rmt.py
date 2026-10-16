"""Random-matrix-theory cleaning of correlation and covariance matrices (DECISIONS.md ADR-022).

For T observations of N standardized, uncorrelated returns, the eigenvalues of the sample
correlation matrix follow the Marchenko-Pastur law in the limit N, T -> infinity with q = N / T
fixed, supported on

    [lambda_-, lambda_+] = sigma^2 * [(1 - sqrt(q))^2, (1 + sqrt(q))^2].

Eigenvalue clipping (Laloux, Cizeau, Bouchaud & Potters, Phys. Rev. Lett. 83, 1467, 1999;
reviewed in Bun, Bouchaud & Potters, Physics Reports 666, 2017 -- verify both before citing):
with C = V diag(lambda) V^T, keep the eigenvalues regarded as signal, replace every other
eigenvalue by the average of those replaced (so the trace, i.e. total variance, is preserved),
keep all eigenvectors, then rescale to unit diagonal:

    C~ = V diag(lambda~) V^T,     C_clean = D^(-1/2) C~ D^(-1/2),   D = diag(C~)

Two rules for "signal" are provided:
    ``clip``       keep lambda > lambda_+ only (the textbook rule; main method)
    ``band_only``  keep everything outside [lambda_-, lambda_+]; average only eigenvalues inside

The covariance is rebuilt as Sigma_clean = S C_clean S with S = diag(sample standard deviations),
so only correlations are cleaned and each asset's own variance is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nisq_folio.problem import FloatArray

CLIP_RULES = ("clip", "band_only")


@dataclass(frozen=True)
class CleaningReport:
    """What a cleaning did, for logging.

    Attributes:
        q: N / T.
        lambda_minus: Lower Marchenko-Pastur edge.
        lambda_plus: Upper Marchenko-Pastur edge.
        n_kept: Eigenvalues kept as signal.
        n_replaced: Eigenvalues replaced by their average.
        eigenvalues: Sample eigenvalues, descending.
    """

    q: float
    lambda_minus: float
    lambda_plus: float
    n_kept: int
    n_replaced: int
    eigenvalues: tuple[float, ...]


def marchenko_pastur_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Return (lambda_-, lambda_+) = sigma^2 * (1 -/+ sqrt(q))^2.

    Args:
        q: Aspect ratio N / T, with 0 < q <= 1.
        sigma2: Noise variance (1 for a correlation matrix of pure noise).

    Returns:
        The lower and upper edges of the Marchenko-Pastur spectrum.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    if not sigma2 > 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def clip_correlation(
    correlation: FloatArray, n_observations: int, rule: str = "clip"
) -> tuple[FloatArray, CleaningReport]:
    """Clean a correlation matrix by eigenvalue clipping (module docstring).

    Args:
        correlation: Sample correlation matrix, N x N, symmetric with unit diagonal.
        n_observations: T, the number of observations it was estimated from (T >= N).
        rule: ``"clip"`` or ``"band_only"``.

    Returns:
        ``(clean, report)``: the cleaned correlation (symmetric PSD, unit diagonal) and a report.
    """
    if rule not in CLIP_RULES:
        raise ValueError(f"rule must be one of {CLIP_RULES}, got {rule!r}")
    c = np.asarray(correlation, dtype=np.float64)
    n = c.shape[0]
    if c.shape != (n, n) or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("correlation must be a symmetric square matrix")
    if not np.allclose(np.diag(c), 1.0, atol=1e-8):
        raise ValueError("correlation must have a unit diagonal")
    if n_observations < n:
        raise ValueError(f"need T >= N, got T={n_observations}, N={n}")
    lambda_minus, lambda_plus = marchenko_pastur_edges(n / n_observations)
    values, vectors = np.linalg.eigh(c)
    if rule == "clip":
        keep = values > lambda_plus
    else:
        keep = (values > lambda_plus) | (values < lambda_minus)
    cleaned_values = values.copy()
    if (~keep).any():
        cleaned_values[~keep] = values[~keep].mean()
    rebuilt = (vectors * cleaned_values) @ vectors.T
    scale = 1.0 / np.sqrt(np.diag(rebuilt))
    clean = rebuilt * np.outer(scale, scale)
    clean = (clean + clean.T) / 2.0
    np.fill_diagonal(clean, 1.0)
    report = CleaningReport(
        q=n / n_observations,
        lambda_minus=float(lambda_minus),
        lambda_plus=float(lambda_plus),
        n_kept=int(keep.sum()),
        n_replaced=int((~keep).sum()),
        eigenvalues=tuple(float(v) for v in values[::-1]),
    )
    return clean, report


def denoised_covariance(
    returns: pd.DataFrame, periods_per_year: int = 252, rule: str = "clip"
) -> tuple[pd.DataFrame, CleaningReport]:
    """Annualized covariance with RMT-cleaned correlations: Sigma = A * S C_clean S.

    Args:
        returns: Per-period returns (estimation window only); columns are assets.
        periods_per_year: A, the annualization factor (matching ``annualized_moments``).
        rule: Clipping rule passed to :func:`clip_correlation`.

    Returns:
        ``(sigma, report)``, with ``sigma`` indexed and columned like ``returns.columns``.
    """
    if returns.isna().to_numpy().any():
        raise ValueError("returns contain missing values")
    values = returns.to_numpy(dtype=np.float64)
    std = values.std(axis=0, ddof=1)
    if not (std > 0).all():
        raise ValueError("every asset needs non-zero return variance")
    correlation = np.corrcoef(values, rowvar=False)
    clean, report = clip_correlation(correlation, len(returns), rule)
    sigma = periods_per_year * clean * np.outer(std, std)
    return pd.DataFrame(sigma, index=returns.columns, columns=returns.columns), report
