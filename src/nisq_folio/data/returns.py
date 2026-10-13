"""Returns and annualized mean/covariance estimates from a price panel."""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) returns from adjusted prices.

    Implements, for each asset i and date t (the first date is dropped):

        r_{t,i} = P_{t,i} / P_{t-1,i} - 1

    Simple returns are used (not log returns) because a portfolio's simple return is the
    weighted sum of asset simple returns, R_p = w^T r, which the mean-variance objective
    assumes.

    Args:
        prices: Strictly positive prices; rows are dates (ascending), columns are assets.

    Returns:
        Returns with one fewer row than ``prices``.

    Raises:
        ValueError: If there are fewer than two rows, any missing values, or a non-positive price.
    """
    if len(prices) < 2:
        raise ValueError("need at least two price observations to compute returns")
    if prices.isna().to_numpy().any():
        raise ValueError("prices contain missing values")
    if (prices <= 0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    return (prices / prices.shift(1) - 1.0).iloc[1:]


def annualized_moments(
    returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate the annualized mean vector and sample covariance matrix of returns.

    With T observations r_1..r_T (vectors over assets), sample mean r_bar, and
    A = ``periods_per_year``:

        mu    = A * r_bar,                 r_bar = (1/T) * sum_t r_t
        Sigma = A * S,                     S = 1/(T-1) * sum_t (r_t - r_bar)(r_t - r_bar)^T

    Annualizing by A assumes returns are i.i.d. across periods; this is the standard
    convention, not a claim about the data.

    Args:
        returns: Per-period returns; rows are dates, columns are assets.
        periods_per_year: Number of return periods per year (252 for daily trading data).

    Returns:
        ``(mu, sigma)``: the annualized mean (Series indexed by asset) and covariance
        (DataFrame indexed and columned by asset).

    Raises:
        ValueError: If there are fewer than two rows, any missing values, or
            ``periods_per_year`` is not positive.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if len(returns) < 2:
        raise ValueError("need at least two return observations to estimate a covariance")
    if returns.isna().to_numpy().any():
        raise ValueError("returns contain missing values")
    mu = returns.mean() * periods_per_year
    sigma = returns.cov(ddof=1) * periods_per_year
    return mu, sigma
