"""Tests for nisq_folio.data.returns, mostly property-based."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nisq_folio.data.returns import annualized_moments, simple_returns

# Price panels: 2-60 dates x 1-6 assets, prices in a realistic positive range.
price_panels = st.tuples(st.integers(2, 60), st.integers(1, 6)).flatmap(
    lambda shape: arrays(np.float64, shape, elements=st.floats(1.0, 1000.0))
)
return_panels = st.tuples(st.integers(3, 60), st.integers(1, 6)).flatmap(
    lambda shape: arrays(np.float64, shape, elements=st.floats(-0.5, 0.5))
)


def frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=values.shape[0])
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(values.shape[1])])


@given(price_panels)
def test_returns_compound_back_to_total_price_change(values: np.ndarray) -> None:
    """prod_t (1 + r_t) = P_T / P_0 for every asset."""
    prices = frame(values)
    growth = (1.0 + simple_returns(prices)).prod()
    np.testing.assert_allclose(growth, prices.iloc[-1] / prices.iloc[0], rtol=1e-9)


@given(price_panels, st.floats(1e-3, 1e3))
def test_returns_invariant_to_rescaling_prices(values: np.ndarray, scale: float) -> None:
    """Multiplying all prices by a constant leaves returns unchanged.

    This is why later re-adjustment of Yahoo's adjusted closes (which rescales the whole
    history by a constant factor) does not change our returns (ADR-011).
    """
    prices = frame(values)
    np.testing.assert_allclose(
        simple_returns(prices * scale), simple_returns(prices), rtol=1e-9, atol=1e-12
    )


def test_returns_known_values() -> None:
    prices = frame(np.array([[100.0], [110.0], [99.0]]))
    np.testing.assert_allclose(simple_returns(prices)["A0"], [0.10, -0.10])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([[100.0]]), "at least two"),
        (np.array([[100.0], [np.nan]]), "missing"),
        (np.array([[100.0], [0.0]]), "positive"),
    ],
)
def test_simple_returns_rejects_bad_prices(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simple_returns(frame(values))


@given(return_panels)
def test_covariance_is_symmetric_psd_and_matches_numpy(values: np.ndarray) -> None:
    """Sigma = A * S is symmetric positive semidefinite and equals A * np.cov(ddof=1)."""
    mu, sigma = annualized_moments(frame(values), periods_per_year=252)
    s = sigma.to_numpy()
    np.testing.assert_allclose(s, s.T)
    expected = 252 * np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    np.testing.assert_allclose(s, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(mu.to_numpy(), 252 * values.mean(axis=0), rtol=1e-9, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(s)
    assert eigenvalues.min() >= -1e-9 * max(1.0, abs(eigenvalues).max())


@given(return_panels, st.data())
def test_moments_are_permutation_equivariant(values: np.ndarray, data: st.DataObject) -> None:
    """Reordering assets reorders mu and Sigma the same way and changes nothing else."""
    returns = frame(values)
    order = data.draw(st.permutations(list(returns.columns)))
    mu, sigma = annualized_moments(returns)
    mu_p, sigma_p = annualized_moments(returns[order])
    pd.testing.assert_series_equal(mu_p, mu[order])
    pd.testing.assert_frame_equal(sigma_p, sigma.loc[order, order])


@given(return_panels, st.integers(1, 365))
def test_moments_scale_linearly_with_periods_per_year(values: np.ndarray, a: int) -> None:
    returns = frame(values)
    mu_1, sigma_1 = annualized_moments(returns, periods_per_year=1)
    mu_a, sigma_a = annualized_moments(returns, periods_per_year=a)
    np.testing.assert_allclose(mu_a, a * mu_1, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sigma_a, a * sigma_1, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    ("values", "periods", "message"),
    [
        (np.zeros((5, 2)), 0, "positive"),
        (np.zeros((1, 2)), 252, "at least two"),
        (np.array([[0.1, np.nan], [0.2, 0.1]]), 252, "missing"),
    ],
)
def test_annualized_moments_rejects_bad_input(
    values: np.ndarray, periods: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        annualized_moments(frame(values), periods_per_year=periods)
