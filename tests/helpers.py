"""Test helpers shared across test modules."""

import numpy as np
from hypothesis import strategies as st

from nisq_folio.problem import PortfolioProblem


def random_problem(n: int, k: int, risk_aversion: float, seed: int) -> PortfolioProblem:
    """Build a random but realistic-scale instance from an explicit seed.

    mu_i ~ N(0.12, 0.08^2); Sigma = F F^T / n + diag(d) with F_ij ~ N(0, 0.25^2) and
    d_i ~ U(0.01, 0.05), so Sigma is symmetric positive definite with variances near 0.03-0.1.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.12, 0.08, size=n)
    factors = rng.normal(0.0, 0.25, size=(n, n))
    sigma = factors @ factors.T / n + np.diag(rng.uniform(0.01, 0.05, size=n))
    return PortfolioProblem(mu, sigma, k, risk_aversion, tuple(f"A{i}" for i in range(n)))


@st.composite
def problems(draw: st.DrawFn, min_n: int = 3, max_n: int = 8) -> PortfolioProblem:
    """Hypothesis strategy: random instances with 1 <= K < N and lambda in [0, 20]."""
    n = draw(st.integers(min_n, max_n))
    k = draw(st.integers(1, n - 1))
    risk_aversion = draw(st.sampled_from([0.0, 0.5, 2.0, 4.5, 20.0]))
    seed = draw(st.integers(0, 2**32 - 1))
    return random_problem(n, k, risk_aversion, seed)
