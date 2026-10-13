"""Calibration of the risk-aversion parameter lambda (DECISIONS.md ADR-015)."""

from __future__ import annotations

import numpy as np

from nisq_folio.baselines.brute_force import all_subsets
from nisq_folio.problem import PortfolioProblem


def balanced_risk_aversion(problem: PortfolioProblem) -> float:
    """Return the lambda at which the return and risk terms vary equally across selections.

    Over all K-subsets S, with equal weights w = x / K, let R(S) = mu^T w (portfolio return)
    and V(S) = w^T Sigma w (portfolio variance). Then

        lambda_bal = sd_S[R(S)] / sd_S[V(S)]

    makes lambda * V and R equally spread across the feasible set, so neither term decides the
    optimum alone. ``problem.risk_aversion`` is ignored.

    Args:
        problem: The instance (only mu, Sigma, K are used).

    Returns:
        lambda_bal (``inf`` if every subset has the same variance).
    """
    subsets = all_subsets(problem.n, problem.k)
    k = problem.k
    returns = problem.mu[subsets].sum(axis=1) / k
    variances = problem.sigma[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2)) / k**2
    spread = float(np.std(variances))
    return float(np.std(returns)) / spread if spread > 0 else float("inf")
