"""Continuous long-only Markowitz optimization followed by rounding to K assets."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from nisq_folio.problem import FloatArray, PortfolioProblem, Solution, make_solution


def markowitz_weights(problem: PortfolioProblem, tol: float = 1e-12) -> FloatArray:
    """Solve the continuous relaxation of the problem on the long-only simplex.

        minimize_w   lambda * w^T Sigma w - mu^T w
        subject to   sum_i w_i = 1,  w_i >= 0

    This is the same objective as f(x) at w = x / K, with the cardinality and equal-weight
    constraints dropped. Solved with SLSQP from the equal-weight start w_i = 1/N, with the
    analytic gradient 2 * lambda * Sigma w - mu. Deterministic: no randomness involved.

    Args:
        problem: The problem instance.
        tol: SLSQP convergence tolerance.

    Returns:
        Optimal weights, shape (N,), clipped at 0 and renormalized to sum exactly 1.

    Raises:
        RuntimeError: If the optimizer does not report success.
    """
    lam, sigma, mu, n = problem.risk_aversion, problem.sigma, problem.mu, problem.n

    def objective(w: FloatArray) -> float:
        return float(lam * w @ sigma @ w - mu @ w)

    def gradient(w: FloatArray) -> FloatArray:
        return 2.0 * lam * sigma @ w - mu

    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": tol, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP did not converge: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def markowitz_rounding(problem: PortfolioProblem) -> Solution:
    """Select the K assets with the largest continuous Markowitz weights.

    Solves :func:`markowitz_weights`, then keeps the K largest weights (ties go to the lower
    index) and holds them in equal weights, as the discrete problem requires.

    Args:
        problem: The problem instance.

    Returns:
        The rounded solution. ``n_evaluations`` is reported as 1 (one relaxation solve), because
        the optimizer's continuous function evaluations are not comparable to discrete moves.
    """
    weights = markowitz_weights(problem)
    order = np.argsort(-weights, kind="stable")
    return make_solution(problem, order[: problem.k], "markowitz_rounding", n_evaluations=1)
