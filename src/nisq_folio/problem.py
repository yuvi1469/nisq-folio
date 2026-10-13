"""The cardinality-constrained portfolio selection problem shared by every solver.

Select exactly K of N assets and hold them in equal weights, w = x / K with x in {0,1}^N and
sum(x) = K. Minimize the mean-variance objective of that portfolio (DECISIONS.md ADR-015):

    f(x) = lambda * w^T Sigma w - mu^T w
         = (lambda / K^2) * x^T Sigma x - (1 / K) * mu^T x

where mu is the annualized mean return vector, Sigma the annualized covariance matrix, and
lambda >= 0 the risk aversion. Classical baselines, the QUBO encoding, and QAOA all solve this
same problem, so their results are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PortfolioProblem:
    """One problem instance.

    Attributes:
        mu: Annualized mean returns, shape (N,).
        sigma: Annualized covariance matrix, shape (N, N), symmetric.
        k: Number of assets to select, 1 <= K < N.
        risk_aversion: lambda >= 0 in the objective.
        asset_names: Name of each asset, length N, unique.
    """

    mu: FloatArray
    sigma: FloatArray
    k: int
    risk_aversion: float
    asset_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate inputs and store read-only float64 copies of the arrays."""
        mu = np.array(self.mu, dtype=np.float64)
        sigma = np.array(self.sigma, dtype=np.float64)
        if mu.ndim != 1 or mu.size < 2:
            raise ValueError("mu must be a 1-D array with at least two assets")
        n = mu.size
        if sigma.shape != (n, n):
            raise ValueError(f"sigma must have shape ({n}, {n}), got {sigma.shape}")
        if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
            raise ValueError("mu and sigma must be finite")
        if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12):
            raise ValueError("sigma must be symmetric")
        if (
            isinstance(self.k, bool)
            or not isinstance(self.k, int | np.integer)
            or not 1 <= self.k < n
        ):
            raise ValueError(f"k must be an integer with 1 <= k < {n}, got {self.k!r}")
        if not (np.isfinite(self.risk_aversion) and self.risk_aversion >= 0):
            raise ValueError(f"risk_aversion must be finite and >= 0, got {self.risk_aversion}")
        if len(self.asset_names) != n or len(set(self.asset_names)) != n:
            raise ValueError("asset_names must contain N unique names")
        mu.setflags(write=False)
        sigma.setflags(write=False)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "risk_aversion", float(self.risk_aversion))
        object.__setattr__(self, "asset_names", tuple(self.asset_names))

    @classmethod
    def from_moments(
        cls, mu: pd.Series, sigma: pd.DataFrame, k: int, risk_aversion: float
    ) -> PortfolioProblem:
        """Build a problem from labelled moments, aligning ``sigma`` to the order of ``mu``.

        Args:
            mu: Annualized mean returns indexed by asset name.
            sigma: Annualized covariance indexed and columned by asset name.
            k: Number of assets to select.
            risk_aversion: lambda in the objective.

        Returns:
            The problem instance, with assets in the order of ``mu.index``.
        """
        names = [str(a) for a in mu.index]
        aligned = sigma.loc[mu.index, mu.index]
        return cls(mu.to_numpy(), aligned.to_numpy(), k, risk_aversion, tuple(names))

    @property
    def n(self) -> int:
        """Number of assets N."""
        return int(self.mu.size)

    @property
    def quadratic_coefficient(self) -> float:
        """A = lambda / K^2, the weight on x^T Sigma x in f(x)."""
        return self.risk_aversion / self.k**2

    @property
    def linear_coefficient(self) -> float:
        """B = 1 / K, the weight on mu^T x in f(x) (which enters with a minus sign)."""
        return 1.0 / self.k

    def objective(self, x: NDArray[np.integer] | NDArray[np.bool_]) -> float:
        """Evaluate f(x) = (lambda / K^2) x^T Sigma x - (1 / K) mu^T x for a 0/1 vector.

        The formula is evaluated for any binary x, feasible or not; use :meth:`is_feasible`
        to check sum(x) = K.

        Args:
            x: Binary vector of length N.

        Returns:
            The objective value.
        """
        xf = self._as_binary(x)
        return float(
            self.quadratic_coefficient * xf @ self.sigma @ xf
            - self.linear_coefficient * self.mu @ xf
        )

    def objective_of_subsets(self, subsets: IntArray) -> FloatArray:
        """Evaluate f for many K-subsets at once.

        Args:
            subsets: Integer array of shape (m, K); each row lists distinct selected indices.

        Returns:
            Objective values, shape (m,).
        """
        subsets = np.asarray(subsets, dtype=np.int64)
        if subsets.ndim != 2 or subsets.shape[1] != self.k:
            raise ValueError(f"subsets must have shape (m, {self.k})")
        quad = self.sigma[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2))
        lin = self.mu[subsets].sum(axis=1)
        return self.quadratic_coefficient * quad - self.linear_coefficient * lin

    def is_feasible(self, x: NDArray[np.integer] | NDArray[np.bool_]) -> bool:
        """Return True if x is binary with exactly K ones."""
        return int(self._as_binary(x).sum()) == self.k

    def _as_binary(self, x: NDArray[np.integer] | NDArray[np.bool_]) -> FloatArray:
        arr = np.asarray(x)
        if arr.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},), got {arr.shape}")
        if not np.isin(arr, (0, 1)).all():
            raise ValueError("x must be a 0/1 vector")
        return arr.astype(np.float64)


@dataclass(frozen=True)
class Solution:
    """A solver's answer to a :class:`PortfolioProblem`.

    Attributes:
        selection: Sorted indices of the selected assets (length K).
        objective: f(x) of the selection.
        solver: Name of the solver that produced it.
        n_evaluations: Objective evaluations used, counting a full or an incremental
            (swap / add) evaluation as one. A rough, solver-independent cost measure.
    """

    selection: tuple[int, ...]
    objective: float
    solver: str
    n_evaluations: int

    def to_vector(self, n: int) -> IntArray:
        """Return the selection as a 0/1 vector of length ``n``."""
        x = np.zeros(n, dtype=np.int64)
        x[list(self.selection)] = 1
        return x


def make_solution(
    problem: PortfolioProblem, selected: NDArray[np.integer], solver: str, n_evaluations: int
) -> Solution:
    """Build a :class:`Solution`, recomputing the objective from scratch as a safeguard.

    Args:
        problem: The problem that was solved.
        selected: Indices of the selected assets (any order, no duplicates).
        solver: Solver name.
        n_evaluations: Evaluations used by the solver.

    Returns:
        The solution, with ``selection`` sorted.

    Raises:
        ValueError: If the selection is not a feasible K-subset.
    """
    selection = tuple(sorted(int(i) for i in selected))
    if len(selection) != problem.k or len(set(selection)) != problem.k:
        raise ValueError(f"{solver}: selection {selection} is not a set of K={problem.k} assets")
    if selection[0] < 0 or selection[-1] >= problem.n:
        raise ValueError(f"{solver}: selection {selection} has out-of-range indices")
    x = np.zeros(problem.n, dtype=np.int64)
    x[list(selection)] = 1
    return Solution(selection, problem.objective(x), solver, n_evaluations)
