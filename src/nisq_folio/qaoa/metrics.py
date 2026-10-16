"""Quality metrics of a QAOA output distribution (DECISIONS.md ADR-019).

At N <= 14 a QAOA run draws ~10^5 samples, more than the 2^N <= 16,384 bitstrings, so "best
bitstring ever sampled" would also be found by blind guessing. QAOA is therefore judged by how
it concentrates probability on good solutions, measured on its final output distribution p(x):

    p_opt       = sum of p(x) over optimal feasible x
    p_feasible  = sum of p(x) over x with 1^T x = K
    E[gap]      = sum_x p(x) * g(x),   g(x) = (f(x) - f*) / (f_worst - f*) if feasible, else 1
    amplification_feasible = p_opt / p_opt(uniform over feasible x)   (uniform: m / C(N, K))
    amplification_uniform  = p_opt / p_opt(uniform over all 2^N x)    (uniform: m / 2^N)
    samples_to_90 = ceil( ln(0.1) / ln(1 - p_opt) )   samples needed to see an optimum w.p. 0.9

where m is the number of optimal selections (normally 1). Amplification > 1 means QAOA does
better than random guessing; the noise crossover is where it falls back to 1 (ADR-019).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from nisq_folio.problem import FloatArray, PortfolioProblem
from nisq_folio.qubo.encoding import all_bitstrings


@dataclass(frozen=True)
class SolutionLandscape:
    """The exact objective over all 2^N bitstrings, for scoring output distributions.

    Attributes:
        objective: f(x) for every bitstring (little-endian).
        feasible: Mask of bitstrings with exactly K ones.
        optimal: Mask of feasible bitstrings with f(x) = f* (within tolerance).
        f_star: Optimal feasible objective.
        f_worst: Worst feasible objective.
    """

    objective: FloatArray
    feasible: np.ndarray
    optimal: np.ndarray
    f_star: float
    f_worst: float

    def __post_init__(self) -> None:
        """Check that the masks are consistent: some x is feasible and optimal implies feasible."""
        if not np.any(self.feasible):
            raise ValueError("landscape has no feasible bitstring")
        if not np.any(self.optimal) or np.any(self.optimal & ~self.feasible):
            raise ValueError("optimal bitstrings must be a non-empty subset of the feasible ones")

    @classmethod
    def from_problem(cls, problem: PortfolioProblem) -> SolutionLandscape:
        """Evaluate f on all 2^N bitstrings and locate the feasible optimum (N <= ~20).

        f*, f_worst and the optimal set are all taken from the same ``objective`` array, so
        gaps are exactly 0 at the optimum and exactly 1 at the worst feasible selection
        (mixing two summation orders would leave ~1e-16 inconsistencies).
        """
        bits = all_bitstrings(problem.n)
        a, b = problem.quadratic_coefficient, problem.linear_coefficient
        objective = a * np.einsum("ki,ij,kj->k", bits, problem.sigma, bits) - b * bits @ problem.mu
        feasible = bits.sum(axis=1) == problem.k
        f_star = float(objective[feasible].min())
        f_worst = float(objective[feasible].max())
        tolerance = 1e-12 * max(1.0, abs(f_star))
        optimal = feasible & (objective <= f_star + tolerance)
        return cls(objective, feasible, optimal, f_star, f_worst)

    def gaps(self) -> FloatArray:
        """g(x) for every bitstring: normalized feasible gap, and 1 for infeasible x."""
        span = self.f_worst - self.f_star
        gap = np.ones_like(self.objective)
        if span > 0:
            gap[self.feasible] = (self.objective[self.feasible] - self.f_star) / span
        else:
            gap[self.feasible] = 0.0
        gap[self.optimal] = 0.0
        return gap


@dataclass(frozen=True)
class DistributionMetrics:
    """Metrics of one output distribution; see the module docstring for definitions."""

    p_optimal: float
    p_feasible: float
    expected_gap: float
    amplification_feasible: float
    amplification_uniform: float
    samples_to_90: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dict (JSON-serializable)."""
        return asdict(self)


def distribution_metrics(
    probabilities: FloatArray, landscape: SolutionLandscape
) -> DistributionMetrics:
    """Score a distribution over bitstrings against the exact landscape.

    Args:
        probabilities: p(x) for every bitstring (little-endian), summing to 1.
        landscape: The exact objective landscape of the same problem.

    Returns:
        The metrics. ``samples_to_90`` is ``inf`` if p_opt = 0.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != landscape.objective.shape:
        raise ValueError("probabilities and landscape have different lengths")
    if np.any(p < -1e-12) or not math.isclose(p.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("probabilities must be non-negative and sum to 1")
    n_optimal = int(landscape.optimal.sum())
    p_opt = float(p[landscape.optimal].sum())
    uniform_feasible = n_optimal / int(landscape.feasible.sum())
    uniform_all = n_optimal / p.size
    return DistributionMetrics(
        p_optimal=p_opt,
        p_feasible=float(p[landscape.feasible].sum()),
        expected_gap=float(p @ landscape.gaps()),
        amplification_feasible=p_opt / uniform_feasible,
        amplification_uniform=p_opt / uniform_all,
        samples_to_90=samples_to_probability(p_opt, 0.9),
    )


def samples_to_probability(p_success: float, target: float) -> float:
    """Return ceil(ln(1 - target) / ln(1 - p)): i.i.d. samples to succeed at least once w.p. target.

    Returns 1 if p >= 1 and ``inf`` if p <= 0.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    if p_success >= 1.0:
        return 1.0
    if p_success <= 0.0:
        return math.inf
    return float(math.ceil(math.log(1.0 - target) / math.log1p(-p_success)))
