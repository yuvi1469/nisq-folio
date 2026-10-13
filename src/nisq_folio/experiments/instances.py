"""Problem instances: which N assets of the universe each instance uses.

The same instances are reused by every solver and every experiment cell, so comparisons are
paired (PROJECT_PLAN.md §3).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import pandas as pd

from nisq_folio.data.returns import annualized_moments
from nisq_folio.problem import PortfolioProblem
from nisq_folio.seeding import STREAM_INSTANCES, child_rng


def sample_instance_tickers(
    universe: Sequence[str], n: int, n_instances: int, root_seed: int
) -> list[tuple[str, ...]]:
    """Draw distinct N-asset subsets of the universe, uniformly at random.

    Uses stream (STREAM_INSTANCES, n), so instances of size N do not change when other sizes
    or instance counts are added. Tickers inside an instance keep universe order.

    Args:
        universe: All tickers, in config order.
        n: Assets per instance.
        n_instances: Number of distinct instances to draw.
        root_seed: The config's root seed.

    Returns:
        ``n_instances`` distinct ticker tuples.

    Raises:
        ValueError: If N is out of range or more instances are requested than exist.
    """
    if not 2 <= n <= len(universe):
        raise ValueError(f"n must be in [2, {len(universe)}], got {n}")
    if n_instances > comb(len(universe), n):
        raise ValueError(f"only {comb(len(universe), n)} distinct instances of size {n} exist")
    rng = child_rng(root_seed, STREAM_INSTANCES, n)
    seen: set[tuple[int, ...]] = set()
    instances: list[tuple[str, ...]] = []
    while len(instances) < n_instances:
        idx = tuple(sorted(int(i) for i in rng.choice(len(universe), size=n, replace=False)))
        if idx not in seen:
            seen.add(idx)
            instances.append(tuple(universe[i] for i in idx))
    return instances


def build_problem(
    returns: pd.DataFrame, tickers: Sequence[str], k: int, risk_aversion: float
) -> PortfolioProblem:
    """Build a problem from the returns of the given tickers.

    Args:
        returns: Daily returns (estimation window only), columns include ``tickers``.
        tickers: The instance's assets.
        k: Assets to select.
        risk_aversion: lambda in the objective.

    Returns:
        The problem, with mu and Sigma annualized from ``returns[tickers]``.
    """
    mu, sigma = annualized_moments(returns.loc[:, list(tickers)])
    return PortfolioProblem.from_moments(mu, sigma, k, risk_aversion)
