"""Problem instances: which N assets of the universe each instance uses.

The same instances are reused by every solver and every experiment cell, so comparisons are
paired (PROJECT_PLAN.md §3).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from math import comb

import pandas as pd

from nisq_folio.data.returns import annualized_moments
from nisq_folio.problem import PortfolioProblem
from nisq_folio.seeding import STREAM_INSTANCES, child_rng


def sample_instance_tickers(
    universe: Sequence[str],
    n: int,
    n_instances: int,
    root_seed: int,
    stream: int = STREAM_INSTANCES,
    exclude: Collection[tuple[str, ...]] = (),
) -> list[tuple[str, ...]]:
    """Draw distinct N-asset subsets of the universe, uniformly at random.

    Uses stream (``stream``, n), so instances of size N do not change when other sizes or
    instance counts are added. Tickers inside an instance keep universe order. Evaluation
    instances use the default stream; QAOA tuning uses a separate stream and excludes the
    evaluation instances, so settings are never chosen on the instances they are scored on.

    Args:
        universe: All tickers, in config order.
        n: Assets per instance.
        n_instances: Number of distinct instances to draw.
        root_seed: The config's root seed.
        stream: First key of the random stream (see ``nisq_folio.seeding``).
        exclude: Instances that must not be returned.

    Returns:
        ``n_instances`` distinct ticker tuples, none of them in ``exclude``.

    Raises:
        ValueError: If N is out of range or more instances are requested than exist.
    """
    if not 2 <= n <= len(universe):
        raise ValueError(f"n must be in [2, {len(universe)}], got {n}")
    excluded = {tuple(t) for t in exclude if len(t) == n}
    if n_instances > comb(len(universe), n) - len(excluded):
        raise ValueError(f"only {comb(len(universe), n)} distinct instances of size {n} exist")
    rng = child_rng(root_seed, stream, n)
    seen: set[tuple[int, ...]] = set()
    instances: list[tuple[str, ...]] = []
    while len(instances) < n_instances:
        idx = tuple(sorted(int(i) for i in rng.choice(len(universe), size=n, replace=False)))
        if idx in seen:
            continue
        seen.add(idx)
        tickers = tuple(universe[i] for i in idx)
        if tickers not in excluded:
            instances.append(tickers)
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
