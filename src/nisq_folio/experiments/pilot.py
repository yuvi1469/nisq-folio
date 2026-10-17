"""Pilot for the noisy grid and the pre-registered rules that turn it into ADR-025.

Measurements (tuning instances only; evaluation instances are never touched):

1. Timing: seconds per noisy circuit evaluation for each mixer and N at m = 1, with Aer on
   automatic threads, and (for ``thread_test_sizes``) also capped at 1 thread, to decide whether
   W parallel single-thread workers beat one multi-thread worker: throughput gain
   = W * t_auto / t_1thread.
2. Bracketing: full noisy runs at N = ``accuracy_size`` for every mixer and the
   ``accuracy_noise_levels``, on ``accuracy_instances`` tuning instances, recording A.

Rule 1, upper noise level (pre-registered): m_max is the smallest ladder value >= 1 at which
the median pilot A of every mixer is <= 1 (QAOA no better than random feasible guessing), taken
by monotone interpolation in log m between the measured pilot levels; if A > 1 at every measured
level for some mixer, m_max is the largest ladder value.

Rule 3, parallelism (pre-registered): run the noisy grid with W = 3 single-thread workers if the
measured gain 3 * t_auto / t_1thread exceeds 1.2 at every ``thread_test_sizes`` N, else with 1
worker. The gain measured at those N is applied to them only; larger N are assumed to gain
nothing (conservative), so the hours estimate is not optimistic.

Rule 2, levels and budget (pre-registered): start from every ladder value <= m_max (plus m = 0).
While the estimated noisy-grid hours exceed ``budget_hours``, drop the smallest non-zero level,
but keep at least ``min_noisy_levels``; if still over budget, report the overrun rather than
silently cutting cells (the user decides).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise

import numpy as np

from nisq_folio.mixers.variants import VariantSettings
from nisq_folio.noise.density import ReferenceRates
from nisq_folio.noise.runner import make_evaluator
from nisq_folio.problem import PortfolioProblem
from nisq_folio.qubo.penalty import sufficient_penalty


def time_evaluation(
    problem: PortfolioProblem,
    settings: VariantSettings,
    rates_level: float,
    rates: ReferenceRates,
    repeats: int,
    max_parallel_threads: int,
    backend: str,
) -> float:
    """Median seconds of one noisy evaluation (after a warm-up that includes compilation).

    ADAPT is timed on a representative sequence: the XY ring alternating with single pairs.
    """
    evaluator, _ = make_evaluator(
        problem, sufficient_penalty(problem), settings, rates_level, rates,
        backend=backend, max_parallel_threads=max_parallel_threads,
    )  # fmt: skip
    p = settings.layers
    if settings.mixer == "adapt":
        names = ["xy_ring" if i % 2 == 0 else f"xy_0_{1 + i}" for i in range(p)]
    else:
        names = [settings.mixer] * p
    angles = np.linspace(0.2, 0.9, p)
    evaluator.distribution(names, angles, angles)  # warm-up: transpile + first run
    times = []
    for r in range(repeats):
        start = time.perf_counter()
        evaluator.distribution(names, angles + 0.01 * (r + 1), angles)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def interpolated_crossing(levels: Sequence[float], medians: Sequence[float]) -> float | None:
    """First m where median A falls to 1, interpolating linearly in log m; None if never.

    Args:
        levels: Increasing positive noise levels.
        medians: Median A at each level.
    """
    pts = list(zip(levels, medians, strict=True))
    if pts[0][1] <= 1.0:
        return pts[0][0]
    for (m0, a0), (m1, a1) in pairwise(pts):
        if a0 > 1.0 >= a1:
            t = (a0 - 1.0) / (a0 - a1)
            return float(np.exp(np.log(m0) + t * (np.log(m1) - np.log(m0))))
    return None


def upper_noise_level(crossings: Mapping[str, float | None], ladder: Sequence[float]) -> float:
    """Rule 1 (module docstring): the smallest ladder value >= 1 at or above every crossing."""
    ladder = sorted(ladder)
    if any(c is None for c in crossings.values()):
        return ladder[-1]
    needed = max(max(c for c in crossings.values() if c is not None), 1.0)
    return next((m for m in ladder if m >= needed), ladder[-1])


def choose_levels(
    ladder: Sequence[float],
    m_max: float,
    hours_for: Callable[[list[float]], float],
    budget_hours: float,
    min_noisy_levels: int,
) -> tuple[list[float], float, bool]:
    """Rule 2 (module docstring). Returns (noisy levels, estimated hours, within_budget)."""
    levels = sorted(m for m in ladder if m <= m_max)
    hours = hours_for(levels)
    while hours > budget_hours and len(levels) > min_noisy_levels:
        levels = levels[1:]
        hours = hours_for(levels)
    return levels, hours, hours <= budget_hours
