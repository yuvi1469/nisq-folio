"""Pre-registered analysis of the noisy grid (DECISIONS.md ADR-028).

Written and committed before any evaluation-instance noisy result was looked at. All cells have
the same 30 evaluation instances per N at every noise level (ADR-024), so every statistic is
paired by instance.

Crossover (the noise level where QAOA stops beating random feasible guessing, ADR-019):
    Let M(m) be the median of A over a cell's instances at noise level m, on the grid
    0 = m_0 < m_1 < ... < m_L. The crossover m* is the first point where M falls to 1:
      * M(m_0) <= 1                      -> m* = 0       (no advantage even without noise)
      * M(m_j) > 1 for every j           -> m* = +inf    (censored: "> m_L")
      * otherwise, with j the first index with M(m_j) <= 1, interpolate ln M between
        m_{j-1} and m_j:  t = ln M(m_{j-1}) / (ln M(m_{j-1}) - ln M(m_j))  in (0, 1], and
            m* = t * m_j                                         if m_{j-1} = 0 (linear in m)
            m* = exp(ln m_{j-1} + t (ln m_j - ln m_{j-1}))       otherwise      (linear in ln m)
    The first crossing defines m* even if M later rises above 1 again.

Uncertainty: percentile bootstrap over instances, B = 2000 replicates; the CI bounds are order
statistics (``inverted_cdf``), so a bound can be 0 or +inf. One resample of the
instance indices per (N, replicate) is drawn from stream (STREAM_ANALYSIS, N) and reused for
every mixer, denoising level and noise level at that N, so mixer and denoising comparisons
are paired. Different N have different instances, so their replicates are independent.

Tables:
    N1 curve        per (denoising, mixer, N, m): median A with 95% bootstrap CI, share A > 1,
                    median p_feasible (feasibility leakage), median E[gap]
    N2 crossover    per (denoising, mixer, N): m* with 95% CI and the censored shares
    N3 comparisons  P_boot(m*_a > m*_b) for mixer pairs at each N, and for N -> next N per mixer
    N4 decay        paired A(m) / A(0) per (denoising, mixer, N, m > 0) (geometric mean ratio)
    N5 mixers       ADR-026 T2 at every noise level
    N6 denoising    ADR-026 T3 at every noise level
    N7 deployment   ADR-026 T6 at every noise level (modal-optimal rate, OOS variance ratio)
    N8 routing      superseded full-device run vs compact-region run of record, paired at the
                    noise levels > 0 they share (robustness check, ADR-027)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nisq_folio.analysis.summary import (
    MIXER_PAIRS,
    denoising_table,
    deployment_table,
    mixer_table,
    paired_log_ratio,
)
from nisq_folio.seeding import STREAM_ANALYSIS, child_rng

BOOTSTRAP_REPLICATES = 2000
CI_LEVEL = 0.95
CELL = ["denoising", "mixer", "n"]


def crossover(levels: Sequence[float], medians: Sequence[float]) -> float:
    """Noise level m* where the median amplification first falls to 1 (module docstring).

    Args:
        levels: Strictly increasing noise levels, starting at 0 or above.
        medians: Median A at each level.

    Returns:
        m*, 0.0 if there is no advantage at the lowest level, ``inf`` if never crossed.
    """
    m = np.asarray(levels, dtype=float)
    a = np.log(np.maximum(np.asarray(medians, dtype=float), 1e-300))
    if m.ndim != 1 or m.shape != a.shape or m.size == 0 or np.any(np.diff(m) <= 0):
        raise ValueError("levels must be strictly increasing and match medians")
    if a[0] <= 0:
        return 0.0 if m[0] == 0 else float(m[0])
    below = np.flatnonzero(a <= 0)
    if below.size == 0:
        return float("inf")
    j = int(below[0])
    t = a[j - 1] / (a[j - 1] - a[j])
    if m[j - 1] == 0:
        return float(t * m[j])
    return float(np.exp(np.log(m[j - 1]) + t * (np.log(m[j]) - np.log(m[j - 1]))))


def complete_cells(qaoa: pd.DataFrame) -> pd.DataFrame:
    """Keep only cells with every instance present at every noise level of that N."""
    keep = []
    for _, g in qaoa.groupby("n"):
        instances, levels = set(g["instance"]), set(g["noise_level"])
        for _, cell in g.groupby(["denoising", "mixer"]):
            if len(levels) > 1 and len(cell) == len(instances) * len(levels):
                keep.append(cell)
    return pd.concat(keep, ignore_index=True) if keep else qaoa.iloc[0:0]


def _matrix(cell: pd.DataFrame, column: str = "A") -> tuple[np.ndarray, np.ndarray]:
    """(levels, values[instance, level]) for one cell, instances in sorted order."""
    wide = cell.pivot(index="instance", columns="noise_level", values=column).sort_index()
    return wide.columns.to_numpy(float), wide.to_numpy(float)


def bootstrap_indices(root_seed: int, n: int, instances: int, replicates: int) -> np.ndarray:
    """Resampled instance positions, shape (replicates, instances), from stream (ANALYSIS, n)."""
    rng = child_rng(root_seed, STREAM_ANALYSIS, int(n))
    return rng.integers(0, instances, size=(replicates, instances))


def _ci(values: np.ndarray) -> tuple[float, float]:
    tail = 100 * (1 - CI_LEVEL) / 2
    # Order statistics (no interpolation), so censored replicates (inf) give a finite or inf
    # bound instead of NaN.
    lo, hi = np.percentile(values, [tail, 100 - tail], method="inverted_cdf")
    return float(lo), float(hi)


def _replicate_crossovers(levels: np.ndarray, values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    medians = np.median(values[idx], axis=1)  # (replicates, levels)
    return np.array([crossover(levels, row) for row in medians])


def analyze(
    qaoa: pd.DataFrame,
    classical: pd.DataFrame,
    root_seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, pd.DataFrame]:
    """Compute tables N1-N7 (module docstring) from complete noisy cells.

    Args:
        qaoa: The QAOA table from ``analysis.summary.flatten``.
        classical: The classical table from ``analysis.summary.flatten``.
        root_seed: The config's root seed, for the bootstrap streams.
        replicates: Bootstrap replicates B.

    Returns:
        Tables keyed "N1".."N7"; empty DataFrames when no cell is complete.
    """
    data = complete_cells(qaoa)
    curve, cross, draws = [], [], {}
    for n, g in data.groupby("n"):
        idx = None
        for (den, mixer), cell in g.groupby(["denoising", "mixer"]):
            levels, values = _matrix(cell)
            if idx is None:
                idx = bootstrap_indices(root_seed, n, values.shape[0], replicates)
            _, feasible = _matrix(cell, "p_feasible")
            _, gap = _matrix(cell, "expected_gap")
            boot_medians = np.median(values[idx], axis=1)
            for k, m in enumerate(levels):
                lo, hi = _ci(boot_medians[:, k])
                curve.append({
                    "denoising": den, "mixer": mixer, "n": n, "noise_level": m,
                    "A_median": float(np.median(values[:, k])), "A_ci_low": lo, "A_ci_high": hi,
                    "share_A_gt_1": float(np.mean(values[:, k] > 1)),
                    "p_feasible_median": float(np.median(feasible[:, k])),
                    "gap_median": float(np.median(gap[:, k])),
                })  # fmt: skip
            reps = _replicate_crossovers(levels, values, idx)
            draws[(den, mixer, n)] = reps
            lo, hi = _ci(reps)
            cross.append({
                "denoising": den, "mixer": mixer, "n": n,
                "crossover": crossover(levels, np.median(values, axis=0)),
                "ci_low": lo, "ci_high": hi,
                "share_no_advantage": float(np.mean(reps == 0)),
                "share_censored": float(np.mean(np.isinf(reps))),
                "max_level": float(levels[-1]),
            })  # fmt: skip
    return {
        "N1": pd.DataFrame(curve),
        "N2": pd.DataFrame(cross),
        "N3": comparisons(draws),
        "N4": decay_table(data),
        "N5": by_level(data, mixer_table),
        "N6": by_level(data, denoising_table),
        "N7": by_level(data, lambda d: deployment_table(d, classical)),
    }


def by_level(data: pd.DataFrame, table: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Apply a noiseless ADR-026 table separately at every noise level and stack the results."""
    if data.empty:
        return pd.DataFrame()
    parts = [table(data[data["noise_level"] == m]).assign(noise_level=m) for m in levels_of(data)]
    return pd.concat(parts, ignore_index=True)


def levels_of(df: pd.DataFrame) -> list[float]:
    """Sorted distinct noise levels present."""
    return sorted(float(m) for m in df["noise_level"].unique())


def _prob_greater(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) over paired replicates, ties counted one half (inf == inf is a tie)."""
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def comparisons(draws: dict[tuple[str, str, int], np.ndarray]) -> pd.DataFrame:
    """N3: bootstrap probability that one crossover exceeds another (module docstring)."""
    rows: list[dict[str, Any]] = []
    for (den, mixer, n), reps in sorted(draws.items()):
        for first, second in MIXER_PAIRS:
            if mixer == first and (den, second, n) in draws:
                rows.append({
                    "denoising": den, "n": n, "comparison": f"{first} > {second}",
                    "p_greater": _prob_greater(reps, draws[(den, second, n)]),
                })  # fmt: skip
        larger = sorted(k[2] for k in draws if k[:2] == (den, mixer) and k[2] > n)
        if larger:
            rows.append({
                "denoising": den, "n": n, "comparison": f"{mixer}: N={n} > N={larger[0]}",
                "p_greater": _prob_greater(reps, draws[(den, mixer, larger[0])]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def decay_table(data: pd.DataFrame) -> pd.DataFrame:
    """N4: paired geometric-mean ratio A(m) / A(0) per cell and noise level m > 0."""
    rows: list[dict[str, Any]] = []
    for (den, mixer, n), cell in data.groupby(CELL):
        base = cell[cell["noise_level"] == 0].set_index("instance")["A"]
        for m in levels_of(cell):
            if m > 0 and not base.empty:
                a = cell[cell["noise_level"] == m].set_index("instance")["A"].loc[base.index]
                rows.append({"denoising": den, "mixer": mixer, "n": n, "noise_level": m,
                             **paired_log_ratio(a, base)})  # fmt: skip
    return pd.DataFrame(rows)


def routing_table(record: pd.DataFrame, superseded: pd.DataFrame) -> pd.DataFrame:
    """N8: A(superseded run) / A(run of record), paired by instance, at shared levels m > 0.

    Only (denoising, mixer, N, m) groups complete in both runs with identical instances count.
    """
    rows = []
    shared = set(levels_of(record)) & set(levels_of(superseded)) - {0.0}
    for (den, mixer, n), cell in record.groupby(CELL):
        old = superseded[
            (superseded["denoising"] == den)
            & (superseded["mixer"] == mixer)
            & (superseded["n"] == n)
        ]
        for m in sorted(shared):
            a = cell[cell["noise_level"] == m].set_index("instance")["A"]
            b = old[old["noise_level"] == m].set_index("instance")["A"]
            if len(a) and a.index.equals(b.index.intersection(a.index)) and len(a) == len(b):
                rows.append({"denoising": den, "mixer": mixer, "n": n, "noise_level": m,
                             **paired_log_ratio(b.loc[a.index], a)})  # fmt: skip
    return pd.DataFrame(rows)


def routing_robust(record_cross: pd.DataFrame, superseded_cross: pd.DataFrame) -> pd.DataFrame:
    """N8 criterion: is each superseded crossover inside the record's 95% CI (ADR-028)?"""
    merged = record_cross.merge(
        superseded_cross[[*CELL, "crossover"]], on=CELL, suffixes=("", "_superseded")
    )
    s = merged["crossover_superseded"]
    merged["inside_ci"] = (s >= merged["ci_low"]) & (s <= merged["ci_high"])
    return merged[[*CELL, "crossover", "ci_low", "ci_high", "crossover_superseded", "inside_ci"]]


def verdicts(curve: pd.DataFrame) -> pd.DataFrame:
    """Apply the ADR-026 interpretation rule at each noise level.

    A cell "beats random" iff its median A > 1 and A > 1 on at least two thirds of instances;
    otherwise it is "not supported".
    """
    out = curve[[*CELL, "noise_level", "A_median", "share_A_gt_1"]].copy()
    beats = (out["A_median"] > 1) & (out["share_A_gt_1"] >= 2 / 3)
    out["verdict"] = np.where(beats, "beats random", "not supported")
    return out
