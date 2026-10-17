"""Pre-registered summaries of experiment results (DECISIONS.md ADR-026).

All comparisons are paired by evaluation instance (every cell has the same 30 instances,
ADR-024). Paired tests are two-sided Wilcoxon signed-rank on the per-instance differences of
ln A (zeros dropped). Instances share assets, so p-values are descriptive, not confirmatory.
Ratios of A are summarized as the geometric mean ratio exp(median of ln A_1 - ln A_2).

Tables (noiseless, m = 0):
    T1 cells        per (denoising, mixer, N): median A [IQR], share A > 1, median E[gap],
                    modal-feasible-optimal rate, median total shots
    T2 mixers       paired A ratio for adapt/xy_ring, xy_ring/x, adapt/x (denoising = none)
    T3 denoising    paired A ratio rmt_clip/none per (mixer, N)
    T4 classical    optimal rate of greedy, Markowitz + rounding, SA at each budget, per N
    T5 budget       median QAOA total shots vs the smallest SA budget with >= 90% optimal
    T6 deployment   median realized out-of-sample variance of the modal selection relative to
                    the true optimum's, per (mixer, N), denoising = none
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MIXER_PAIRS = (("adapt", "xy_ring"), ("xy_ring", "x"), ("adapt", "x"))


def flatten(records: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn task records into a QAOA table and a classical table (one row per task)."""
    qaoa_rows, classical_rows = [], []
    for r in records:
        task = r["task"]
        key = {"n": task["n"], "instance": task["instance"], "denoising": task["denoising"]}
        if task["kind"] == "classical":
            row = {
                **key,
                "f_star": r["f_star"],
                "f_worst": r["f_worst"],
                "optimum_oos_variance": r["optimum_out_of_sample"]["realized_variance"],
            }
            tol = 1e-12 * max(1.0, abs(r["f_star"]))
            row["greedy_optimal"] = r["greedy"]["objective"] <= r["f_star"] + tol
            row["markowitz_optimal"] = r["markowitz_rounding"]["objective"] <= r["f_star"] + tol
            for budget, sol in r["simulated_annealing"].items():
                row[f"sa_{budget}_optimal"] = sol["objective"] <= r["f_star"] + tol
            classical_rows.append(row)
        else:
            m, mode = r["run"]["metrics"], r["mode_feasible"]
            qaoa_rows.append(
                {
                    **key,
                    "mixer": task["mixer"],
                    "noise_level": task["noise_level"],
                    "A": m["amplification_feasible"],
                    "p_optimal": m["p_optimal"],
                    "p_feasible": m["p_feasible"],
                    "expected_gap": m["expected_gap"],
                    "total_shots": r["run"]["total_shots"],
                    "mode_optimal": mode["is_optimal"],
                    "mode_oos_variance": mode["out_of_sample"]["realized_variance"],
                }
            )
    return pd.DataFrame(qaoa_rows), pd.DataFrame(classical_rows)


def paired_log_ratio(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Geometric-mean ratio a/b, share of instances with a > b, and Wilcoxon p (paired)."""
    diff = np.log(np.maximum(a.to_numpy(float), 1e-300)) - np.log(
        np.maximum(b.to_numpy(float), 1e-300)
    )
    nonzero = diff[diff != 0]
    return {
        "ratio": float(np.exp(np.median(diff))),
        "share_better": float(np.mean(diff > 0)),
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size else float("nan"),
        "pairs": int(diff.size),
    }


def cell_table(qaoa: pd.DataFrame) -> pd.DataFrame:
    """T1 (module docstring)."""
    rows = []
    for (den, mixer, n), g in qaoa.groupby(["denoising", "mixer", "n"]):
        rows.append(
            {
                "denoising": den,
                "mixer": mixer,
                "N": n,
                "instances": len(g),
                "A_median": g["A"].median(),
                "A_q25": g["A"].quantile(0.25),
                "A_q75": g["A"].quantile(0.75),
                "share_A_gt_1": (g["A"] > 1).mean(),
                "gap_median": g["expected_gap"].median(),
                "mode_optimal": g["mode_optimal"].mean(),
                "shots_median": g["total_shots"].median(),
            }
        )
    return pd.DataFrame(rows)


def _paired(qaoa: pd.DataFrame, filters_a: dict, filters_b: dict, by: str = "n") -> list[dict]:
    out = []
    for n in sorted(qaoa[by].unique()):
        a = _select(qaoa, {**filters_a, by: n}).set_index("instance")["A"]
        b = _select(qaoa, {**filters_b, by: n}).set_index("instance")["A"]
        common = a.index.intersection(b.index)
        if len(common):
            out.append({by: n, **paired_log_ratio(a.loc[common], b.loc[common])})
    return out


def _select(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for column, value in filters.items():
        mask &= df[column].to_numpy() == value
    return df[mask]


def mixer_table(qaoa: pd.DataFrame) -> pd.DataFrame:
    """T2 (module docstring)."""
    rows = []
    for first, second in MIXER_PAIRS:
        if {first, second} <= set(qaoa["mixer"]):
            for r in _paired(
                qaoa, {"mixer": first, "denoising": "none"}, {"mixer": second, "denoising": "none"}
            ):
                rows.append({"comparison": f"{first} / {second}", **r})
    return pd.DataFrame(rows)


def denoising_table(qaoa: pd.DataFrame) -> pd.DataFrame:
    """T3 (module docstring)."""
    rows = []
    if {"none", "rmt_clip"} <= set(qaoa["denoising"]):
        for mixer in sorted(qaoa["mixer"].unique()):
            for r in _paired(
                qaoa,
                {"mixer": mixer, "denoising": "rmt_clip"},
                {"mixer": mixer, "denoising": "none"},
            ):
                rows.append({"mixer": mixer, **r})
    return pd.DataFrame(rows)


def classical_table(classical: pd.DataFrame) -> pd.DataFrame:
    """T4 (module docstring)."""
    columns = [c for c in classical.columns if c.endswith("_optimal")]
    return classical.groupby(["denoising", "n"])[columns].mean().reset_index()


def budget_table(qaoa: pd.DataFrame, classical: pd.DataFrame, target: float = 0.9) -> pd.DataFrame:
    """T5 (module docstring); ``sa_budget_90`` is None if no budget reaches the target."""
    sa_cols = sorted(
        (c for c in classical.columns if c.startswith("sa_")), key=lambda c: int(c.split("_")[1])
    )
    rows = []
    for (den, n), g in classical.groupby(["denoising", "n"]):
        reached = [int(c.split("_")[1]) for c in sa_cols if g[c].mean() >= target]
        sa = min(reached) if reached else None
        for mixer, q in _select(qaoa, {"denoising": den, "n": n}).groupby("mixer"):
            rows.append(
                {
                    "denoising": den,
                    "N": n,
                    "mixer": mixer,
                    "qaoa_total_shots": q["total_shots"].median(),
                    "sa_budget_90": sa,
                }
            )
    return pd.DataFrame(rows)


def deployment_table(qaoa: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    """T6 (module docstring)."""
    merged = _select(qaoa, {"denoising": "none"}).merge(
        _select(classical, {"denoising": "none"})[["n", "instance", "optimum_oos_variance"]],
        on=["n", "instance"],
    )
    merged["ratio"] = merged["mode_oos_variance"] / merged["optimum_oos_variance"]
    return (
        merged.groupby(["mixer", "n"])
        .agg(oos_variance_ratio_median=("ratio", "median"), mode_optimal=("mode_optimal", "mean"))
        .reset_index()
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (floats to 3 significant figures)."""

    def cell(value: Any) -> str:  # noqa: ANN401
        if isinstance(value, float | np.floating):
            return "nan" if np.isnan(value) else f"{value:.3g}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
