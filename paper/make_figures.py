"""Generate the paper's figures directly from recorded results (no hand-typed numbers).

Usage::

    uv run python paper/make_figures.py --config configs/main.yaml

Writes ``paper/generated/fig_*.pdf``:

    fig_noiseless   median A [IQR] vs N per mixer at m = 0 (ADR-026, sample covariance)
    fig_noise       median A with 95% bootstrap CI vs noise level m, one panel per complete N
                    (ADR-028 N1, sample covariance)
    fig_leakage     median feasible probability vs m, same panels (ADR-028 N1)
    fig_spectrum    eigenvalues of the in-sample universe correlation against the
                    Marchenko-Pastur bulk [lambda_-, lambda_+] (ADR-022)

Colours are the first three categorical slots of the validated reference palette, and every
series also has its own marker and a direct label, so identity never depends on colour alone.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nisq_folio.analysis.noise import analyze
from nisq_folio.analysis.summary import flatten
from nisq_folio.config import load_config
from nisq_folio.experiments.store import experiment_dir, load_results

OUT = Path(__file__).resolve().parent / "generated"
MIXERS = {  # name, colour (palette slots 1-3), marker
    "adapt": ("ADAPT", "#2a78d6", "o"),
    "xy_ring": ("XY ring", "#eb6834", "s"),
    "x": ("X", "#1baf7a", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "mathtext.fontset": "cm",
})  # fmt: skip


def _log_ticks(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def _reference_line(ax: plt.Axes) -> None:
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.annotate("random feasible guess (A = 1)", (0.99, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=6.5, color=MUTED)  # fmt: skip


def noiseless(qaoa: pd.DataFrame) -> plt.Figure:
    """fig_noiseless (module docstring)."""
    q = qaoa[(qaoa["noise_level"] == 0) & (qaoa["denoising"] == "none")]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    offsets = {"adapt": -0.15, "xy_ring": 0.0, "x": 0.15}
    for mixer, (label, colour, marker) in MIXERS.items():
        g = q[q["mixer"] == mixer].groupby("n")["A"]
        n = np.array(sorted(q["n"].unique()))
        med, lo, hi = g.median().loc[n], g.quantile(0.25).loc[n], g.quantile(0.75).loc[n]
        x = n + offsets[mixer]
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=colour, marker=marker, ms=4.5,
                    lw=1.5, capsize=0, elinewidth=1.2, mec="white", mew=0.6)  # fmt: skip
        ax.annotate(label, (x[-1], med.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)  # fmt: skip
    _reference_line(ax)
    _log_ticks(ax)
    ax.set_xticks(sorted(q["n"].unique()))
    ax.set_xlim(7.3, 15.6)
    ax.set_xlabel("Number of assets $N$ (qubits)")
    ax.set_ylabel("Amplification $A$ (median, IQR)")
    return fig


def _panels(curve: pd.DataFrame, column: str, ylabel: str, log: bool) -> plt.Figure:
    curve = curve[curve["denoising"] == "none"]
    sizes = sorted(curve["n"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.3 * len(sizes) + 0.4, 2.4), sharey=True,
                             squeeze=False)  # fmt: skip
    for ax, n in zip(axes[0], sizes, strict=True):
        for mixer, (label, colour, marker) in MIXERS.items():
            c = curve[(curve["n"] == n) & (curve["mixer"] == mixer)].sort_values("noise_level")
            if c.empty:
                continue
            if column == "A_median":
                ax.fill_between(c["noise_level"], c["A_ci_low"], c["A_ci_high"], color=colour,
                                alpha=0.15, lw=0)  # fmt: skip
            ax.plot(c["noise_level"], c[column], color=colour, marker=marker, ms=4.5, lw=1.5,
                    mec="white", mew=0.6, label=label)  # fmt: skip
        if column == "A_median":
            _reference_line(ax)
        ax.set_title(f"$N = {n}$", color=INK)
        ax.set_xlabel("Noise scale $m$ (1 = FakeKingston medians)")
        ax.set_xticks([0, 0.25, 0.5, 1.0])
        if log:
            _log_ticks(ax)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(loc="upper right", fontsize=7, handlelength=1.6)
    return fig


def spectrum(record: dict) -> plt.Figure:
    """fig_spectrum (module docstring)."""
    rep = record["results"]["reports"]["rmt_clip"]
    values = np.array(rep["eigenvalues"])
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    ax.axvspan(rep["lambda_minus"], rep["lambda_plus"], color=GRID, lw=0, zorder=0)
    kept = values > rep["lambda_plus"]
    ax.vlines(values, 0, 1, color=np.where(kept, MIXERS["adapt"][1], MUTED), lw=1.2)
    ax.annotate("MP bulk", ((rep["lambda_minus"] * rep["lambda_plus"]) ** 0.5, 1.13),
                ha="center", fontsize=6.5, color=MUTED)  # fmt: skip
    ax.annotate(f"{int(kept.sum())} kept as signal", (values[kept].min() * 1.15, 1.04),
                ha="left", fontsize=6.5, color=INK)  # fmt: skip
    ax.annotate(f"{int((~kept).sum())} averaged", (values.min(), 1.04), ha="left",
                fontsize=6.5, color=MUTED)  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Eigenvalue of the in-sample correlation ($q = N/T = {rep['q']:.3f}$)")
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    """Write every figure (module docstring); returns the exit code."""
    parser = argparse.ArgumentParser(prog="paper/make_figures.py")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    qaoa, classical = flatten(load_results(experiment_dir(config)))
    OUT.mkdir(exist_ok=True)
    figures = {"fig_noiseless": noiseless(qaoa)}
    curve = analyze(qaoa, classical, config.seed)["N1"]
    if not curve.empty:
        figures["fig_noise"] = _panels(curve, "A_median", "Amplification $A$ (median, 95% CI)",
                                       log=True)  # fmt: skip
        figures["fig_leakage"] = _panels(curve, "p_feasible_median",
                                         "Feasible probability (median)", log=False)  # fmt: skip
    records = sorted((config.results_dir / "denoise_evaluation").glob("*.json"))
    if records:
        figures["fig_spectrum"] = spectrum(json.loads(records[-1].read_text(encoding="utf-8")))
    for name, fig in figures.items():
        fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
        fig.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(fig)
    print(f"wrote {len(figures)} figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
