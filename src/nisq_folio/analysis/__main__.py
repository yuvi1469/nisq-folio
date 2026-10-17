"""Write the pre-registered summary tables of an experiment (DECISIONS.md ADR-026).

Usage::

    uv run python -m nisq_folio.analysis --config configs/main.yaml --max-noise 0

Reads every finished task of the configured experiment, keeps noise levels <= ``--max-noise``,
prints tables T1-T6 (see ``analysis/summary.py``) and writes them as Markdown to
``<results_dir>/analysis/<experiment>_m<max-noise>.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from nisq_folio.analysis.summary import (
    budget_table,
    cell_table,
    classical_table,
    denoising_table,
    deployment_table,
    flatten,
    markdown_table,
    mixer_table,
)
from nisq_folio.config import load_config
from nisq_folio.experiments.store import experiment_dir, load_results

TITLES = {
    "T1": "T1. Cells: amplification over random feasible guessing",
    "T2": "T2. Mixers, paired (geometric-mean A ratio, denoising = none)",
    "T3": "T3. Denoising, paired (A ratio rmt_clip / none)",
    "T4": "T4. Classical baselines: share of instances solved optimally",
    "T5": "T5. Budget: QAOA total shots vs SA steps for 90% optimal",
    "T6": "T6. Deployment: modal selection's out-of-sample variance / optimum's",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Summarize an experiment (module docstring); returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.analysis", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--max-noise", type=float, default=0.0, help="highest noise level kept")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = experiment_dir(config)
    qaoa, classical = flatten(load_results(root))
    if qaoa.empty or classical.empty:
        print(f"{root}: no finished QAOA and classical tasks yet")
        return 1
    qaoa = qaoa[qaoa["noise_level"] <= args.max_noise]
    tables = {
        "T1": cell_table(qaoa),
        "T2": mixer_table(qaoa),
        "T3": denoising_table(qaoa),
        "T4": classical_table(classical),
        "T5": budget_table(qaoa, classical),
        "T6": deployment_table(qaoa, classical),
    }
    lines = [
        f"# {config.require_experiment().name}: noise levels <= {args.max_noise:g}",
        "",
        f"{len(qaoa)} QAOA tasks, {len(classical)} classical tasks.",
        "",
    ]
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 3):
        for key, table in tables.items():
            print(f"\n{TITLES[key]}\n{table.to_string(index=False)}")
            lines += [f"## {TITLES[key]}", "", markdown_table(table), ""]
    out = (
        config.results_dir
        / "analysis"
        / f"{config.require_experiment().name}_m{args.max_noise:g}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
