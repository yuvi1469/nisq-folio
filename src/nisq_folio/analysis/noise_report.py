"""Write the pre-registered noisy-grid report (DECISIONS.md ADR-028).

Usage::

    uv run python -m nisq_folio.analysis.noise_report --config configs/main.yaml --superseded main

Reads the configured experiment, keeps the cells that are complete at every noise level
(``analysis.noise.complete_cells``), prints tables N1-N7 and the ADR-026 verdict at every level,
and, with ``--superseded``, the N8 routing check against that older experiment directory. The
report is written to ``<results_dir>/analysis/<experiment>_noise.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from nisq_folio.analysis.noise import (
    analyze,
    complete_cells,
    routing_robust,
    routing_table,
    verdicts,
)
from nisq_folio.analysis.summary import flatten, markdown_table
from nisq_folio.config import load_config
from nisq_folio.experiments.store import experiment_dir, load_results

TITLES = {
    "N1": "N1. Amplification vs noise (median A, 95% bootstrap CI; feasibility leakage)",
    "V": "Verdicts (ADR-026 rule at each noise level)",
    "N2": "N2. Crossover m* where the median A falls to 1 (95% bootstrap CI)",
    "N3": "N3. Crossover comparisons: bootstrap P(first > second)",
    "N4": "N4. Decay: paired A(m) / A(0)",
    "N5": "N5. Mixers, paired, at each noise level",
    "N6": "N6. Denoising (rmt_clip / none), paired, at each noise level",
    "N7": "N7. Deployment at each noise level",
    "N8": "N8. Routing: superseded full-device / compact-region record, paired",
    "N8c": "N8. Routing: superseded crossover inside the record's 95% CI?",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Write the noisy report (module docstring); returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.analysis.noise_report")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--superseded", help="older experiment name for the routing check")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    name = config.require_experiment().name
    qaoa, classical = flatten(load_results(experiment_dir(config)))
    if qaoa.empty or complete_cells(qaoa).empty:
        print(f"{name}: no cell is complete at every noise level yet")
        return 1
    tables = analyze(qaoa, classical, config.seed)
    tables = {"N1": tables["N1"], "V": verdicts(tables["N1"]), **tables}
    if args.superseded:
        old, _ = flatten(load_results(config.results_dir / "experiments" / args.superseded))
        record = complete_cells(qaoa)
        tables["N8"] = routing_table(record, old)
        sizes = set(tables["N8"]["n"]) if not tables["N8"].empty else set()
        old_cross = analyze(old[old["n"].isin(sizes)], classical, config.seed)["N2"]
        if not old_cross.empty:
            tables["N8c"] = routing_robust(tables["N2"], old_cross)

    done = complete_cells(qaoa)
    lines = [
        f"# {name}: noisy analysis (ADR-028)",
        "",
        f"Complete cells: {done.groupby(['denoising', 'mixer', 'n']).ngroups}; "
        f"{len(done)} QAOA tasks; sizes {sorted(int(n) for n in done['n'].unique())}.",
        "",
    ]
    with pd.option_context("display.width", 180, "display.max_columns", 20, "display.precision", 3):
        for key, table in tables.items():
            print(f"\n{TITLES[key]}\n{table.to_string(index=False)}")
            lines += [f"## {TITLES[key]}", "", markdown_table(table), ""]
    out = config.results_dir / "analysis" / f"{name}_noise.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
