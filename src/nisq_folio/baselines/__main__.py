"""Validate the classical baselines against the exact brute-force optimum on real data.

For each problem size N in the config, draws the configured number of seeded instances from
the universe, builds each problem from in-sample returns, and runs every baseline. Reports,
per solver and N, how often the optimum is found and the normalized optimality gap

    gap(x) = (f(x) - f*) / (f_worst - f*)      in [0, 1]

where f* and f_worst are the best and worst objective over all K-subsets. Writes a full run
record (config, git commit, data checksum, per-instance results) under ``results/``.

Usage::

    uv run python -m nisq_folio.baselines --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from nisq_folio.baselines.brute_force import enumerate_objectives
from nisq_folio.baselines.greedy import greedy
from nisq_folio.baselines.markowitz import markowitz_rounding
from nisq_folio.baselines.simulated_annealing import AnnealingSettings, simulated_annealing
from nisq_folio.config import load_config
from nisq_folio.data.prices import PriceFetcher, fetch_yfinance, load_or_download
from nisq_folio.data.returns import simple_returns, split_returns
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.experiments.records import write_run_record
from nisq_folio.problem import PortfolioProblem, Solution
from nisq_folio.provenance import git_state
from nisq_folio.seeding import STREAM_SIMULATED_ANNEALING, child_rng

SOLVERS = ("greedy", "markowitz_rounding", "simulated_annealing")


def evaluate_instance(
    problem: PortfolioProblem, solvers: dict[str, Callable[[PortfolioProblem], Solution]]
) -> dict[str, Any]:
    """Run the solvers on one problem and score each against the brute-force optimum.

    Args:
        problem: The instance.
        solvers: Name -> solver function.

    Returns:
        A JSON-serializable record with the optimum, worst value, and per-solver scores
        (objective, normalized gap, rank among all subsets, optimal flag, evaluations).
    """
    subsets, values = enumerate_objectives(problem)
    best, worst = float(values.min()), float(values.max())
    span = worst - best
    tolerance = 1e-12 * max(1.0, abs(best))
    record: dict[str, Any] = {
        "tickers": list(problem.asset_names),
        "optimum": {
            "selection": [int(i) for i in subsets[int(np.argmin(values))]],
            "objective": best,
        },
        "worst_objective": worst,
        "n_subsets": len(values),
        "solvers": {},
    }
    for name, solve in solvers.items():
        sol = solve(problem)
        optimal = bool(sol.objective <= best + tolerance)
        # An optimal selection's gap is exactly 0; without this, the two summation orders
        # (per-subset vs full vector) can disagree by ~1e-17 and print as "-0.0000".
        gap = 0.0 if optimal or span <= 0 else (sol.objective - best) / span
        record["solvers"][name] = {
            "selection": list(sol.selection),
            "objective": sol.objective,
            "gap": gap,
            "rank": int((values < sol.objective - tolerance).sum()) + 1,
            "optimal": optimal,
            "n_evaluations": sol.n_evaluations,
        }
    return record


def summarize(records: Sequence[dict[str, Any]], solver: str) -> dict[str, float]:
    """Aggregate one solver's scores over instances: hit rate, mean/max gap, median rank."""
    scores = [r["solvers"][solver] for r in records]
    gaps = np.array([s["gap"] for s in scores])
    return {
        "optimal_rate": float(np.mean([s["optimal"] for s in scores])),
        "mean_gap": float(gaps.mean()),
        "max_gap": float(gaps.max()),
        "median_rank": float(np.median([s["rank"] for s in scores])),
    }


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the validation for a config file, print a summary table, and write a run record.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source, used only if the data cache is missing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.baselines", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problem_cfg, baseline_cfg = config.require_problem(), config.require_baselines()
    annealing = AnnealingSettings(**baseline_cfg.simulated_annealing)
    git = git_state(config.source_path.parent)
    u = config.universe
    prices, data_meta = load_or_download(
        u.tickers, u.start_date, u.end_date, config.data_dir, git=git, fetcher=fetcher
    )
    in_sample, _ = split_returns(simple_returns(prices), problem_cfg.in_sample_end)

    results: dict[str, Any] = {"sizes": {}}
    header = f"{'N':>3} {'K':>3}  {'solver':<20}{'optimal':>9}{'mean gap':>10}{'max gap':>9}"
    print(header + f"{'med rank':>9}")
    for n in problem_cfg.sizes:
        k = n // 2
        records = []
        instances = sample_instance_tickers(
            u.tickers, n, baseline_cfg.validation_instances_per_size, config.seed
        )
        for index, tickers in enumerate(instances):
            problem = build_problem(in_sample, tickers, k, problem_cfg.risk_aversion)
            rng = child_rng(config.seed, STREAM_SIMULATED_ANNEALING, n, index)
            solvers = {
                "greedy": greedy,
                "markowitz_rounding": markowitz_rounding,
                "simulated_annealing": lambda p, rng=rng: simulated_annealing(p, rng, annealing),
            }
            records.append(evaluate_instance(problem, solvers))
        summary = {name: summarize(records, name) for name in SOLVERS}
        results["sizes"][str(n)] = {"k": k, "summary": summary, "instances": records}
        for name in SOLVERS:
            s = summary[name]
            print(
                f"{n:>3} {k:>3}  {name:<20}{s['optimal_rate']:>9.0%}{s['mean_gap']:>10.4f}"
                f"{s['max_gap']:>9.4f}{s['median_rank']:>9.1f}"
            )

    path = write_run_record(config, "baselines_validation", git, data_meta, results)
    print(f"\nrun record: {path}")
    if git.dirty:
        print("warning: working tree has uncommitted changes; the commit hash alone does not")
        print("         identify the code that produced this record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
