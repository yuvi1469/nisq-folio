"""Run the noiseless QAOA tuning study and apply the pre-registered selection rule (ADR-020).

For every N, draws tuning instances from a separate stream (excluding the evaluation
instances), runs shot-based X-mixer QAOA for every grid configuration, scores each final
distribution exactly (ADR-019), prints a summary, selects a configuration, and writes a run
record with every per-instance result.

Usage::

    uv run python -m nisq_folio.qaoa --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np

from nisq_folio.config import load_config
from nisq_folio.data.prices import PriceFetcher, fetch_yfinance, load_or_download
from nisq_folio.data.returns import simple_returns, split_returns
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.experiments.records import write_run_record
from nisq_folio.provenance import git_state
from nisq_folio.qaoa.metrics import SolutionLandscape
from nisq_folio.qaoa.optimize import QaoaSettings, run_noiseless_qaoa
from nisq_folio.qaoa.tuning import TuningConfiguration, configuration_score, select_configuration
from nisq_folio.qubo.penalty import sufficient_penalty
from nisq_folio.seeding import STREAM_QAOA_SHOTS, STREAM_TUNING_INSTANCES, child_rng


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the tuning study for a config file, print a summary, and write a run record.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source, used only if the data cache is missing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.qaoa", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problem_cfg, qaoa_cfg = config.require_problem(), config.require_qaoa()
    margin, tuning = config.require_qubo().penalty_margin, qaoa_cfg.tuning
    git = git_state(config.source_path.parent)
    u = config.universe
    prices, data_meta = load_or_download(
        u.tickers, u.start_date, u.end_date, config.data_dir, git=git, fetcher=fetcher
    )
    in_sample, _ = split_returns(simple_returns(prices), problem_cfg.in_sample_end)

    grid = [
        TuningConfiguration(p, c, d)
        for p, c, d in itertools.product(tuning.layers, tuning.penalty_scales, tuning.ramp_deltas)
    ]
    amplifications: dict[TuningConfiguration, dict[int, list[float]]] = {c: {} for c in grid}
    records: list[dict[str, Any]] = []
    for n in problem_cfg.sizes:
        evaluation = (
            sample_instance_tickers(u.tickers, n, tuning.excluded_evaluation_instances, config.seed)
            if tuning.excluded_evaluation_instances
            else []
        )
        instances = sample_instance_tickers(
            u.tickers,
            n,
            tuning.instances_per_size,
            config.seed,
            stream=STREAM_TUNING_INSTANCES,
            exclude=evaluation,
        )
        for index, tickers in enumerate(instances):
            problem = build_problem(in_sample, tickers, n // 2, problem_cfg.risk_aversion)
            landscape = SolutionLandscape.from_problem(problem)
            bound = sufficient_penalty(problem, margin)
            for g, cfg in enumerate(grid):
                settings = QaoaSettings(
                    layers=cfg.layers,
                    shots=qaoa_cfg.shots,
                    max_evaluations=qaoa_cfg.max_evaluations,
                    rhobeg=qaoa_cfg.rhobeg,
                    ramp_delta=cfg.ramp_delta,
                )
                rng = child_rng(config.seed, STREAM_QAOA_SHOTS, n, index, g)
                run = run_noiseless_qaoa(
                    problem, cfg.penalty_scale * bound, settings, rng, landscape
                )
                amplifications[cfg].setdefault(n, []).append(run.metrics.amplification_feasible)
                records.append({"n": n, "instance": index, "tickers": list(tickers),
                                "config": cfg.__dict__, **run.as_dict()})  # fmt: skip
        print(
            f"N={n}: {len(instances)} tuning instances x {len(grid)} configurations done",
            flush=True,
        )

    scores = {c: configuration_score(amplifications[c]) for c in grid}
    chosen = select_configuration(scores)
    sizes = problem_cfg.sizes
    print(
        f"\n{'configuration':<26}{'score':>7}"
        + "".join(f"{'A med N=' + str(n):>12}" for n in sizes)
    )
    for cfg in sorted(grid, key=lambda c: -scores[c]):
        meds = "".join(f"{np.median(amplifications[cfg][n]):>12.2f}" for n in sizes)
        mark = "  <- selected" if cfg == chosen else ""
        print(f"{cfg.label():<26}{scores[cfg]:>7.3f}{meds}{mark}")

    results = {
        "selected": chosen.__dict__,
        "scores": [{"config": c.__dict__, "score": scores[c]} for c in grid],
        "runs": records,
    }
    path = write_run_record(config, "qaoa_tuning", git, data_meta, results)
    print(f"\nselected: {chosen.label()}\nrun record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
