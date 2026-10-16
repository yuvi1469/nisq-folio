"""Run the noiseless mixer study (ADR-021) on the QAOA tuning instances and write a run record.

Uses the same tuning instances as ADR-020 (separate stream, evaluation instances excluded).
For every mixer x optimizer x depth (x penalty scale, X mixer only), runs noiseless QAOA, scores
the final distribution exactly, and applies the pre-registered per-mixer selection rule.

Usage::

    uv run python -m nisq_folio.mixers --config configs/base.yaml
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
from nisq_folio.mixers.study import MixerConfiguration, select_per_mixer
from nisq_folio.mixers.variants import VariantSettings, run_variant
from nisq_folio.provenance import git_state
from nisq_folio.qaoa.metrics import SolutionLandscape
from nisq_folio.qaoa.optimizers import SpsaSettings
from nisq_folio.qaoa.tuning import configuration_score
from nisq_folio.qubo.penalty import sufficient_penalty
from nisq_folio.seeding import STREAM_MIXER_STUDY, STREAM_TUNING_INSTANCES, child_rng


def build_grid(
    mixers: Sequence[str],
    optimizers: Sequence[str],
    layers: Sequence[int],
    x_penalty_scales: Sequence[float],
) -> list[MixerConfiguration]:
    """Return the study grid; penalty scales vary for the X mixer only (others use 1.0, unused)."""
    grid = []
    for mixer, optimizer, p in itertools.product(mixers, optimizers, layers):
        scales = x_penalty_scales if mixer == "x" else (1.0,)
        grid += [MixerConfiguration(mixer, optimizer, p, c) for c in scales]
    return grid


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the mixer study for a config file, print a summary, and write a run record.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source, used only if the data cache is missing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.mixers", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problem_cfg, qaoa_cfg = config.require_problem(), config.require_qaoa()
    study = config.require_mixer_study()
    margin, tuning = config.require_qubo().penalty_margin, qaoa_cfg.tuning
    git = git_state(config.source_path.parent)
    u = config.universe
    prices, data_meta = load_or_download(
        u.tickers, u.start_date, u.end_date, config.data_dir, git=git, fetcher=fetcher
    )
    in_sample, _ = split_returns(simple_returns(prices), problem_cfg.in_sample_end)
    spsa = SpsaSettings(**study.spsa)
    grid = build_grid(study.mixers, study.optimizers, study.layers, study.x_penalty_scales)

    amplifications: dict[MixerConfiguration, dict[int, list[float]]] = {c: {} for c in grid}
    records: list[dict[str, Any]] = []
    for n in problem_cfg.sizes:
        evaluation = (
            sample_instance_tickers(u.tickers, n, tuning.excluded_evaluation_instances, config.seed)
            if tuning.excluded_evaluation_instances
            else []
        )
        instances = sample_instance_tickers(
            u.tickers, n, tuning.instances_per_size, config.seed,
            stream=STREAM_TUNING_INSTANCES, exclude=evaluation,
        )  # fmt: skip
        for index, tickers in enumerate(instances):
            problem = build_problem(in_sample, tickers, n // 2, problem_cfg.risk_aversion)
            landscape = SolutionLandscape.from_problem(problem)
            bound = sufficient_penalty(problem, margin)
            for g, cfg in enumerate(grid):
                settings = VariantSettings(
                    mixer=cfg.mixer,
                    optimizer=cfg.optimizer,
                    layers=cfg.layers,
                    shots=qaoa_cfg.shots,
                    max_evaluations=qaoa_cfg.max_evaluations,
                    rhobeg=qaoa_cfg.rhobeg,
                    ramp_delta=study.ramp_delta,
                    penalty_scale=cfg.penalty_scale,
                    spsa=spsa,
                    adapt_gradient_shift=study.adapt_gradient_shift,
                )
                rng = child_rng(config.seed, STREAM_MIXER_STUDY, n, index, g)
                run = run_variant(problem, bound, settings, rng, landscape)
                amplifications[cfg].setdefault(n, []).append(run.metrics.amplification_feasible)
                records.append({"n": n, "instance": index, "tickers": list(tickers),
                                "config": cfg.__dict__, **run.as_dict()})  # fmt: skip
        print(f"N={n}: {len(instances)} instances x {len(grid)} configurations done", flush=True)

    scores = {c: configuration_score(amplifications[c]) for c in grid}
    selected = select_per_mixer(scores)
    sizes = problem_cfg.sizes
    header = f"{'configuration':<28}{'score':>7}" + "".join(f"{'A N=' + str(n):>9}" for n in sizes)
    print("\n" + header)
    for cfg in sorted(grid, key=lambda c: (c.mixer, -scores[c])):
        meds = "".join(f"{np.median(amplifications[cfg][n]):>9.2f}" for n in sizes)
        mark = "  <- selected" if selected[cfg.mixer] == cfg else ""
        print(f"{cfg.label():<28}{scores[cfg]:>7.3f}{meds}{mark}")

    results = {
        "selected": {m: c.__dict__ for m, c in selected.items()},
        "scores": [{"config": c.__dict__, "score": scores[c]} for c in grid],
        "runs": records,
    }
    path = write_run_record(config, "mixer_study", git, data_meta, results)
    print("\nselected: " + "; ".join(c.label() for c in selected.values()))
    print(f"run record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
