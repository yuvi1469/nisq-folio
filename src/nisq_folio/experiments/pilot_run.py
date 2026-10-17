"""Run the noisy-grid pilot and apply the pre-registered rules (DECISIONS.md ADR-025).

Usage::

    uv run python -m nisq_folio.experiments.pilot_run --config configs/base.yaml

Prints timings, the pilot A-vs-noise curve at N = ``accuracy_size``, and the noise levels,
worker count and hours that Rules 1-3 in ``experiments/pilot.py`` produce; writes a run record.
Nothing is changed in the config automatically: the chosen levels are written into ADR-025 and
``configs/base.yaml`` by hand after review.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np

from nisq_folio.config import load_config
from nisq_folio.experiments.cost import evaluations_per_run
from nisq_folio.experiments.execute import ExperimentContext, variant_settings
from nisq_folio.experiments.grid import Task, expand_tasks
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.experiments.pilot import (
    choose_levels,
    interpolated_crossing,
    time_evaluation,
    upper_noise_level,
)
from nisq_folio.experiments.records import write_run_record
from nisq_folio.mixers.variants import run_variant
from nisq_folio.noise.density import ReferenceRates
from nisq_folio.noise.runner import run_noisy_variant
from nisq_folio.provenance import git_state
from nisq_folio.qaoa.metrics import SolutionLandscape
from nisq_folio.qubo.penalty import sufficient_penalty
from nisq_folio.seeding import STREAM_PILOT, STREAM_TUNING_INSTANCES, child_rng


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot for a config file (module docstring); returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.experiments.pilot_run")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pilot, experiment = config.require_pilot(), config.require_experiment()
    problem_cfg = config.require_problem()
    git = git_state(config.source_path.parent)
    ctx = ExperimentContext.build(config, git)
    rates = ReferenceRates.from_backend(experiment.backend)
    u = config.universe
    excluded = max(
        experiment.instances_per_size, config.require_qaoa().tuning.excluded_evaluation_instances
    )

    def tuning_problems(n: int, count: int) -> list:
        evaluation = sample_instance_tickers(u.tickers, n, excluded, config.seed)
        tickers = sample_instance_tickers(
            u.tickers, n, count, config.seed, stream=STREAM_TUNING_INSTANCES, exclude=evaluation
        )
        return [build_problem(ctx.in_sample, t, n // 2, problem_cfg.risk_aversion) for t in tickers]

    # 1. timing ---------------------------------------------------------------------------
    timing: dict[str, dict[str, Any]] = {}
    print("seconds per noisy evaluation at m = 1 (auto threads | 1 thread):", flush=True)
    for n in pilot.timing_sizes:
        problem = tuning_problems(n, 1)[0]
        for mixer in experiment.mixers:
            if n in pilot.skip_timing.get(mixer, ()):
                continue
            settings = variant_settings(config, mixer)
            auto = time_evaluation(
                problem, settings, 1.0, rates, pilot.timing_repeats, 0, experiment.backend
            )
            single = (
                time_evaluation(
                    problem, settings, 1.0, rates, pilot.timing_repeats, 1, experiment.backend
                )
                if n in pilot.thread_test_sizes
                else None
            )
            timing.setdefault(mixer, {})[str(n)] = {"auto": auto, "one_thread": single}
            extra = f" | {single:7.2f} s" if single is not None else ""
            print(f"  N={n:2d} {mixer:<8} {auto:7.2f} s{extra}", flush=True)

    # Rule 3: parallelism
    gains = [
        3 * t["auto"] / t["one_thread"]
        for m in timing.values()
        for t in m.values()
        if t["one_thread"]
    ]
    workers = 3 if gains and min(gains) > 1.2 else 1
    print(
        f"3-worker throughput gain at {pilot.thread_test_sizes}: "
        f"{', '.join(f'{g:.2f}' for g in gains)} -> workers = {workers}",
        flush=True,
    )

    # 2. bracketing ------------------------------------------------------------------------
    n = pilot.accuracy_size
    problems = tuning_problems(n, pilot.accuracy_instances)
    curves: dict[str, dict[str, list[float]]] = {}
    print(f"\npilot A at N={n} ({len(problems)} tuning instances):", flush=True)
    for mixer in experiment.mixers:
        settings = variant_settings(config, mixer)
        curves[mixer] = {}
        for level in (0.0, *pilot.accuracy_noise_levels):
            values = []
            for i, problem in enumerate(problems):
                bound = sufficient_penalty(problem, config.require_qubo().penalty_margin)
                land = SolutionLandscape.from_problem(problem)
                rng = child_rng(
                    config.seed,
                    STREAM_PILOT,
                    n,
                    i,
                    round(level * 1e6),
                    experiment.mixers.index(mixer),
                )
                if level == 0.0:
                    run = run_variant(problem, bound, settings, rng, land)
                else:
                    run = run_noisy_variant(
                        problem,
                        bound,
                        settings,
                        level,
                        rates,
                        rng,
                        land,
                        backend=experiment.backend,
                    )
                values.append(run.metrics.amplification_feasible)
            curves[mixer][f"{level:g}"] = values
        medians = [np.median(curves[mixer][f"{m:g}"]) for m in pilot.accuracy_noise_levels]
        print(
            f"  {mixer:<8} m=0: {np.median(curves[mixer]['0']):.2f} | "
            + " | ".join(
                f"m={m:g}: {a:.2f}"
                for m, a in zip(pilot.accuracy_noise_levels, medians, strict=True)
            ),
            flush=True,
        )

    # Rule 1: upper level, Rule 2: levels within budget -------------------------------------
    crossings = {
        mixer: interpolated_crossing(
            pilot.accuracy_noise_levels,
            [float(np.median(curves[mixer][f"{m:g}"])) for m in pilot.accuracy_noise_levels],
        )
        for mixer in experiment.mixers
    }
    m_max = upper_noise_level(crossings, pilot.noise_ladder)

    def seconds_per_eval(task: Task) -> float:
        entry = timing[task.mixer or ""][str(task.n)]
        if workers > 1 and entry["one_thread"] is not None:
            return entry["one_thread"] / workers  # W single-thread workers in parallel
        return entry["auto"]  # no assumed speedup where not measured (Rule 3)

    def hours_for(levels: list[float]) -> float:
        grid = expand_tasks(_with_levels(experiment, levels))
        total = 0.0
        for task in grid:
            if task.kind != "qaoa" or task.noise_level == 0.0:
                continue
            layers = experiment.mixer_settings[task.mixer or ""].layers
            evals = evaluations_per_run(
                task.mixer or "", task.n, layers, config.require_qaoa().max_evaluations
            )
            total += evals * seconds_per_eval(task)
        return total / 3600

    levels, hours, within = choose_levels(
        pilot.noise_ladder, m_max, hours_for, pilot.budget_hours, pilot.min_noisy_levels
    )
    print(
        "\ncrossings (m where median A = 1): "
        + ", ".join(f"{k}: {'none' if v is None else f'{v:.2f}'}" for k, v in crossings.items())
    )
    print(f"Rule 1: m_max = {m_max:g}")
    print(
        f"Rule 2: noisy levels = {levels}, estimated {hours:.1f} h "
        f"({'within' if within else 'OVER'} the {pilot.budget_hours:g} h budget), "
        f"workers = {workers}"
    )

    results = {
        "timing": timing,
        "throughput_gains": gains,
        "workers": workers,
        "curves": curves,
        "crossings": crossings,
        "m_max": m_max,
        "noisy_levels": levels,
        "estimated_hours": hours,
        "within_budget": within,
    }
    path = write_run_record(config, "pilot", git, ctx.data, results)
    print(f"run record: {path}")
    return 0


def _with_levels(experiment: Any, levels: list[float]) -> Any:  # noqa: ANN401
    from dataclasses import replace

    return replace(experiment, noise_levels=(0.0, *levels))


if __name__ == "__main__":
    raise SystemExit(main())
