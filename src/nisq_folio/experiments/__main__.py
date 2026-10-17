"""Run (or estimate) the factorial experiment; resumable (DECISIONS.md ADR-024).

Usage::

    uv run python -m nisq_folio.experiments --config configs/base.yaml --estimate
    uv run python -m nisq_folio.experiments --config configs/base.yaml [--workers 3]
        [--max-noise 0] [--limit 50]

``--estimate`` prints predicted hours per group and exits without running anything.
Re-running the same command skips every task that already has a result file.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from nisq_folio.config import load_config
from nisq_folio.experiments.cost import summarize_cost, task_seconds
from nisq_folio.experiments.execute import ExperimentContext, run_task
from nisq_folio.experiments.grid import expand_tasks
from nisq_folio.experiments.parallel import worker_init, worker_run
from nisq_folio.experiments.store import (
    completed_task_ids,
    config_fingerprint,
    open_experiment,
    save_result,
)
from nisq_folio.provenance import git_state


def main(argv: Sequence[str] | None = None) -> int:
    """Estimate or run the experiment for a config file.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.experiments", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--estimate", action="store_true", help="print predicted hours and exit")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    parser.add_argument(
        "--max-noise", type=float, default=None, help="only run tasks with noise level <= this"
    )
    parser.add_argument("--limit", type=int, default=None, help="run at most this many tasks")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    config = load_config(args.config)
    tasks = expand_tasks(config.require_experiment())
    if args.max_noise is not None:
        tasks = [t for t in tasks if t.noise_level <= args.max_noise]

    git = git_state(config.source_path.parent)
    if args.estimate:
        groups = summarize_cost(tasks, config)
        known = [g["hours"] for g in groups.values() if g["hours"] is not None]
        print(f"{'group':<32}{'tasks':>7}{'hours':>10}")
        for key, g in groups.items():
            hours = "unknown" if g["hours"] is None else f"{g['hours']:.2f}"
            print(f"{key:<32}{g['tasks']:>7}{hours:>10}")
        print(f"{'total (known groups)':<32}{len(tasks):>7}{sum(known):>10.2f}")
        return 0

    ctx = ExperimentContext.build(config, git)
    root = open_experiment(config, git, ctx.data)
    fingerprint = config_fingerprint(config)
    done = completed_task_ids(root)
    todo = [t for t in tasks if t.task_id not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{root}: {len(tasks)} tasks, {len(done)} already done, running {len(todo)}"
        f" with {args.workers} worker(s)",
        flush=True,
    )
    if git.dirty:
        print("warning: uncommitted changes; results will be marked git.dirty=true", flush=True)

    start = time.perf_counter()
    predicted = sum(task_seconds(t, config) or 0.0 for t in todo)

    def report(i: int, record: dict[str, Any]) -> None:
        save_result(root, record, git, fingerprint)
        elapsed = time.perf_counter() - start
        print(
            f"[{i}/{len(todo)}] {record['task_id']} ({record['wall_seconds']:.1f}s; "
            f"elapsed {elapsed / 60:.1f} min, predicted total {predicted / 60:.1f} min)",
            flush=True,
        )

    if args.workers == 1:
        for i, task in enumerate(todo, 1):
            report(i, run_task(task, ctx))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=worker_init, initargs=(str(config.source_path),)
        ) as pool:
            futures = [pool.submit(worker_run, t) for t in todo]
            for i, future in enumerate(as_completed(futures), 1):
                report(i, future.result())
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
