"""Predicted wall time of experiment tasks, before running them (DECISIONS.md ADR-024).

Circuit evaluations per QAOA run (budget E = ``qaoa.max_evaluations``):
    x        about 0.45 E   (COBYLA stopped early, median 42-47 of 100, ADR-020/021)
    xy_ring  E               (SPSA spends exactly its budget)
    adapt    E + 2 |pool| p + 1, |pool| = 1 + C(N, 2)   (operator selection, ADR-021)
plus one final evaluation. Seconds per evaluation come from ``experiment.cost_model``
(measured in ADR-023 at m = 1; density-matrix cost does not depend on m). Estimates are rough:
this laptop's speed varied by up to 60x between identical runs (ADR-023), so treat them as
order-of-magnitude guidance, to be replaced by the W10 pilot's measurements.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

from nisq_folio.config import Config
from nisq_folio.experiments.grid import Task

CLASSICAL_SECONDS = 1.0
COBYLA_SHARE = 0.45


def evaluations_per_run(mixer: str, n: int, layers: int, budget: int) -> int:
    """Expected circuit evaluations of one QAOA run (module docstring)."""
    if mixer == "x":
        return round(COBYLA_SHARE * budget) + 1
    if mixer == "xy_ring":
        return budget + 1
    if mixer == "adapt":
        return budget + 2 * (1 + comb(n, 2)) * layers + 1
    raise ValueError(f"unknown mixer {mixer!r}")


def task_seconds(task: Task, config: Config) -> float | None:
    """Predicted seconds for one task, or ``None`` if the cost model has no entry for it."""
    experiment = config.require_experiment()
    if task.kind == "classical":
        return CLASSICAL_SECONDS
    assert task.mixer is not None
    layers = experiment.mixer_settings[task.mixer].layers
    evals = evaluations_per_run(task.mixer, task.n, layers, config.require_qaoa().max_evaluations)
    if task.noise_level == 0.0:
        return evals * experiment.noiseless_seconds_per_eval
    per_eval = experiment.noisy_seconds_per_eval.get(task.mixer, {}).get(task.n)
    return None if per_eval is None else evals * per_eval


def summarize_cost(tasks: list[Task], config: Config) -> dict[str, dict[str, float | int | None]]:
    """Group predicted hours by (mixer or 'classical', N, m); ``None`` hours = unknown cost."""
    groups: dict[str, dict[str, float | int | None]] = defaultdict(
        lambda: {"tasks": 0, "hours": 0.0}
    )
    for task in tasks:
        key = f"{task.mixer or 'classical'} N={task.n} m={task.noise_level:g}"
        seconds = task_seconds(task, config)
        group = groups[key]
        group["tasks"] = int(group["tasks"] or 0) + 1
        if seconds is None or group["hours"] is None:
            group["hours"] = None
        else:
            group["hours"] = float(group["hours"]) + seconds / 3600
    return dict(groups)
