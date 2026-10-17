"""Worker-process entry points for parallel experiment runs (DECISIONS.md ADR-024).

These must live in an importable module, not in ``__main__``: on Windows, worker processes are
spawned fresh and unpickle functions by module path, and functions defined in a module run with
``python -m`` are not importable from the workers (that fails with ``Can't get attribute``).
"""

from __future__ import annotations

from typing import Any

from nisq_folio.config import load_config
from nisq_folio.experiments.execute import ExperimentContext, run_task
from nisq_folio.experiments.grid import Task
from nisq_folio.provenance import git_state

_WORKER: dict[str, Any] = {}


def worker_init(config_path: str) -> None:
    """Build the per-process context once (Aer threads capped at 1 to avoid oversubscription)."""
    config = load_config(config_path)
    _WORKER["ctx"] = ExperimentContext.build(
        config, git_state(config.source_path.parent), max_parallel_threads=1
    )


def worker_run(task: Task) -> dict[str, Any]:
    """Run one task in a worker process."""
    return run_task(task, _WORKER["ctx"])
