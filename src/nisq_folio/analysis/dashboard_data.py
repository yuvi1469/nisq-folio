"""Data preparation for the read-only results dashboard (``dashboard/app.py``).

Everything here is a pure function of the experiment directory, so the dashboard can never
change a result, and every number it shows comes from the same code as the reports
(``analysis.summary``, ``analysis.noise``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nisq_folio.analysis.noise import analyze, verdicts
from nisq_folio.analysis.summary import flatten
from nisq_folio.config import Config
from nisq_folio.experiments.grid import expand_tasks
from nisq_folio.experiments.store import completed_task_ids, experiment_dir, load_results


@dataclass(frozen=True)
class DashboardData:
    """Everything the dashboard displays for one experiment.

    Attributes:
        progress: Done / total tasks per (kind, N, mixer, noise level).
        qaoa: One row per finished QAOA task (``analysis.summary.flatten``).
        classical: One row per finished classical task.
        noisy: Tables N1-N7 of ADR-028 plus "V" (verdicts); empty until a cell is complete.
    """

    progress: pd.DataFrame
    qaoa: pd.DataFrame
    classical: pd.DataFrame
    noisy: dict[str, pd.DataFrame]


def progress_table(config: Config, root: Path | None = None) -> pd.DataFrame:
    """Done and total task counts per (kind, N, mixer, noise level) of the configured grid."""
    root = root or experiment_dir(config)
    done = completed_task_ids(root)
    rows = [
        {
            "kind": t.kind,
            "n": t.n,
            "mixer": t.mixer or "-",
            "noise_level": t.noise_level,
            "done": t.task_id in done,
        }
        for t in expand_tasks(config.require_experiment())
    ]
    table = (
        pd.DataFrame(rows)
        .groupby(["kind", "n", "mixer", "noise_level"], as_index=False)
        .agg(done=("done", "sum"), total=("done", "size"))
    )
    table["share"] = table["done"] / table["total"]
    return table


def load_dashboard(config: Config, replicates: int = 500) -> DashboardData:
    """Read the experiment of ``config`` and compute every dashboard table.

    Args:
        config: A config with an ``experiment`` section.
        replicates: Bootstrap replicates for the noisy tables. The dashboard uses fewer than
            the 2,000 of record (ADR-028) to stay responsive; the reports use the full number.

    Returns:
        The tables, possibly empty while the grid is still running.
    """
    root = experiment_dir(config)
    qaoa, classical = flatten(load_results(root))
    noisy: dict[str, pd.DataFrame] = {}
    if not qaoa.empty:
        noisy = analyze(qaoa, classical, config.seed, replicates)
        if not noisy["N1"].empty:
            noisy["V"] = verdicts(noisy["N1"])
    return DashboardData(progress_table(config, root), qaoa, classical, noisy)
