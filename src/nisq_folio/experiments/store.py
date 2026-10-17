"""Resumable on-disk storage of experiment results (DECISIONS.md ADR-024).

Layout under ``<results_dir>/experiments/<name>/``:

    manifest.json        config text + SHA-256 fingerprint, data snapshot, git state at start
    runs/<task_id>.json  one file per finished task, each with its own git state + fingerprint

Results are written atomically (temporary file, then ``os.replace``), so an interrupted run
never leaves a half-written result; a task counts as done exactly when its file exists. A
directory created under one config refuses to accept results from a different config, so a
single experiment can never silently mix settings.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nisq_folio.config import Config
from nisq_folio.data.prices import PriceMetadata
from nisq_folio.provenance import GitState


class ExperimentMismatchError(RuntimeError):
    """Raised when an experiment directory was created with a different config."""


def config_fingerprint(config: Config) -> str:
    """SHA-256 of the config file's exact text."""
    return hashlib.sha256(config.source_path.read_bytes()).hexdigest()


def experiment_dir(config: Config) -> Path:
    """``<results_dir>/experiments/<experiment.name>``."""
    return config.results_dir / "experiments" / config.require_experiment().name


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def open_experiment(config: Config, git: GitState, data: PriceMetadata) -> Path:
    """Create the experiment directory and manifest, or verify an existing one.

    Raises:
        ExperimentMismatchError: If the directory exists with a different config fingerprint.
    """
    root = experiment_dir(config)
    manifest = root / "manifest.json"
    fingerprint = config_fingerprint(config)
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing["config_sha256"] != fingerprint:
            raise ExperimentMismatchError(
                f"{root} was created with a different config (sha256 "
                f"{existing['config_sha256'][:12]} vs {fingerprint[:12]}); use a new "
                "experiment.name or delete the directory"
            )
        return root
    _atomic_write(
        manifest,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "config_path": str(config.source_path),
            "config_sha256": fingerprint,
            "config_text": config.source_path.read_text(encoding="utf-8"),
            "git": asdict(git),
            "data": asdict(data),
        },
    )
    return root


def completed_task_ids(root: Path) -> set[str]:
    """Ids of tasks whose result file exists (temporary files are ignored)."""
    runs = root / "runs"
    return {p.stem for p in runs.glob("*.json")} if runs.exists() else set()


def save_result(root: Path, record: dict[str, Any], git: GitState, fingerprint: str) -> Path:
    """Write one task's record with its provenance (CLAUDE.md rule 6); returns the path."""
    path = root / "runs" / f"{record['task_id']}.json"
    _atomic_write(
        path,
        {
            **record,
            "git": asdict(git),
            "config_sha256": fingerprint,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return path


def load_results(root: Path) -> list[dict[str, Any]]:
    """All finished task records, sorted by task id."""
    runs = root / "runs"
    if not runs.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs.glob("*.json"))]
