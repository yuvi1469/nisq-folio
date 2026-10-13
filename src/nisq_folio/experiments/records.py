"""Write self-describing run records (CLAUDE.md rule 6).

Every run record is one JSON file containing the full config text, the git state, the data
snapshot's provenance, and the run's results, so it can be traced back to exact inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nisq_folio.config import Config
from nisq_folio.data.prices import PriceMetadata
from nisq_folio.provenance import GitState


def write_run_record(
    config: Config,
    run_name: str,
    git: GitState,
    data: PriceMetadata,
    results: dict[str, Any],
    now: datetime | None = None,
) -> Path:
    """Write a run record to ``<results_dir>/<run_name>/<timestamp>_<commit>.json``.

    Args:
        config: The loaded config; its source file's text is embedded verbatim.
        run_name: Subdirectory name, e.g. ``"baselines_validation"``.
        git: Git state of the code that produced the results.
        data: Provenance of the price snapshot used.
        results: JSON-serializable results.
        now: Timestamp to use (defaults to the current UTC time).

    Returns:
        Path of the written file.
    """
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    commit = (git.commit or "nogit")[:7] + ("-dirty" if git.dirty else "")
    out_dir = config.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}_{commit}.json"
    record = {
        "run_name": run_name,
        "created_at": now.isoformat(timespec="seconds"),
        "git": asdict(git),
        "config_path": str(config.source_path),
        "config_text": config.source_path.read_text(encoding="utf-8"),
        "data": asdict(data),
        "results": results,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
