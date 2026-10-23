"""Run the experiment as a Windows scheduled task (not part of the research code).

Meant to be started by Task Scheduler with ``pythonw.exe`` (no console window)::

    .venv/Scripts/pythonw.exe tools/run_scheduled.py --config configs/main.yaml \
        --log-dir results/experiments/main_r2

It appends the runner's output to ``<log-dir>/run_stdout.log`` and ``run_stderr.log``, keeps
Windows from idle standby while it runs (the same request as ``tools/keep_awake.py``, from a
background thread), and exits with the runner's code. An unexpected error exits with 1, so the
task's restart policy starts it again; the runner resumes where it stopped (ADR-024).
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _keep_awake(interval_seconds: float = 60.0) -> None:
    """Re-assert the system-required request forever (released when the process exits)."""
    while True:
        ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        time.sleep(interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment with file logging and keep-awake; returns the exit code."""
    parser = argparse.ArgumentParser(prog="tools/run_scheduled.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(args.log_dir / "run_stdout.log", "a", encoding="utf-8", buffering=1)
    sys.stderr = open(args.log_dir / "run_stderr.log", "a", encoding="utf-8", buffering=1)
    print(f"{datetime.now():%d %b %H:%M:%S} scheduled run starting", flush=True)
    if sys.platform == "win32":
        threading.Thread(target=_keep_awake, daemon=True).start()
    from nisq_folio.experiments.__main__ import main as run_experiment

    try:
        code = run_experiment(["--config", args.config])
    except BaseException:
        traceback.print_exc()
        code = 1
    print(f"{datetime.now():%d %b %H:%M:%S} scheduled run exiting with code {code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
