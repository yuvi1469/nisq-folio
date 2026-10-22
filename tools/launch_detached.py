"""Start a long-running command fully detached on Windows (not part of the research code).

The child gets no console (``DETACHED_PROCESS``), its own process group, and leaves the
launcher's job object when allowed, so console Ctrl+C / Ctrl+Break events, job limits, and the
closing of the terminal or app that started it cannot stop it. Its
stdout and stderr are appended to the given log files. The launcher prints the child's PID and
exits immediately.

Usage (Windows only)::

    uv run python tools/launch_detached.py --stdout out.log --stderr err.log -- <command...>

for example with the command
``.venv/Scripts/python.exe -u -m nisq_folio.experiments --config configs/main.yaml``.

Stop the child with ``Stop-Process -Id <pid>``. A result file is only written when a task
finishes (ADR-024), so stopping it never corrupts results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the command after ``--`` detached; returns 0 once it has started."""
    parser = argparse.ArgumentParser(prog="tools/launch_detached.py")
    parser.add_argument("--stdout", required=True, type=Path, help="log file (appended)")
    parser.add_argument("--stderr", required=True, type=Path, help="log file (appended)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after --")
    if sys.platform != "win32":
        print("launch_detached: only supported on Windows", flush=True)
        return 1
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    with args.stdout.open("ab") as out, args.stderr.open("ab") as err:
        try:
            # Leave any job object the launcher is in (e.g. WMI's, which has memory quotas and
            # ends every process in the job when one exceeds them).
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, close_fds=True,
            )  # fmt: skip
            breakaway = True
        except PermissionError:  # the job does not allow breakaway
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=flags, close_fds=True,
            )  # fmt: skip
            breakaway = False
    print(f"started PID {child.pid} (job breakaway: {breakaway}): {' '.join(command)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
