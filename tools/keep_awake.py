"""Keep Windows from idle standby while a long experiment runs (not part of the research code).

Re-asserts ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)``
every 60 s, the same request a media player makes. It changes no settings, and the request is
released automatically when this process exits. It cannot prevent standby caused by closing the
lid, pressing the power button, or a low battery.

Usage (Windows only)::

    uv run python tools/keep_awake.py
"""

from __future__ import annotations

import ctypes
import sys
import time
from datetime import datetime

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def main(interval_seconds: float = 60.0) -> int:
    """Re-assert the keep-awake request forever; returns 1 if not on Windows."""
    if sys.platform != "win32":
        print("keep_awake: only supported on Windows", flush=True)
        return 1
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    print(f"{datetime.now():%d %b %H:%M} keep-awake request active", flush=True)
    while True:
        if ctypes.windll.kernel32.SetThreadExecutionState(flags) == 0:  # type: ignore[attr-defined]
            print(f"{datetime.now():%d %b %H:%M} SetThreadExecutionState FAILED", flush=True)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
