"""Regression test: the runner must work when launched as ``python -m`` with workers > 1.

Calling ``main()`` from pytest did not catch a Windows-only failure: with ``python -m``, worker
functions defined in ``__main__`` cannot be unpickled by spawned processes (ADR-024). This test
runs the real command in a subprocess.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from nisq_folio.config import load_config
from nisq_folio.data.prices import load_or_download
from nisq_folio.experiments.store import completed_task_ids
from nisq_folio.provenance import GitState
from test_experiment_runner import small_config


def test_python_dash_m_with_two_workers(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame]
) -> None:
    path = small_config(tmp_path, name="sub", noise_levels=[0.0], mixers=["xy_ring"],
                        denoising=["none"])  # fmt: skip
    config = load_config(path)
    u = config.universe

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    result = subprocess.run(
        [sys.executable, "-m", "nisq_folio.experiments", "--config", str(path), "--workers", "2"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert len(completed_task_ids(config.results_dir / "experiments" / "sub")) == 4
