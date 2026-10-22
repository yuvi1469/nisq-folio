"""Tests for the read-only dashboard: its data module and a headless render of the app."""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nisq_folio.analysis.dashboard_data import load_dashboard, progress_table
from nisq_folio.config import load_config
from nisq_folio.data.prices import load_or_download
from nisq_folio.experiments import __main__ as runner_cli
from nisq_folio.provenance import GitState
from test_experiment_runner import small_config

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


@pytest.fixture
def tiny(tmp_path: Path, make_prices: Callable[..., pd.DataFrame]) -> Path:
    """A finished tiny experiment (N = 4, 2 instances, m in {0, 0.5})."""
    path = small_config(tmp_path, name="tiny", routing="compact_region")
    config = load_config(path)
    u = config.universe

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    return path


def test_progress_counts_before_and_after_running(tiny: Path) -> None:
    config = load_config(tiny)
    before = progress_table(config)
    assert before["done"].sum() == 0
    assert before["total"].sum() == 2 * 2 * (1 + 3 * 2)  # instances x denoising x tasks
    assert runner_cli.main(["--config", str(tiny)]) == 0
    after = progress_table(config)
    assert (after["done"] == after["total"]).all() and (after["share"] == 1).all()

    data = load_dashboard(config, replicates=50)
    assert len(data.qaoa) == 2 * 2 * 3 * 2 and len(data.classical) == 2 * 2
    assert set(data.noisy) == {"N1", "N2", "N3", "N4", "N5", "N6", "N7", "V"}
    assert set(data.noisy["V"]["verdict"]) <= {"beats random", "not supported"}


def test_load_dashboard_before_any_result(tiny: Path) -> None:
    data = load_dashboard(load_config(tiny))
    assert data.qaoa.empty and data.noisy == {}


def test_app_renders_every_tab(tiny: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NISQ_FOLIO_CONFIG", str(tiny))
    empty = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not empty.exception and empty.warning[0].value == "No finished QAOA tasks yet."

    assert runner_cli.main(["--config", str(tiny)]) == 0
    st.cache_data.clear()  # the first render cached the empty experiment
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Progress", "Noiseless", "Noise", "Instance explorer"]
    assert app.metric[0].value == "28 / 28"
    app.radio(key="den1").set_value("rmt_clip").run()
    app.radio[2].set_value("Feasible probability").run()
    assert not app.exception
