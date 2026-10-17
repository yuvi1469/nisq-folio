"""Tests for the pre-registered summary tables (ADR-026) and the analysis CLI."""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nisq_folio.analysis import __main__ as analysis_cli
from nisq_folio.analysis.summary import (
    budget_table,
    cell_table,
    flatten,
    markdown_table,
    mixer_table,
    paired_log_ratio,
)
from nisq_folio.config import load_config
from nisq_folio.data.prices import load_or_download
from nisq_folio.experiments import __main__ as runner_cli
from nisq_folio.experiments.store import experiment_dir, load_results
from nisq_folio.provenance import GitState
from test_experiment_runner import small_config


def test_paired_log_ratio_known_values() -> None:
    a, b = pd.Series([2.0, 4.0, 1.0]), pd.Series([1.0, 1.0, 1.0])
    out = paired_log_ratio(a, b)
    assert out["ratio"] == pytest.approx(2.0)  # median of ln(2), ln(4), 0
    assert out["share_better"] == pytest.approx(2 / 3)
    assert out["pairs"] == 3 and 0 <= out["wilcoxon_p"] <= 1
    assert np.isnan(paired_log_ratio(pd.Series([1.0]), pd.Series([1.0]))["wilcoxon_p"])


def test_markdown_table() -> None:
    text = markdown_table(pd.DataFrame({"N": [8], "A": [1.23456], "x": [float("nan")]}))
    assert text.splitlines() == ["| N | A | x |", "|---|---|---|", "| 8 | 1.23 | nan |"]


def test_budget_table_picks_smallest_budget_reaching_90_percent() -> None:
    classical = pd.DataFrame({
        "denoising": ["none"] * 10, "n": [8] * 10,
        "sa_25_optimal": [True] * 5 + [False] * 5, "sa_100_optimal": [True] * 9 + [False],
        "sa_2000_optimal": [True] * 10,
    })  # fmt: skip
    qaoa = pd.DataFrame({"denoising": ["none"] * 2, "n": [8, 8], "mixer": ["x", "x"],
                         "total_shots": [100, 300]})  # fmt: skip
    table = budget_table(qaoa, classical)
    assert table.loc[0, "sa_budget_90"] == 100 and table.loc[0, "qaoa_total_shots"] == 200


def test_tables_from_a_real_tiny_experiment(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    path = small_config(tmp_path, name="tiny", noise_levels=[0.0])
    config = load_config(path)
    u = config.universe

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    assert analysis_cli.main(["--config", str(path)]) == 1  # nothing to summarize yet
    assert runner_cli.main(["--config", str(path)]) == 0

    qaoa, classical = flatten(load_results(experiment_dir(config)))
    assert len(qaoa) == 2 * 2 * 3 and len(classical) == 2 * 2  # instances x denoising (x mixers)
    cells = cell_table(qaoa)
    assert len(cells) == 2 * 3 and (cells["instances"] == 2).all()
    assert set(mixer_table(qaoa)["comparison"]) == {"adapt / xy_ring", "xy_ring / x", "adapt / x"}

    assert analysis_cli.main(["--config", str(path), "--max-noise", "0"]) == 0
    out = capsys.readouterr().out
    for key in ("T1.", "T2.", "T3.", "T4.", "T5.", "T6."):
        assert key in out
    report = (config.results_dir / "analysis" / "tiny_m0.md").read_text(encoding="utf-8")
    assert report.count("## T") == 6 and "|---|" in report
