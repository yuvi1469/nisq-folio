"""Tests for ``python -m nisq_folio.baselines`` (network faked, small problem sizes)."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nisq_folio.baselines.__main__ import main

TICKERS = [f"T{i}" for i in range(8)]


def write_config(tmp_path: Path) -> Path:
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 5,
        "universe": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "n_stocks": 8,
            "n_sectors": 2,
            "sectors": {"S1": TICKERS[:4], "S2": TICKERS[4:]},
        },
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4, 6], "risk_aversion": 4.5},
        "baselines": {
            "validation_instances_per_size": 3,
            "simulated_annealing": {"n_steps": 300},
        },
    }
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def test_validation_run_prints_summary_and_writes_record(
    tmp_path: Path,
    make_prices: Callable[..., pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetcher(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(tickers, date(2020, 1, 2), 250, seed=9)

    assert main(["--config", str(write_config(tmp_path))], fetcher=fetcher) == 0
    out = capsys.readouterr().out
    for solver in ("greedy", "markowitz_rounding", "simulated_annealing"):
        assert out.count(solver) == 2  # one row per size

    records = list((tmp_path / "results" / "baselines_validation").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert set(record["results"]["sizes"]) == {"4", "6"}
    size6 = record["results"]["sizes"]["6"]
    assert size6["k"] == 3
    assert len(size6["instances"]) == 3
    for instance in size6["instances"]:
        assert instance["n_subsets"] == 20
        for score in instance["solvers"].values():
            assert 0.0 <= score["gap"] <= 1.0
            assert score["rank"] >= 1
            assert score["optimal"] == (score["rank"] == 1)
    assert record["data"]["n_rows"] == 250
