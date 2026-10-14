"""Tests for the benchmark's time limit: stop repeating slow cases and skip harder ones."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qiskit import QuantumCircuit

from helpers import random_problem
from nisq_folio.noise.__main__ import main
from nisq_folio.noise.benchmark import BenchmarkSettings, benchmark_case, bound_qaoa_circuit


def settings(max_seconds: float) -> BenchmarkSettings:
    return BenchmarkSettings(
        backend="FakeGuadalupeV2",
        shots=8,
        repeats=3,
        optimization_level=1,
        max_memory_mb=2000,
        penalty_margin=0.01,
        max_seconds_per_run=max_seconds,
    )


def circuit() -> QuantumCircuit:
    return bound_qaoa_circuit(random_problem(4, 2, 4.5, 0), 1, np.random.default_rng(0), 0.01)


def test_slow_case_stops_after_first_repeat() -> None:
    record = benchmark_case(circuit(), "ideal_statevector", settings(0.0), seed=1)
    assert record["status"] == "ok"
    assert record["exceeded_time_limit"] is True
    assert len(record["seconds"]) == 1


def test_fast_case_runs_all_repeats() -> None:
    record = benchmark_case(circuit(), "ideal_statevector", settings(1e6), seed=1)
    assert record["exceeded_time_limit"] is False
    assert len(record["seconds"]) == 3


def test_default_settings_never_stop_early() -> None:
    plain = BenchmarkSettings("FakeGuadalupeV2", 8, 2, 1, 2000, 0.01)
    record = benchmark_case(circuit(), "ideal_statevector", plain, seed=1)
    assert len(record["seconds"]) == 2
    assert record["exceeded_time_limit"] is False


def test_deeper_circuit_is_skipped_after_shallower_one_exceeds_limit(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame]
) -> None:
    tickers = [f"T{i}" for i in range(4)]
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 2,
        "universe": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "n_stocks": 4,
            "n_sectors": 1,
            "sectors": {"S": tickers},
        },
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4], "risk_aversion": 4.5},
        "qubo": {"penalty_margin": 0.01},
        "noise_benchmark": {
            "backend": "FakeGuadalupeV2",
            "sizes": [4],
            "layers": [1, 2],
            "shots": 8,
            "repeats": 2,
            "methods": ["ideal_statevector"],
            "optimization_level": 1,
            "max_seconds_per_run": 1e-9,
            "max_memory_mb": 2000,
        },
    }
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=1)

    assert main(["--config", str(config)], fetcher=fetcher) == 0
    record_path = next((tmp_path / "results" / "noise_benchmark").glob("*.json"))
    cases = json.loads(record_path.read_text(encoding="utf-8"))["results"]["cases"]
    assert [(c["layers"], c["status"]) for c in cases] == [
        (1, "ok"),
        (2, "skipped (shallower depth exceeded limits)"),
    ]
    assert cases[0]["exceeded_time_limit"] is True
