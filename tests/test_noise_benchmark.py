"""Tests for nisq_folio.noise (fake backends, simulators, runtime benchmark, CLI)."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from helpers import random_problem
from nisq_folio.config import NOISE_BENCHMARK_METHODS
from nisq_folio.noise.__main__ import main
from nisq_folio.noise.benchmark import (
    BENCHMARK_METHODS,
    BenchmarkSettings,
    benchmark_case,
    bound_qaoa_circuit,
    machine_info,
)
from nisq_folio.noise.models import SIMULATION_METHODS, fake_backend, noisy_simulator

SETTINGS = BenchmarkSettings(
    backend="FakeGuadalupeV2",  # small 16-qubit device: fast noise-model build in tests
    shots=16,
    repeats=1,
    optimization_level=1,
    max_memory_mb=2000,
    penalty_margin=0.01,
)


def test_config_and_benchmark_method_lists_agree() -> None:
    assert NOISE_BENCHMARK_METHODS == BENCHMARK_METHODS


def test_fake_backend_lookup() -> None:
    assert fake_backend("FakeKingston").num_qubits == 156
    for bad in ("FakeDoesNotExist", "AerSimulator"):
        with pytest.raises(ValueError, match="unknown fake backend"):
            fake_backend(bad)


def test_noisy_simulator_methods() -> None:
    backend = fake_backend("FakeGuadalupeV2")
    for method in SIMULATION_METHODS:
        assert noisy_simulator(backend, method, seed=1).options.method == method
    with pytest.raises(ValueError, match="method"):
        noisy_simulator(backend, "stabilizer", seed=1)


def test_bound_circuit_is_measured_and_fully_bound() -> None:
    circuit = bound_qaoa_circuit(random_problem(4, 2, 4.5, 0), 2, np.random.default_rng(0), 0.01)
    assert not circuit.parameters
    assert circuit.count_ops()["measure"] == 4


@pytest.mark.parametrize("method", BENCHMARK_METHODS)
def test_benchmark_case_times_each_method(method: str) -> None:
    circuit = bound_qaoa_circuit(random_problem(4, 2, 4.5, 0), 1, np.random.default_rng(0), 0.01)
    record = benchmark_case(circuit, method, SETTINGS, seed=3)
    assert record["status"] == "ok"
    assert record["median_seconds"] > 0
    assert len(record["seconds"]) == 1
    if method == "ideal_statevector":
        assert record["two_qubit_gates"] == 6  # the 6 logical RZZ, no routing
    else:
        assert record["two_qubit_gates"] >= 12  # each RZZ needs >= 2 CZ on hardware


def test_benchmark_case_reports_memory_refusal() -> None:
    # N=10 needs 4^10 * 16 B = 16 MB for rho, above the 1 MB cap. (N=8 needs exactly 1 MB,
    # which Aer accepts, so it cannot trigger the refusal.)
    circuit = bound_qaoa_circuit(random_problem(10, 5, 4.5, 0), 1, np.random.default_rng(0), 0.01)
    tiny = BenchmarkSettings(**{**SETTINGS.__dict__, "max_memory_mb": 1})
    record = benchmark_case(circuit, "density_matrix", tiny, seed=3)
    assert record["status"] == "refused"
    assert "memory" in record["error"].lower()


def test_benchmark_case_rejects_unknown_method() -> None:
    circuit = bound_qaoa_circuit(random_problem(3, 1, 4.5, 0), 1, np.random.default_rng(0), 0.01)
    with pytest.raises(ValueError, match="method"):
        benchmark_case(circuit, "mps", SETTINGS, seed=0)


def test_machine_info_describes_hardware() -> None:
    info = machine_info()
    assert info["logical_cpus"] >= 1
    assert info["total_ram_gb"] is None or info["total_ram_gb"] > 0
    assert set(info) >= {"platform", "processor", "python"}


def test_cli_runs_and_skips_larger_sizes_after_limit(
    tmp_path: Path,
    make_prices: Callable[..., pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tickers = [f"T{i}" for i in range(6)]
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 2,
        "universe": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "n_stocks": 6,
            "n_sectors": 1,
            "sectors": {"S": tickers},
        },
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4], "risk_aversion": 4.5},
        "qubo": {"penalty_margin": 0.01},
        "noise_benchmark": {
            "backend": "FakeGuadalupeV2",
            "sizes": [4, 6],
            "layers": [1],
            "shots": 8,
            "repeats": 1,
            "methods": ["ideal_statevector"],
            "optimization_level": 1,
            "max_seconds_per_run": 1e-9,  # every run "exceeds" this -> N=6 is skipped
            "max_memory_mb": 2000,
        },
    }
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=1)

    assert main(["--config", str(config)], fetcher=fetcher) == 0
    assert "skipped" in capsys.readouterr().out
    record_path = next((tmp_path / "results" / "noise_benchmark").glob("*.json"))
    cases = json.loads(record_path.read_text(encoding="utf-8"))["results"]["cases"]
    assert [(c["n"], c["status"]) for c in cases] == [
        (4, "ok"),
        (6, "skipped (smaller N exceeded limits)"),
    ]
