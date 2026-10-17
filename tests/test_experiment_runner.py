"""Tests for the experiment runner (ADR-024): grid, seeds, cost model, store, resume, CLI.

The key properties: results do not depend on task order, on resuming, or on parallelism; an
interrupted run resumes without redoing finished tasks; one directory never mixes configs.
"""

from collections.abc import Callable, Sequence
from datetime import date
from math import comb
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nisq_folio.config import ConfigError, load_config
from nisq_folio.experiments import __main__ as runner_cli
from nisq_folio.experiments.cost import evaluations_per_run, summarize_cost, task_seconds
from nisq_folio.experiments.grid import Task, expand_tasks
from nisq_folio.experiments.store import (
    ExperimentMismatchError,
    completed_task_ids,
    load_results,
    open_experiment,
)
from nisq_folio.provenance import GitState

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
TICKERS = [f"T{i}" for i in range(8)]


def small_config(tmp_path: Path, name: str = "demo", **experiment_changes: object) -> Path:
    """A tiny but complete config: N = 4, 2 instances, both denoising levels, all mixers."""
    experiment = {
        "name": name, "sizes": [4], "noisy_sizes": [4], "denoising": ["none", "rmt_clip"],
        "mixers": ["x", "xy_ring", "adapt"], "noise_levels": [0.0, 0.5],
        "instances_per_size": 2, "backend": "FakeKingston",
        "mixer_settings": {"x": {"optimizer": "cobyla", "layers": 1, "penalty_scale": 0.1},
                           "xy_ring": {"optimizer": "spsa", "layers": 1},
                           "adapt": {"optimizer": "cobyla", "layers": 1}},
        "classical_sa_budgets": [25, 100],
        "cost_model": {"noiseless_seconds_per_eval": 0.01,
                       "noisy_seconds_per_eval": {"x": {4: 0.1}, "xy_ring": {4: 0.1}}},
        **experiment_changes,
    }  # fmt: skip
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 11,
        "universe": {"start_date": "2020-01-01", "end_date": "2020-12-31", "n_stocks": 8,
                     "n_sectors": 1, "sectors": {"S": TICKERS}},
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4], "risk_aversion": 4.5},
        "qubo": {"penalty_margin": 0.01},
        "qaoa": {"shots": 32, "max_evaluations": 12, "rhobeg": 0.25,
                 "tuning": {"instances_per_size": 1, "excluded_evaluation_instances": 0,
                            "layers": [1], "penalty_scales": [1.0], "ramp_deltas": [0.75]}},
        "mixer_study": {"mixers": ["x"], "optimizers": ["spsa"], "layers": [1],
                        "x_penalty_scales": [0.1], "ramp_delta": 0.75,
                        "adapt_gradient_shift": 0.39, "spsa": {"calibration_samples": 2}},
        "experiment": experiment,
    }  # fmt: skip
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path: Path, make_prices: Callable[..., pd.DataFrame]) -> Path:
    """Write a synthetic price cache so the runner never touches the network."""
    from nisq_folio.data.prices import load_or_download

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    path = small_config(tmp_path)
    config = load_config(path)
    u = config.universe
    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    return path


def strip_volatile(record: dict) -> dict:
    """Drop fields that legitimately differ between identical runs (timing, provenance)."""
    return {k: v for k, v in record.items() if k not in ("wall_seconds", "finished_at", "git")}


# ---------------------------------------------------------------- config + grid


def test_base_config_experiment_section() -> None:
    experiment = load_config(REPO_ROOT / "configs" / "base.yaml").require_experiment()
    assert experiment.sizes == (8, 10, 12, 14) and experiment.noisy_sizes == (8, 10, 12)
    assert experiment.mixer_settings["adapt"].layers == 4
    assert experiment.mixer_settings["x"].penalty_scale == 0.1
    assert experiment.noisy_seconds_per_eval["adapt"][12] == 217.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"noisy_sizes": [6]}, "subset"), ({"noise_levels": [0.0, 0.0]}, "noise_levels"),
     ({"noise_levels": [-1.0]}, "noise_levels"), ({"mixers": ["x", "z"]}, "mixers"),
     ({"mixer_settings": {"x": {"optimizer": "adam", "layers": 1}}}, "optimizer"),
     ({"mixers": ["xy_ring"], "mixer_settings": {}}, "needs an entry"),
     ({"instances_per_size": 0}, "instances_per_size"), ({"sizes": [5]}, "even")],
)  # fmt: skip
def test_invalid_experiment_sections(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(small_config(tmp_path, **changes))


def test_grid_size_ordering_and_noise_only_for_noisy_sizes(tmp_path: Path) -> None:
    experiment = load_config(REPO_ROOT / "configs" / "base.yaml").require_experiment()
    tasks = expand_tasks(experiment)
    assert len(tasks) == 4 * 30 * 2 * (
        1 + 3
    )  # sizes x instances x denoising x (classical + mixers)
    assert len({t.task_id for t in tasks}) == len(tasks)
    small = load_config(small_config(tmp_path, sizes=[4, 6], noisy_sizes=[4],
                                     noise_levels=[0.5])).require_experiment()  # fmt: skip
    tasks = expand_tasks(small)
    noisy = [t for t in tasks if t.noise_level > 0]
    assert {t.n for t in noisy} == {4}
    assert any(t.noise_level == 0.0 and t.kind == "qaoa" and t.n == 4 for t in tasks)  # 0 added
    kinds = [(t.kind, t.noise_level > 0) for t in tasks]
    assert kinds == sorted(kinds, key=lambda k: (k[0] != "classical", k[1]))  # cheapest first


def test_seed_keys_are_unique_and_depend_only_on_identity() -> None:
    a = Task("qaoa", 8, 3, "none", "xy_ring", 0.25)
    b = Task("qaoa", 8, 3, "none", "xy_ring", 0.5)
    assert a.seed_keys() != b.seed_keys()
    assert a.seed_keys() == Task("qaoa", 8, 3, "none", "xy_ring", 0.25).seed_keys()
    experiment = load_config(REPO_ROOT / "configs" / "base.yaml").require_experiment()
    tasks = expand_tasks(experiment)
    assert len({t.seed_keys() for t in tasks}) == len(tasks)
    assert a.task_id == "qaoa__N08__i003__none__xy_ring__m0.25"
    assert Task("classical", 10, 0, "rmt_clip").task_id == "classical__N10__i000__rmt_clip"


# ---------------------------------------------------------------- cost model


def test_evaluations_per_run() -> None:
    assert evaluations_per_run("x", 8, 2, 100) == 46
    assert evaluations_per_run("xy_ring", 8, 1, 100) == 101
    assert evaluations_per_run("adapt", 8, 4, 100) == 100 + 2 * (1 + comb(8, 2)) * 4 + 1
    with pytest.raises(ValueError):
        evaluations_per_run("y", 8, 1, 100)


def test_cost_summary_marks_unknown_costs(tmp_path: Path) -> None:
    config = load_config(small_config(tmp_path))
    tasks = expand_tasks(config.require_experiment())
    noisy_adapt = Task("qaoa", 4, 0, "none", "adapt", 0.5)
    assert task_seconds(noisy_adapt, config) is None  # not in the cost table
    groups = summarize_cost(tasks, config)
    assert groups["adapt N=4 m=0.5"]["hours"] is None
    assert groups["xy_ring N=4 m=0.5"]["hours"] == pytest.approx(4 * 13 * 0.1 / 3600)


# ---------------------------------------------------------------- store


def test_store_refuses_a_different_config(tmp_path: Path, prepared: Path) -> None:
    config = load_config(prepared)
    from nisq_folio.data.prices import PriceMetadata

    meta = PriceMetadata([], "", "", "", "", 0, "", "", "", None, False, "")
    root = open_experiment(config, GitState(None, False), meta)
    assert (root / "manifest.json").exists()
    assert open_experiment(config, GitState(None, False), meta) == root  # same config: fine
    prepared.write_text(prepared.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(ExperimentMismatchError, match="different config"):
        open_experiment(load_config(prepared), GitState(None, False), meta)


# ---------------------------------------------------------------- end to end


def test_run_resume_and_records(prepared: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = load_config(prepared)
    root = config.results_dir / "experiments" / "demo"
    total = len(expand_tasks(config.require_experiment()))

    assert runner_cli.main(["--config", str(prepared), "--limit", "5"]) == 0
    first = {r["task_id"]: r for r in load_results(root)}
    assert len(first) == 5

    assert runner_cli.main(["--config", str(prepared)]) == 0  # resume
    assert len(completed_task_ids(root)) == total
    out = capsys.readouterr().out
    assert f"{total} tasks, 5 already done, running {total - 5}" in out
    after = {r["task_id"]: r for r in load_results(root)}
    for task_id, record in first.items():  # finished results were not redone
        assert after[task_id]["finished_at"] == record["finished_at"]
    assert not list((root / "runs").glob("*.tmp"))

    assert runner_cli.main(["--config", str(prepared)]) == 0
    assert "running 0" in capsys.readouterr().out

    for record in after.values():
        assert record["config_sha256"] and "git" in record
        if record["task"]["kind"] == "classical":
            assert set(record["simulated_annealing"]) == {"25", "100"}
            assert record["f_star"] <= record["greedy"]["objective"] + 1e-12
        else:
            mode = record["mode_feasible"]
            assert len(mode["selection"]) == 2 and 0 < mode["probability"] <= 1
            metrics = record["run"]["metrics"]
            if record["task"]["mixer"] != "x" and record["task"]["noise_level"] == 0.0:
                assert metrics["p_feasible"] == pytest.approx(1.0, abs=1e-10)
            if record["task"]["noise_level"] > 0:
                assert record["run"]["circuit_runs"] >= record["run"]["n_evaluations"]


def test_results_do_not_depend_on_parallelism(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame]
) -> None:
    from nisq_folio.data.prices import load_or_download

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    results = []
    for name, workers in (("serial", "1"), ("parallel", "2")):
        path = small_config(tmp_path, name=name, noise_levels=[0.0], mixers=["x", "xy_ring"],
                            denoising=["none"])  # fmt: skip
        config = load_config(path)
        u = config.universe
        load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                         GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
        assert runner_cli.main(["--config", str(path), "--workers", workers]) == 0
        results.append({r["task_id"]: strip_volatile(r) for r in
                        load_results(config.results_dir / "experiments" / name)})  # fmt: skip
    serial, parallel = results
    assert serial.keys() == parallel.keys()
    for task_id in serial:
        assert {k: v for k, v in serial[task_id].items() if k != "config_sha256"} == {
            k: v for k, v in parallel[task_id].items() if k != "config_sha256"
        }


def test_estimate_and_noise_filter(prepared: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert runner_cli.main(["--config", str(prepared), "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "adapt N=4 m=0.5" in out and "unknown" in out
    assert runner_cli.main(["--config", str(prepared), "--estimate", "--max-noise", "0"]) == 0
    assert "m=0.5" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        runner_cli.main(["--config", str(prepared), "--workers", "0"])
