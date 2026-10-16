"""Tests for the QAOA tuning CLI, the qaoa config section, and separate instance streams."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nisq_folio.config import ConfigError, load_config
from nisq_folio.experiments.instances import sample_instance_tickers
from nisq_folio.qaoa.__main__ import main
from nisq_folio.seeding import STREAM_TUNING_INSTANCES

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
UNIVERSE = tuple(f"T{i:02d}" for i in range(30))


def write(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- instance streams


def test_tuning_stream_differs_from_evaluation_stream() -> None:
    evaluation = sample_instance_tickers(UNIVERSE, 8, 20, 1)
    tuning = sample_instance_tickers(UNIVERSE, 8, 20, 1, stream=STREAM_TUNING_INSTANCES)
    assert evaluation != tuning
    assert evaluation == sample_instance_tickers(UNIVERSE, 8, 20, 1)  # default unchanged


def test_excluded_instances_are_never_returned() -> None:
    small = tuple(f"T{i}" for i in range(6))
    everything = sample_instance_tickers(small, 3, 20, 0)  # all C(6, 3) = 20 instances
    excluded = everything[:15]
    rest = sample_instance_tickers(small, 3, 5, 0, stream=7, exclude=excluded)
    assert set(rest) == set(everything[15:])
    with pytest.raises(ValueError):
        sample_instance_tickers(small, 3, 6, 0, stream=7, exclude=excluded)


# ---------------------------------------------------------------- config section


def test_base_config_qaoa_section() -> None:
    qaoa = load_config(REPO_ROOT / "configs" / "base.yaml").require_qaoa()
    assert (qaoa.shots, qaoa.max_evaluations, qaoa.rhobeg) == (1024, 100, 0.25)
    assert qaoa.tuning.layers == (1, 2, 3, 4)
    assert qaoa.tuning.penalty_scales == (1.0, 0.5, 0.25, 0.1)
    assert qaoa.tuning.excluded_evaluation_instances == 50


def test_qaoa_section_is_required_on_demand(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, {k: v for k, v in BASE.items() if k != "qaoa"}))
    with pytest.raises(ConfigError, match="'qaoa' section"):
        config.require_qaoa()


def with_tuning(**changes: object) -> dict:
    return {**BASE, "qaoa": {**BASE["qaoa"], "tuning": {**BASE["qaoa"]["tuning"], **changes}}}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({**BASE, "qaoa": {**BASE["qaoa"], "shots": 0}}, "shots"),
        ({**BASE, "qaoa": {**BASE["qaoa"], "rhobeg": -1}}, "rhobeg"),
        (with_tuning(layers=[]), "must not be empty"),
        (with_tuning(layers=[0]), "qaoa.tuning.layers"),
        (with_tuning(penalty_scales=[True]), "qaoa.tuning.penalty_scales"),
        (with_tuning(excluded_evaluation_instances=-1), "excluded_evaluation_instances"),
    ],
)
def test_invalid_qaoa_sections_are_rejected(tmp_path: Path, content: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, content))


# ---------------------------------------------------------------- CLI


def test_tuning_cli_selects_and_records(
    tmp_path: Path,
    make_prices: Callable[..., pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tickers = [f"T{i}" for i in range(8)]
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 4,
        "universe": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "n_stocks": 8,
            "n_sectors": 1,
            "sectors": {"S": tickers},
        },
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4, 6], "risk_aversion": 4.5},
        "qubo": {"penalty_margin": 0.01},
        "qaoa": {
            "shots": 32,
            "max_evaluations": 10,
            "rhobeg": 0.25,
            "tuning": {
                "instances_per_size": 2,
                "excluded_evaluation_instances": 3,
                "layers": [1, 2],
                "penalty_scales": [1.0, 0.25],
                "ramp_deltas": [0.75],
            },
        },
    }

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=6)

    assert main(["--config", str(write(tmp_path, content))], fetcher=fetcher) == 0
    out = capsys.readouterr().out
    assert "selected: p=" in out
    path = next((tmp_path / "results" / "qaoa_tuning").glob("*.json"))
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    assert len(results["scores"]) == 4
    assert len(results["runs"]) == 2 * 2 * 4  # sizes x instances x configurations
    assert set(results["selected"]) == {"layers", "penalty_scale", "ramp_delta"}
    evaluation6 = sample_instance_tickers(tickers, 6, 3, 4)
    assert all(tuple(r["tickers"]) not in evaluation6 for r in results["runs"] if r["n"] == 6)
