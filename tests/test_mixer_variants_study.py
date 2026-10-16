"""Tests for the mixer variants, the mixer-study rule, its config section, and the CLI."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.config import STUDY_MIXERS, STUDY_OPTIMIZERS, ConfigError, load_config
from nisq_folio.mixers.__main__ import build_grid, main
from nisq_folio.mixers.study import MixerConfiguration, select_per_mixer
from nisq_folio.mixers.variants import MIXERS, VariantSettings, adapt_pool, evolve, run_variant
from nisq_folio.mixers.xy import dicke_state
from nisq_folio.problem import PortfolioProblem
from nisq_folio.qaoa.optimizers import OPTIMIZERS
from nisq_folio.qaoa.tuning import SCORE_TOLERANCE
from nisq_folio.qubo.penalty import sufficient_penalty

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))


def fast(mixer: str, optimizer: str = "spsa", layers: int = 2) -> VariantSettings:
    return VariantSettings(mixer=mixer, optimizer=optimizer, layers=layers, shots=32,
                           max_evaluations=40)  # fmt: skip


# ---------------------------------------------------------------- variants


def test_constants_agree_with_config_module() -> None:
    assert STUDY_MIXERS == MIXERS
    assert STUDY_OPTIMIZERS == OPTIMIZERS


@settings(max_examples=15)
@given(problems(min_n=3, max_n=6), st.sampled_from(["xy_ring", "adapt"]), st.integers(0, 999))
def test_constraint_preserving_mixers_stay_feasible(
    problem: PortfolioProblem, mixer: str, seed: int
) -> None:
    run = run_variant(
        problem, sufficient_penalty(problem), fast(mixer), np.random.default_rng(seed)
    )
    assert run.metrics.p_feasible == pytest.approx(1.0, abs=1e-10)


def test_zero_angles_give_the_initial_states() -> None:
    n = 4
    phases = np.linspace(-1, 1, 2**n)
    ident = [lambda s, b: s] * 2
    np.testing.assert_allclose(
        evolve(dicke_state(n, 2), phases, np.zeros(2), np.zeros(2), ident), dicke_state(n, 2)
    )


@pytest.mark.parametrize("optimizer", ["cobyla", "spsa"])
def test_adapt_accounts_for_selection_shots(optimizer: str) -> None:
    problem = random_problem(5, 2, 4.5, seed=3)
    settings_ = fast("adapt", optimizer, layers=3)
    run = run_variant(problem, sufficient_penalty(problem), settings_, np.random.default_rng(1))
    pool_size = 1 + comb(5, 2)
    assert len(adapt_pool(5)) == pool_size
    assert run.selection_shots == 3 * 2 * pool_size * 32
    assert run.total_shots == run.optimization_shots + run.selection_shots
    assert run.optimization_shots == run.n_evaluations * 32
    assert run.n_evaluations <= 40
    assert len(run.mixer_sequence) == 3
    assert set(run.mixer_sequence) <= set(adapt_pool(5))
    json.dumps(run.as_dict())


@pytest.mark.parametrize("mixer", MIXERS)
def test_runs_are_reproducible(mixer: str) -> None:
    problem = random_problem(5, 2, 4.5, seed=4)
    bound = sufficient_penalty(problem)
    a = run_variant(problem, bound, fast(mixer), np.random.default_rng(7))
    b = run_variant(problem, bound, fast(mixer), np.random.default_rng(7))
    assert a.as_dict() == b.as_dict()
    assert a.selection_shots == (0 if mixer != "adapt" else a.selection_shots)


@pytest.mark.parametrize(
    "kwargs",
    [{"mixer": "y"}, {"optimizer": "adam"}, {"layers": 0}, {"penalty_scale": 0.0},
     {"adapt_gradient_shift": -1.0}],
)  # fmt: skip
def test_invalid_variant_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        VariantSettings(**{"mixer": "x", "optimizer": "spsa", "layers": 1, **kwargs})


# ---------------------------------------------------------------- study rule


def test_grid_varies_penalty_for_x_only() -> None:
    grid = build_grid(["x", "xy_ring"], ["cobyla", "spsa"], [1, 2], [1.0, 0.1])
    assert sum(c.mixer == "x" for c in grid) == 2 * 2 * 2
    assert {c.penalty_scale for c in grid if c.mixer == "xy_ring"} == {1.0}
    assert MixerConfiguration("x", "spsa", 2, 0.1).label() == "x spsa p=2 c=0.1"
    assert MixerConfiguration("adapt", "cobyla", 3, 1.0).label() == "adapt cobyla p=3"


def test_selection_is_per_mixer_with_tie_breaks() -> None:
    x_shallow = MixerConfiguration("x", "spsa", 1, 0.1)
    x_deep = MixerConfiguration("x", "cobyla", 3, 1.0)
    xy_cobyla = MixerConfiguration("xy_ring", "cobyla", 2, 1.0)
    xy_spsa = MixerConfiguration("xy_ring", "spsa", 2, 1.0)
    scores = {x_shallow: 0.0, x_deep: SCORE_TOLERANCE / 2, xy_cobyla: 1.0, xy_spsa: 1.05}
    chosen = select_per_mixer(scores)
    assert chosen["x"] == x_shallow  # smaller p wins within tolerance
    assert chosen["xy_ring"] == xy_cobyla  # COBYLA preferred within tolerance
    assert select_per_mixer({x_deep: 1.0, x_shallow: 0.0})["x"] == x_deep  # outside tolerance
    with pytest.raises(ValueError):
        select_per_mixer({})


# ---------------------------------------------------------------- config + CLI


def test_base_config_mixer_study_section() -> None:
    study = load_config(REPO_ROOT / "configs" / "base.yaml").require_mixer_study()
    assert study.mixers == ("x", "xy_ring", "adapt")
    assert study.optimizers == ("cobyla", "spsa")
    assert study.x_penalty_scales == (1.0, 0.1)
    assert study.adapt_gradient_shift == pytest.approx(np.pi / 8, abs=1e-4)


def with_study(**changes: object) -> dict:
    return {**BASE, "mixer_study": {**BASE["mixer_study"], **changes}}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (with_study(mixers=["z"]), "mixer_study.mixers"),
        (with_study(mixers=["x", "x"]), "mixer_study.mixers"),
        (with_study(optimizers=[]), "mixer_study.optimizers"),
        (with_study(layers=[0]), "mixer_study.layers"),
        (with_study(ramp_delta=0), "ramp_delta"),
        ({k: v for k, v in BASE.items() if k != "mixer_study"}, None),
    ],
)
def test_mixer_study_section_validation(tmp_path: Path, content: dict, message: str | None) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    if message is None:
        with pytest.raises(ConfigError, match="'mixer_study' section"):
            load_config(path).require_mixer_study()
    else:
        with pytest.raises(ConfigError, match=message):
            load_config(path)


def test_cli_runs_selects_per_mixer_and_records(
    tmp_path: Path,
    make_prices: Callable[..., pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tickers = [f"T{i}" for i in range(6)]
    content = {
        "paths": {"data_dir": "data", "results_dir": "results"},
        "seed": 3,
        "universe": {"start_date": "2020-01-01", "end_date": "2020-12-31", "n_stocks": 6,
                     "n_sectors": 1, "sectors": {"S": tickers}},
        "problem": {"in_sample_end": "2020-08-31", "sizes": [4], "risk_aversion": 4.5},
        "qubo": {"penalty_margin": 0.01},
        "qaoa": {"shots": 16, "max_evaluations": 30, "rhobeg": 0.25,
                 "tuning": {"instances_per_size": 2, "excluded_evaluation_instances": 2,
                            "layers": [1], "penalty_scales": [1.0], "ramp_deltas": [0.75]}},
        "mixer_study": {"mixers": ["x", "xy_ring", "adapt"], "optimizers": ["spsa"],
                        "layers": [1, 2], "x_penalty_scales": [0.1], "ramp_delta": 0.75,
                        "adapt_gradient_shift": 0.39, "spsa": {"calibration_samples": 2}},
    }  # fmt: skip
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=2)

    assert main(["--config", str(path)], fetcher=fetcher) == 0
    assert "selected: " in capsys.readouterr().out
    record = json.loads(next((tmp_path / "results" / "mixer_study").glob("*.json")).read_text())
    results = record["results"]
    assert set(results["selected"]) == {"x", "xy_ring", "adapt"}
    assert len(results["runs"]) == 2 * 6  # instances x configurations
    for r in results["runs"]:
        if r["config"]["mixer"] != "x":
            assert r["metrics"]["p_feasible"] == pytest.approx(1.0, abs=1e-10)
