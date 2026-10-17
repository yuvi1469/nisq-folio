"""Tests for the pre-registered pilot rules (ADR-025), noisy exclusions, and the pilot CLI."""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from nisq_folio.config import ConfigError, load_config
from nisq_folio.data.prices import load_or_download
from nisq_folio.experiments import pilot_run
from nisq_folio.experiments.grid import expand_tasks
from nisq_folio.experiments.pilot import choose_levels, interpolated_crossing, upper_noise_level
from nisq_folio.provenance import GitState
from test_experiment_runner import small_config

LADDER = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# ---------------------------------------------------------------- Rule 1


def test_crossing_interpolates_in_log_m() -> None:
    # A falls from 2 at m=1 to 0.5 at m=4: A = 1 two-thirds of the way in log m.
    m = interpolated_crossing([0.25, 1.0, 4.0], [3.0, 2.0, 0.5])
    assert m == pytest.approx(4 ** (2 / 3))
    assert interpolated_crossing([0.25, 1.0], [0.9, 0.5]) == 0.25  # already below at the start
    assert interpolated_crossing([0.25, 1.0, 4.0], [3.0, 2.0, 1.5]) is None  # never crosses


@given(st.lists(st.floats(0.01, 10.0), min_size=2, max_size=6))
def test_crossing_lies_between_measured_levels(values: list[float]) -> None:
    levels = [0.25 * 2**i for i in range(len(values))]
    m = interpolated_crossing(levels, values)
    if m is not None:
        assert levels[0] <= m <= levels[-1]


def test_upper_level_is_the_smallest_ladder_value_covering_every_crossing() -> None:
    assert upper_noise_level({"x": 0.3, "xy_ring": 1.7, "adapt": 2.5}, LADDER) == 4.0
    assert upper_noise_level({"x": 0.2, "xy_ring": 0.4}, LADDER) == 1.0  # at least m = 1
    assert upper_noise_level({"x": 0.2, "xy_ring": None}, LADDER) == 8.0  # never crosses


# ---------------------------------------------------------------- Rule 2


def test_levels_drop_smallest_first_until_within_budget() -> None:
    def hours(levels: list[float]) -> float:
        return 10.0 * len(levels)

    levels, h, ok = choose_levels(LADDER, 4.0, hours, budget_hours=40, min_noisy_levels=3)
    assert levels == [0.5, 1.0, 2.0, 4.0] and h == 40 and ok
    levels, h, ok = choose_levels(LADDER, 4.0, hours, budget_hours=10, min_noisy_levels=3)
    assert levels == [1.0, 2.0, 4.0] and h == 30 and not ok  # overrun reported, not hidden


# ---------------------------------------------------------------- exclusions + config


def test_noisy_exclusions_keep_the_noiseless_cell(tmp_path: Path) -> None:
    path = small_config(tmp_path, sizes=[4, 6], noisy_sizes=[4, 6], noise_levels=[0.5],
                        noisy_exclusions={"adapt": [6]})  # fmt: skip
    tasks = expand_tasks(load_config(path).require_experiment())
    adapt6 = [t for t in tasks if t.mixer == "adapt" and t.n == 6]
    assert adapt6 and all(t.noise_level == 0.0 for t in adapt6)
    assert any(t.mixer == "adapt" and t.n == 4 and t.noise_level == 0.5 for t in tasks)
    with pytest.raises(ConfigError, match="noisy_exclusions"):
        load_config(small_config(tmp_path, name="bad", noisy_exclusions={"adapt": [8]}))


def test_base_config_pilot_section() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "base.yaml")
    pilot = config.require_pilot()
    assert pilot.budget_hours == 100 and pilot.noise_ladder[0] == 0.125
    assert config.require_experiment().noisy_exclusions == {"adapt": (12,)}


# ---------------------------------------------------------------- CLI


def test_pilot_cli_end_to_end(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    path = small_config(tmp_path, name="pilot")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    content["qaoa"]["tuning"]["excluded_evaluation_instances"] = 2
    content["pilot"] = {
        "timing_sizes": [4], "timing_repeats": 1, "skip_timing": {"adapt": [4]},
        "thread_test_sizes": [4], "accuracy_size": 4, "accuracy_instances": 1,
        "accuracy_noise_levels": [0.5, 2.0], "noise_ladder": [0.5, 1.0, 2.0],
        "min_noisy_levels": 1, "budget_hours": 1.0,
    }  # fmt: skip
    content["experiment"]["mixers"] = ["x", "xy_ring"]
    content["experiment"]["mixer_settings"].pop("adapt")
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    u = config.universe

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    assert pilot_run.main(["--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Rule 1: m_max" in out and "Rule 2: noisy levels" in out
    record = json.loads(next((config.results_dir / "pilot").glob("*.json")).read_text())
    results = record["results"]
    assert set(results["curves"]) == {"x", "xy_ring"}
    assert set(results["curves"]["xy_ring"]) == {"0", "0.5", "2"}
    assert results["workers"] in (1, 3)
