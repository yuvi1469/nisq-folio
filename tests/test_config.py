"""Tests for nisq_folio.config."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from nisq_folio.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "paths": {"data_dir": "../data", "results_dir": "../results"},
    "seed": 7,
    "universe": {
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "n_stocks": 3,
        "n_sectors": 2,
        "sectors": {"Tech": ["AAA", "BBB"], "Energy": ["CCC"]},
    },
}


def write(tmp_path: Path, content: dict) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / "test.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def with_universe(**changes: object) -> dict:
    return {**VALID, "universe": {**VALID["universe"], **changes}}


def test_base_config_matches_project_scope() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    universe = config.universe
    assert len(universe.tickers) == 30
    assert len(set(universe.tickers)) == 30
    assert len(universe.sectors) == 6
    assert all(len(members) == 5 for members in universe.sectors.values())
    assert universe.start_date == date(2015, 1, 1)
    assert universe.end_date == date(2023, 12, 31)
    assert config.data_dir == REPO_ROOT / "data"
    assert config.results_dir == REPO_ROOT / "results"


def test_paths_resolve_relative_to_config_file_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, VALID)
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(path)
    assert config.data_dir == (tmp_path / "data").resolve()
    assert config.source_path == path.resolve()


def test_universe_helpers_preserve_config_order(tmp_path: Path) -> None:
    universe = load_config(write(tmp_path, VALID)).universe
    assert universe.tickers == ("AAA", "BBB", "CCC")
    assert universe.sector_of == {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (with_universe(sectors={"Tech": ["AAA", "AAA"], "Energy": ["CCC"]}), "duplicate"),
        (with_universe(n_stocks=4), "n_stocks"),
        (with_universe(n_sectors=3), "n_sectors"),
        (with_universe(start_date="2021-01-01"), "before"),
        (with_universe(end_date="not-a-date"), "ISO date"),
        (with_universe(sectors={"Tech": [], "Energy": ["CCC"]}), "non-empty"),
        ({**VALID, "seed": -1}, "non-negative"),
        ({**VALID, "seed": True}, "invalid type"),
        ({k: v for k, v in VALID.items() if k != "paths"}, "missing required field 'paths'"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, content: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, content))


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_unquoted_yaml_dates_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "unquoted.yaml"
    text = yaml.safe_dump(VALID, sort_keys=False).replace("'2020-01-01'", "2020-01-01")
    assert "start_date: 2020-01-01" in text  # YAML now parses it as a date, not a string
    path.write_text(text, encoding="utf-8")
    assert load_config(path).universe.start_date == date(2020, 1, 1)
