"""Tests for the optional ``problem`` and ``baselines`` config sections."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from nisq_folio.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))


def write(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def with_problem(**changes: object) -> dict:
    return {**BASE, "problem": {**BASE["problem"], **changes}}


def test_base_config_problem_and_baselines_sections() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    problem = config.require_problem()
    assert problem.sizes == (8, 10, 12, 14)
    assert problem.in_sample_end == date(2020, 12, 31)
    assert problem.risk_aversion == 4.5
    baselines = config.require_baselines()
    assert baselines.validation_instances_per_size == 50
    assert baselines.simulated_annealing["n_steps"] == 2000


def test_sections_are_optional_but_required_on_demand(tmp_path: Path) -> None:
    content = {k: v for k, v in BASE.items() if k not in ("problem", "baselines")}
    config = load_config(write(tmp_path, content))
    assert config.problem is None and config.baselines is None
    with pytest.raises(ConfigError, match="'problem' section"):
        config.require_problem()
    with pytest.raises(ConfigError, match="'baselines' section"):
        config.require_baselines()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (with_problem(sizes=[7]), "even integer"),
        (with_problem(sizes=[32]), "even integer"),
        (with_problem(sizes=[True]), "even integer"),
        (with_problem(in_sample_end="2014-06-30"), "strictly inside"),
        (with_problem(in_sample_end="2023-12-31"), "strictly inside"),
        (with_problem(risk_aversion=-0.5), ">= 0"),
        (with_problem(risk_aversion=True), ">= 0"),
        ({**BASE, "baselines": {**BASE["baselines"], "validation_instances_per_size": 0}}, ">= 1"),
    ],
)
def test_invalid_sections_are_rejected(tmp_path: Path, content: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, content))
