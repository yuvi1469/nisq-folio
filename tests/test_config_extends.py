"""Tests for config inheritance (``extends``) and the experiment config of record (ADR-025)."""

from pathlib import Path

import pytest
import yaml

from nisq_folio.config import ConfigError, config_chain_text, load_config
from nisq_folio.experiments.grid import expand_tasks
from nisq_folio.experiments.store import config_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def test_child_overrides_and_nested_mappings_merge(tmp_path: Path) -> None:
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    write(tmp_path / "base.yaml", base)
    overrides = {"noise_levels": [0.0, 2.0], "cost_model": {"noiseless_seconds_per_eval": 1.0}}
    child = write(
        tmp_path / "child.yaml", {"extends": "base.yaml", "seed": 99, "experiment": overrides}
    )
    config = load_config(child)
    experiment = config.require_experiment()
    assert config.seed == 99
    assert experiment.noise_levels == (0.0, 2.0)  # lists replace
    assert experiment.noiseless_seconds_per_eval == 1.0  # nested mapping merged ...
    assert experiment.noisy_seconds_per_eval["x"][8] == 0.17  # ... siblings kept from the base
    assert experiment.name == base["experiment"]["name"]
    assert [p.name for p in config.source_chain] == ["base.yaml", "child.yaml"]


def test_fingerprint_and_text_cover_the_whole_chain(tmp_path: Path) -> None:
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base_path = write(tmp_path / "base.yaml", base)
    child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "seed": 3})
    before = config_fingerprint(load_config(child))
    text = config_chain_text(load_config(child))
    assert "# --- " in text and "seed: 3" in text
    base_path.write_text(base_path.read_text(encoding="utf-8") + "\n# parent edited\n", "utf-8")
    assert config_fingerprint(load_config(child)) != before  # parent change is detected


def test_single_file_text_is_unchanged() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert config_chain_text(config) == (REPO_ROOT / "configs" / "base.yaml").read_text("utf-8")


def test_invalid_extends_are_rejected(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")
    write(tmp_path / "c.yaml", {"extends": 5})
    with pytest.raises(ConfigError, match="file path"):
        load_config(tmp_path / "c.yaml")


def test_main_config_is_the_pilot_selection() -> None:
    config = load_config(REPO_ROOT / "configs" / "main.yaml")
    experiment = config.require_experiment()
    assert experiment.noise_levels == (0.0, 0.25, 0.5, 1.0)
    assert experiment.noisy_exclusions == {"adapt": (12,)}
    tasks = expand_tasks(experiment)
    noisy = [t for t in tasks if t.noise_level > 0]
    assert {t.n for t in noisy} == {8, 10, 12}
    assert not any(t.mixer == "adapt" and t.n == 12 for t in noisy)
    assert len(tasks) == 2400
