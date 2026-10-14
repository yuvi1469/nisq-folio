"""Tests for the optional ``qubo`` and ``noise_benchmark`` config sections."""

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


def with_bench(**changes: object) -> dict:
    return {**BASE, "noise_benchmark": {**BASE["noise_benchmark"], **changes}}


def test_base_config_sections() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert config.require_qubo().penalty_margin == 0.01
    bench = config.require_noise_benchmark()
    assert bench.backend == "FakeKingston"
    assert bench.sizes == (8, 10, 12, 14)
    assert bench.layers == (1, 2)
    assert bench.methods == ("ideal_statevector", "statevector", "density_matrix")


def test_sections_are_optional_but_required_on_demand(tmp_path: Path) -> None:
    content = {k: v for k, v in BASE.items() if k not in ("qubo", "noise_benchmark")}
    config = load_config(write(tmp_path, content))
    with pytest.raises(ConfigError, match="'qubo' section"):
        config.require_qubo()
    with pytest.raises(ConfigError, match="'noise_benchmark' section"):
        config.require_noise_benchmark()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({**BASE, "qubo": {"penalty_margin": 0}}, "penalty_margin"),
        ({**BASE, "qubo": {"penalty_margin": True}}, "penalty_margin"),
        (with_bench(sizes=[9]), "noise_benchmark.sizes"),
        (with_bench(sizes=[]), "must not be empty"),
        (with_bench(layers=[0]), "noise_benchmark.layers"),
        (with_bench(shots=0), "shots must be > 0"),
        (with_bench(max_seconds_per_run=-1), "max_seconds_per_run"),
        (with_bench(methods=["mps"]), "methods"),
        (with_bench(methods=[]), "methods"),
        (with_bench(optimization_level=4), "optimization_level"),
    ],
)
def test_invalid_sections_are_rejected(tmp_path: Path, content: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, content))
