"""Load and validate project configuration from YAML files under ``configs/``.

Relative paths in a config file are resolved against the directory containing that file,
so results never depend on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or internally inconsistent."""


@dataclass(frozen=True)
class UniverseConfig:
    """The asset universe: tickers grouped by sector, and the historical window.

    Attributes:
        sectors: Mapping of sector name to the tickers in that sector, in config order.
        start_date: First calendar date of the window (inclusive).
        end_date: Last calendar date of the window (inclusive).
    """

    sectors: dict[str, tuple[str, ...]]
    start_date: date
    end_date: date

    @property
    def tickers(self) -> tuple[str, ...]:
        """All tickers, sector by sector, in config order."""
        return tuple(t for members in self.sectors.values() for t in members)

    @property
    def sector_of(self) -> dict[str, str]:
        """Mapping of each ticker to its sector."""
        return {t: sector for sector, members in self.sectors.items() for t in members}


@dataclass(frozen=True)
class ProblemConfig:
    """Settings that define the optimization problem.

    Attributes:
        in_sample_end: Last date (inclusive) of the window used to estimate mu and Sigma.
        sizes: Problem sizes N; each must be even, since K = N / 2.
        risk_aversion: lambda in the objective (ADR-015).
    """

    in_sample_end: date
    sizes: tuple[int, ...]
    risk_aversion: float


@dataclass(frozen=True)
class BaselinesConfig:
    """Settings for the classical baselines.

    Attributes:
        validation_instances_per_size: Instances per N in the brute-force validation run.
        simulated_annealing: Keyword arguments for ``AnnealingSettings`` (validated there).
    """

    validation_instances_per_size: int
    simulated_annealing: dict[str, Any]


@dataclass(frozen=True)
class Config:
    """Validated project configuration.

    Attributes:
        data_dir: Absolute directory for downloaded / cached market data.
        results_dir: Absolute directory for experiment outputs.
        seed: Root seed from which every per-run seed is derived (ADR-004).
        universe: The asset universe and historical window.
        source_path: Absolute path of the YAML file this config was loaded from.
        problem: Problem settings, or ``None`` if the file has no ``problem`` section.
        baselines: Baseline settings, or ``None`` if the file has no ``baselines`` section.
    """

    data_dir: Path
    results_dir: Path
    seed: int
    universe: UniverseConfig
    source_path: Path
    problem: ProblemConfig | None = None
    baselines: BaselinesConfig | None = None

    def require_problem(self) -> ProblemConfig:
        """Return the ``problem`` section, raising ConfigError if the file has none."""
        if self.problem is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'problem' section")
        return self.problem

    def require_baselines(self) -> BaselinesConfig:
        """Return the ``baselines`` section, raising ConfigError if the file has none."""
        if self.baselines is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'baselines' section")
        return self.baselines


def load_config(path: Path | str) -> Config:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration, with paths resolved to absolute paths.

    Raises:
        ConfigError: If a required field is missing or the values are inconsistent.
    """
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: expected a YAML mapping at the top level")

    base_dir = source.parent
    paths = _require(raw, "paths", dict, source)
    universe = _parse_universe(_require(raw, "universe", dict, source), source)

    seed = _require(raw, "seed", int, source)
    if seed < 0:
        raise ConfigError(f"{source}: seed must be non-negative, got {seed}")

    problem = None
    if "problem" in raw:
        problem = _parse_problem(_require(raw, "problem", dict, source), universe, source)
    baselines = None
    if "baselines" in raw:
        baselines = _parse_baselines(_require(raw, "baselines", dict, source), source)

    return Config(
        data_dir=(base_dir / _require(paths, "data_dir", str, source)).resolve(),
        results_dir=(base_dir / _require(paths, "results_dir", str, source)).resolve(),
        seed=seed,
        universe=universe,
        source_path=source,
        problem=problem,
        baselines=baselines,
    )


def _parse_problem(raw: dict[str, Any], universe: UniverseConfig, source: Path) -> ProblemConfig:
    end = _parse_date(_require(raw, "in_sample_end", (str, date), source), "in_sample_end", source)
    if not universe.start_date < end < universe.end_date:
        raise ConfigError(f"{source}: in_sample_end {end} must lie strictly inside the window")
    sizes = _require(raw, "sizes", list, source)
    n_assets = len(universe.tickers)
    for n in sizes:
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2 or n > n_assets:
            raise ConfigError(
                f"{source}: every size must be an even integer in [2, {n_assets}], got {n!r}"
            )
    risk = _require(raw, "risk_aversion", (int, float), source)
    if isinstance(risk, bool) or risk < 0:
        raise ConfigError(f"{source}: risk_aversion must be a number >= 0, got {risk!r}")
    return ProblemConfig(in_sample_end=end, sizes=tuple(sizes), risk_aversion=float(risk))


def _parse_baselines(raw: dict[str, Any], source: Path) -> BaselinesConfig:
    count = _require(raw, "validation_instances_per_size", int, source)
    if count < 1:
        raise ConfigError(f"{source}: validation_instances_per_size must be >= 1")
    annealing = _require(raw, "simulated_annealing", dict, source)
    return BaselinesConfig(validation_instances_per_size=count, simulated_annealing=dict(annealing))


def _parse_universe(raw: dict[str, Any], source: Path) -> UniverseConfig:
    start = _parse_date(_require(raw, "start_date", (str, date), source), "start_date", source)
    end = _parse_date(_require(raw, "end_date", (str, date), source), "end_date", source)
    if start >= end:
        raise ConfigError(f"{source}: start_date {start} must be before end_date {end}")

    raw_sectors = _require(raw, "sectors", dict, source)
    sectors: dict[str, tuple[str, ...]] = {}
    for sector, members in raw_sectors.items():
        if not isinstance(members, list) or not members:
            raise ConfigError(f"{source}: sector {sector!r} must be a non-empty list of tickers")
        sectors[str(sector)] = tuple(str(t) for t in members)

    universe = UniverseConfig(sectors=sectors, start_date=start, end_date=end)
    tickers = universe.tickers
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
    if duplicates:
        raise ConfigError(f"{source}: duplicate tickers in universe: {duplicates}")

    n_stocks = _require(raw, "n_stocks", int, source)
    n_sectors = _require(raw, "n_sectors", int, source)
    if len(tickers) != n_stocks:
        raise ConfigError(f"{source}: n_stocks={n_stocks} but {len(tickers)} tickers listed")
    if len(sectors) != n_sectors:
        raise ConfigError(f"{source}: n_sectors={n_sectors} but {len(sectors)} sectors listed")
    return universe


def _parse_date(value: str | date, field: str, source: Path) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{source}: {field} is not an ISO date: {value!r}") from exc


def _require(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...], source: Path) -> Any:  # noqa: ANN401
    if key not in mapping:
        raise ConfigError(f"{source}: missing required field {key!r}")
    value = mapping[key]
    # bool is a subclass of int; reject it explicitly where an int is expected.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"{source}: field {key!r} has invalid type {type(value).__name__}")
    return value
