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
class QuboConfig:
    """Settings for the QUBO encoding.

    Attributes:
        penalty_margin: Relative margin on the provable penalty bound (ADR-017).
    """

    penalty_margin: float


@dataclass(frozen=True)
class QaoaTuningConfig:
    """The QAOA tuning grid (ADR-020).

    Attributes:
        instances_per_size: Tuning instances per N (drawn from a separate stream).
        excluded_evaluation_instances: How many evaluation instances per N to exclude.
        layers: Candidate depths p.
        penalty_scales: Candidate c in P = c * provable bound.
        ramp_deltas: Candidate ramp Delta values.
    """

    instances_per_size: int
    excluded_evaluation_instances: int
    layers: tuple[int, ...]
    penalty_scales: tuple[float, ...]
    ramp_deltas: tuple[float, ...]


@dataclass(frozen=True)
class QaoaConfig:
    """QAOA optimization protocol settings.

    Attributes:
        shots: Shots per energy evaluation.
        max_evaluations: Optimizer evaluation budget.
        rhobeg: COBYLA initial trust-region radius.
        tuning: The tuning grid.
    """

    shots: int
    max_evaluations: int
    rhobeg: float
    tuning: QaoaTuningConfig


# Must match nisq_folio.noise.benchmark.BENCHMARK_METHODS (checked by a test); duplicated here so
# that loading a config does not import Qiskit.
NOISE_BENCHMARK_METHODS = ("ideal_statevector", "statevector", "density_matrix")


@dataclass(frozen=True)
class NoiseBenchmarkConfig:
    """Settings for the noisy-simulation runtime benchmark (ADR-007).

    Attributes:
        backend: Fake backend class name.
        sizes: Problem sizes N to time (even).
        layers: QAOA depths p to time.
        shots: Shots per energy evaluation.
        repeats: Timed repetitions per case.
        methods: Simulation methods to time.
        optimization_level: Transpiler optimization level, 0-3.
        max_seconds_per_run: Stop timing larger N for a method once it exceeds this.
        max_memory_mb: Aer memory cap.
    """

    backend: str
    sizes: tuple[int, ...]
    layers: tuple[int, ...]
    shots: int
    repeats: int
    methods: tuple[str, ...]
    optimization_level: int
    max_seconds_per_run: float
    max_memory_mb: int


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
        qubo: QUBO settings, or ``None`` if the file has no ``qubo`` section.
        noise_benchmark: Benchmark settings, or ``None`` if absent.
    """

    data_dir: Path
    results_dir: Path
    seed: int
    universe: UniverseConfig
    source_path: Path
    problem: ProblemConfig | None = None
    baselines: BaselinesConfig | None = None
    qubo: QuboConfig | None = None
    noise_benchmark: NoiseBenchmarkConfig | None = None
    qaoa: QaoaConfig | None = None

    def require_qaoa(self) -> QaoaConfig:
        """Return the ``qaoa`` section, raising ConfigError if the file has none."""
        if self.qaoa is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'qaoa' section")
        return self.qaoa

    def require_qubo(self) -> QuboConfig:
        """Return the ``qubo`` section, raising ConfigError if the file has none."""
        if self.qubo is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'qubo' section")
        return self.qubo

    def require_noise_benchmark(self) -> NoiseBenchmarkConfig:
        """Return the ``noise_benchmark`` section, raising ConfigError if the file has none."""
        if self.noise_benchmark is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'noise_benchmark' section")
        return self.noise_benchmark

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
    qubo = None
    if "qubo" in raw:
        margin = _require(
            _require(raw, "qubo", dict, source), "penalty_margin", (int, float), source
        )
        if isinstance(margin, bool) or margin <= 0:
            raise ConfigError(f"{source}: penalty_margin must be > 0, got {margin!r}")
        qubo = QuboConfig(penalty_margin=float(margin))
    qaoa = None
    if "qaoa" in raw:
        qaoa = _parse_qaoa(_require(raw, "qaoa", dict, source), source)
    noise_benchmark = None
    if "noise_benchmark" in raw:
        noise_benchmark = _parse_noise_benchmark(
            _require(raw, "noise_benchmark", dict, source), len(universe.tickers), source
        )

    return Config(
        data_dir=(base_dir / _require(paths, "data_dir", str, source)).resolve(),
        results_dir=(base_dir / _require(paths, "results_dir", str, source)).resolve(),
        seed=seed,
        universe=universe,
        source_path=source,
        problem=problem,
        baselines=baselines,
        qubo=qubo,
        noise_benchmark=noise_benchmark,
        qaoa=qaoa,
    )


def _parse_qaoa(raw: dict[str, Any], source: Path) -> QaoaConfig:
    def positive(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:  # noqa: ANN401
        value = _require(mapping, key, kind, source)
        if isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{source}: qaoa field {key!r} must be > 0, got {value!r}")
        return value

    def positive_list(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> tuple:
        values = _require(mapping, key, list, source)
        if not values:
            raise ConfigError(f"{source}: qaoa.tuning.{key} must not be empty")
        for v in values:
            if isinstance(v, bool) or not isinstance(v, kind) or v <= 0:
                raise ConfigError(f"{source}: invalid value {v!r} in qaoa.tuning.{key}")
        return tuple(values)

    tuning = _require(raw, "tuning", dict, source)
    excluded = _require(tuning, "excluded_evaluation_instances", int, source)
    if isinstance(excluded, bool) or excluded < 0:
        raise ConfigError(f"{source}: excluded_evaluation_instances must be >= 0")
    return QaoaConfig(
        shots=positive(raw, "shots", int),
        max_evaluations=positive(raw, "max_evaluations", int),
        rhobeg=float(positive(raw, "rhobeg", (int, float))),
        tuning=QaoaTuningConfig(
            instances_per_size=positive(tuning, "instances_per_size", int),
            excluded_evaluation_instances=excluded,
            layers=positive_list(tuning, "layers", int),
            penalty_scales=tuple(
                float(c) for c in positive_list(tuning, "penalty_scales", (int, float))
            ),
            ramp_deltas=tuple(float(d) for d in positive_list(tuning, "ramp_deltas", (int, float))),
        ),
    )


def _parse_noise_benchmark(
    raw: dict[str, Any], n_assets: int, source: Path
) -> NoiseBenchmarkConfig:
    def positive_ints(key: str, even: bool = False) -> tuple[int, ...]:
        values = _require(raw, key, list, source)
        for v in values:
            ok = isinstance(v, int) and not isinstance(v, bool) and v >= 1
            if not ok or (even and (v % 2 or v > n_assets)):
                raise ConfigError(f"{source}: invalid value {v!r} in noise_benchmark.{key}")
        if not values:
            raise ConfigError(f"{source}: noise_benchmark.{key} must not be empty")
        return tuple(values)

    def positive(key: str, kind: type | tuple[type, ...]) -> Any:  # noqa: ANN401
        value = _require(raw, key, kind, source)
        if isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{source}: noise_benchmark.{key} must be > 0, got {value!r}")
        return value

    methods = tuple(_require(raw, "methods", list, source))
    unknown = [m for m in methods if m not in NOISE_BENCHMARK_METHODS]
    if unknown or not methods:
        raise ConfigError(f"{source}: unknown or missing noise_benchmark.methods: {unknown}")
    level = _require(raw, "optimization_level", int, source)
    if isinstance(level, bool) or not 0 <= level <= 3:
        raise ConfigError(f"{source}: optimization_level must be 0-3, got {level!r}")
    return NoiseBenchmarkConfig(
        backend=_require(raw, "backend", str, source),
        sizes=positive_ints("sizes", even=True),
        layers=positive_ints("layers"),
        shots=positive("shots", int),
        repeats=positive("repeats", int),
        methods=methods,
        optimization_level=level,
        max_seconds_per_run=float(positive("max_seconds_per_run", (int, float))),
        max_memory_mb=positive("max_memory_mb", int),
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
