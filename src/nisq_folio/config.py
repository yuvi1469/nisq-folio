"""Load and validate project configuration from YAML files under ``configs/``.

Relative paths in a config file are resolved against the directory containing that file,
so results never depend on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


DENOISING_LEVELS = ("none", "rmt_clip")
# Must match nisq_folio.noise.density.ROUTING_MODES (checked by a test).
ROUTING_MODES = ("full_device", "compact_region")


@dataclass(frozen=True)
class MixerSetting:
    """One mixer's fixed protocol in the experiment (an ADR-021 selection)."""

    optimizer: str
    layers: int
    penalty_scale: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """The factorial experiment grid (ADR-024).

    Attributes:
        name: Experiment name; results go to ``results/experiments/<name>/``.
        sizes: Problem sizes N.
        noisy_sizes: Sizes at which noise levels > 0 are run (ADR-007).
        denoising: Denoising levels, from ``DENOISING_LEVELS``.
        mixers: Mixers to run.
        noise_levels: Noise levels m (0 = noiseless).
        instances_per_size: Replicates per cell: the first evaluation instances per N.
        backend: Fake backend for the coupling map of noisy runs.
        mixer_settings: Mixer -> its fixed optimizer, depth and penalty scale.
        classical_sa_budgets: Step budgets for the simulated-annealing curves.
        noiseless_seconds_per_eval: Cost model for ``--estimate``.
        noisy_seconds_per_eval: Mixer -> {N: seconds per noisy evaluation at m = 1}.
        noisy_exclusions: Mixer -> sizes at which it runs noiselessly only (ADR-025).
        routing: Qubit placement of noisy circuits (ADR-027); default ``full_device``.
    """

    name: str
    sizes: tuple[int, ...]
    noisy_sizes: tuple[int, ...]
    denoising: tuple[str, ...]
    mixers: tuple[str, ...]
    noise_levels: tuple[float, ...]
    instances_per_size: int
    backend: str
    mixer_settings: dict[str, MixerSetting]
    classical_sa_budgets: tuple[int, ...]
    noiseless_seconds_per_eval: float
    noisy_seconds_per_eval: dict[str, dict[int, float]]
    noisy_exclusions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    routing: str = "full_device"


@dataclass(frozen=True)
class PilotConfig:
    """Pilot settings for the noisy grid (ADR-025); see ``experiments/pilot.py`` for the rules."""

    timing_sizes: tuple[int, ...]
    timing_repeats: int
    skip_timing: dict[str, tuple[int, ...]]
    thread_test_sizes: tuple[int, ...]
    accuracy_size: int
    accuracy_instances: int
    accuracy_noise_levels: tuple[float, ...]
    noise_ladder: tuple[float, ...]
    min_noisy_levels: int
    budget_hours: float


# Must match the RMT estimators in nisq_folio.denoise.evaluation.ESTIMATORS (checked by a test).
DENOISE_METHODS = ("rmt_clip", "rmt_band_only")


@dataclass(frozen=True)
class DenoiseConfig:
    """Covariance denoising settings (ADR-022).

    Attributes:
        experiment_method: The estimator used as the "RMT" level of the denoising factor.
        evaluation_instances_per_size: Instances per N in the out-of-sample evaluation.
    """

    experiment_method: str
    evaluation_instances_per_size: int


# Must match nisq_folio.mixers.variants.MIXERS and nisq_folio.qaoa.optimizers.OPTIMIZERS
# (checked by a test); duplicated so that loading a config does not import Qiskit.
STUDY_MIXERS = ("x", "xy_ring", "adapt")
STUDY_OPTIMIZERS = ("cobyla", "spsa")


@dataclass(frozen=True)
class MixerStudyConfig:
    """The mixer study grid (ADR-021).

    Attributes:
        mixers: Mixers to compare.
        optimizers: Optimizers to compare.
        layers: Candidate depths p.
        x_penalty_scales: Penalty scales c for the X mixer.
        ramp_delta: Linear-ramp scale for initial angles.
        adapt_gradient_shift: delta for ADAPT's central-difference gradient.
        spsa: Keyword arguments for ``SpsaSettings`` (validated there).
    """

    mixers: tuple[str, ...]
    optimizers: tuple[str, ...]
    layers: tuple[int, ...]
    x_penalty_scales: tuple[float, ...]
    ramp_delta: float
    adapt_gradient_shift: float
    spsa: dict[str, Any]


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
    mixer_study: MixerStudyConfig | None = None
    denoise: DenoiseConfig | None = None
    experiment: ExperimentConfig | None = None
    pilot: PilotConfig | None = None
    source_chain: tuple[Path, ...] = ()
    """Every file read for this config, parent first (just ``source_path`` without extends)."""

    def require_pilot(self) -> PilotConfig:
        """Return the ``pilot`` section, raising ConfigError if the file has none."""
        if self.pilot is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'pilot' section")
        return self.pilot

    def require_experiment(self) -> ExperimentConfig:
        """Return the ``experiment`` section, raising ConfigError if the file has none."""
        if self.experiment is None:
            raise ConfigError(f"{self.source_path}: this command needs an 'experiment' section")
        return self.experiment

    def require_denoise(self) -> DenoiseConfig:
        """Return the ``denoise`` section, raising ConfigError if the file has none."""
        if self.denoise is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'denoise' section")
        return self.denoise

    def require_mixer_study(self) -> MixerStudyConfig:
        """Return the ``mixer_study`` section, raising ConfigError if the file has none."""
        if self.mixer_study is None:
            raise ConfigError(f"{self.source_path}: this command needs a 'mixer_study' section")
        return self.mixer_study

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


def _read_with_extends(source: Path, seen: tuple[Path, ...]) -> tuple[dict[str, Any], list[Path]]:
    """Read a YAML mapping, resolving ``extends: <parent.yaml>`` recursively (ADR-025).

    The parent path is relative to the extending file. Mappings are merged recursively, the
    extending file winning; any other value (lists included) replaces the parent's.
    """
    if source in seen:
        raise ConfigError(f"{source}: circular 'extends' chain")
    with source.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: expected a YAML mapping at the top level")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw, [source]
    if not isinstance(parent, str):
        raise ConfigError(f"{source}: 'extends' must be a file path")
    base, chain = _read_with_extends((source.parent / parent).resolve(), (*seen, source))
    return _deep_merge(base, raw), [*chain, source]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_chain_text(config: Config) -> str:
    """The exact text of the config file, or of every file in its ``extends`` chain.

    For a single file this is its text unchanged; for a chain, each file's text is preceded by
    a ``# --- <path> ---`` header, parent first.
    """
    chain = config.source_chain or (config.source_path,)
    if len(chain) == 1:
        return chain[0].read_text(encoding="utf-8")
    return "".join(f"# --- {p} ---\n{p.read_text(encoding='utf-8')}" for p in chain)


def load_config(path: Path | str) -> Config:
    """Load and validate a YAML config file (optionally extending another, see ``extends``).

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration, with paths resolved to absolute paths.

    Raises:
        ConfigError: If a required field is missing or the values are inconsistent.
    """
    source = Path(path).resolve()
    raw, chain = _read_with_extends(source, ())

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
    mixer_study = None
    if "mixer_study" in raw:
        mixer_study = _parse_mixer_study(_require(raw, "mixer_study", dict, source), source)
    denoise = None
    if "denoise" in raw:
        section = _require(raw, "denoise", dict, source)
        method = _require(section, "experiment_method", str, source)
        if method not in DENOISE_METHODS:
            raise ConfigError(
                f"{source}: denoise.experiment_method must be one of {DENOISE_METHODS}"
            )
        count = _require(section, "evaluation_instances_per_size", int, source)
        if isinstance(count, bool) or count < 1:
            raise ConfigError(f"{source}: denoise.evaluation_instances_per_size must be >= 1")
        denoise = DenoiseConfig(experiment_method=method, evaluation_instances_per_size=count)
    pilot = None
    if "pilot" in raw:
        pilot = _parse_pilot(_require(raw, "pilot", dict, source), source)
    experiment = None
    if "experiment" in raw:
        experiment = _parse_experiment(
            _require(raw, "experiment", dict, source), len(universe.tickers), source
        )
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
        mixer_study=mixer_study,
        denoise=denoise,
        experiment=experiment,
        pilot=pilot,
        source_chain=tuple(chain),
    )


def _parse_pilot(raw: dict[str, Any], source: Path) -> PilotConfig:
    def fail(message: str) -> ConfigError:
        return ConfigError(f"{source}: pilot.{message}")

    def positive_ints(key: str) -> tuple[int, ...]:
        values = _require(raw, key, list, source)
        if not values or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
        ):
            raise fail(f"{key} must be a non-empty list of positive integers")
        return tuple(values)

    def positive_floats(key: str) -> tuple[float, ...]:
        values = _require(raw, key, list, source)
        if not values or any(
            isinstance(v, bool) or not isinstance(v, int | float) or v <= 0 for v in values
        ):
            raise fail(f"{key} must be a non-empty list of numbers > 0")
        return tuple(sorted(float(v) for v in values))

    def positive_int(key: str) -> int:
        value = _require(raw, key, int, source)
        if isinstance(value, bool) or value < 1:
            raise fail(f"{key} must be >= 1")
        return value

    skip = _require(raw, "skip_timing", dict, source)
    budget = _require(raw, "budget_hours", (int, float), source)
    if isinstance(budget, bool) or budget <= 0:
        raise fail("budget_hours must be > 0")
    return PilotConfig(
        timing_sizes=positive_ints("timing_sizes"),
        timing_repeats=positive_int("timing_repeats"),
        skip_timing={str(m): tuple(int(n) for n in ns) for m, ns in skip.items()},
        thread_test_sizes=positive_ints("thread_test_sizes"),
        accuracy_size=positive_int("accuracy_size"),
        accuracy_instances=positive_int("accuracy_instances"),
        accuracy_noise_levels=positive_floats("accuracy_noise_levels"),
        noise_ladder=positive_floats("noise_ladder"),
        min_noisy_levels=positive_int("min_noisy_levels"),
        budget_hours=float(budget),
    )


def _parse_experiment(raw: dict[str, Any], n_assets: int, source: Path) -> ExperimentConfig:
    def fail(message: str) -> ConfigError:
        return ConfigError(f"{source}: experiment.{message}")

    def int_list(key: str) -> tuple[int, ...]:
        values = _require(raw, key, list, source)
        if not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise fail(f"{key} must be a non-empty list of integers")
        return tuple(values)

    def choices(key: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
        values = tuple(_require(raw, key, list, source))
        if not values or any(v not in allowed for v in values) or len(set(values)) != len(values):
            raise fail(f"{key} must be distinct values from {allowed}")
        return values

    sizes = int_list("sizes")
    if any(n < 2 or n % 2 or n > n_assets for n in sizes):
        raise fail(f"sizes must be even integers in [2, {n_assets}]")
    noisy_sizes = int_list("noisy_sizes")
    if not set(noisy_sizes) <= set(sizes):
        raise fail("noisy_sizes must be a subset of sizes")
    levels = _require(raw, "noise_levels", list, source)
    if (
        not levels
        or any(isinstance(m, bool) or not isinstance(m, int | float) or m < 0 for m in levels)
        or len(set(levels)) != len(levels)
    ):
        raise fail("noise_levels must be distinct numbers >= 0")
    mixers = choices("mixers", STUDY_MIXERS)
    raw_settings = _require(raw, "mixer_settings", dict, source)
    settings: dict[str, MixerSetting] = {}
    for mixer in mixers:
        if mixer not in raw_settings or not isinstance(raw_settings[mixer], dict):
            raise fail(f"mixer_settings needs an entry for {mixer!r}")
        entry = raw_settings[mixer]
        optimizer, layers = entry.get("optimizer"), entry.get("layers")
        scale = entry.get("penalty_scale", 1.0)
        if optimizer not in STUDY_OPTIMIZERS:
            raise fail(f"mixer_settings.{mixer}.optimizer must be one of {STUDY_OPTIMIZERS}")
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
            raise fail(f"mixer_settings.{mixer}.layers must be an integer >= 1")
        if isinstance(scale, bool) or not isinstance(scale, int | float) or scale <= 0:
            raise fail(f"mixer_settings.{mixer}.penalty_scale must be > 0")
        settings[mixer] = MixerSetting(optimizer, layers, float(scale))
    count = _require(raw, "instances_per_size", int, source)
    if isinstance(count, bool) or count < 1:
        raise fail("instances_per_size must be >= 1")
    budgets = int_list("classical_sa_budgets")
    if any(b < 1 for b in budgets):
        raise fail("classical_sa_budgets must be positive")
    cost = _require(raw, "cost_model", dict, source)
    noiseless = _require(cost, "noiseless_seconds_per_eval", (int, float), source)
    noisy_raw = _require(cost, "noisy_seconds_per_eval", dict, source)
    noisy: dict[str, dict[int, float]] = {}
    for mixer, table in noisy_raw.items():
        if not isinstance(table, dict) or any(
            isinstance(v, bool) or not isinstance(v, int | float) or v <= 0 for v in table.values()
        ):
            raise fail(f"cost_model.noisy_seconds_per_eval.{mixer} must map N to seconds > 0")
        noisy[str(mixer)] = {int(n): float(v) for n, v in table.items()}
    exclusions_raw = raw.get("noisy_exclusions", {})
    if not isinstance(exclusions_raw, dict):
        raise fail("noisy_exclusions must map mixers to lists of sizes")
    exclusions: dict[str, tuple[int, ...]] = {}
    for mixer, ns in exclusions_raw.items():
        if mixer not in mixers or not isinstance(ns, list) or not set(ns) <= set(noisy_sizes):
            raise fail(f"noisy_exclusions.{mixer} must list noisy sizes of a configured mixer")
        exclusions[str(mixer)] = tuple(int(n) for n in ns)
    routing = raw.get("routing", "full_device")
    if routing not in ROUTING_MODES:
        raise fail(f"routing must be one of {ROUTING_MODES}")
    return ExperimentConfig(
        name=_require(raw, "name", str, source),
        sizes=sizes,
        noisy_sizes=noisy_sizes,
        denoising=choices("denoising", DENOISING_LEVELS),
        mixers=mixers,
        noise_levels=tuple(float(m) for m in levels),
        instances_per_size=count,
        backend=_require(raw, "backend", str, source),
        mixer_settings=settings,
        classical_sa_budgets=budgets,
        noiseless_seconds_per_eval=float(noiseless),
        noisy_seconds_per_eval=noisy,
        noisy_exclusions=exclusions,
        routing=routing,
    )


def _parse_mixer_study(raw: dict[str, Any], source: Path) -> MixerStudyConfig:
    def choices(key: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
        values = tuple(_require(raw, key, list, source))
        if not values or any(v not in allowed for v in values) or len(set(values)) != len(values):
            raise ConfigError(f"{source}: mixer_study.{key} must be distinct values from {allowed}")
        return values

    def positives(key: str, kind: type | tuple[type, ...]) -> tuple:
        values = _require(raw, key, list, source)
        if not values or any(
            isinstance(v, bool) or not isinstance(v, kind) or v <= 0 for v in values
        ):
            raise ConfigError(
                f"{source}: mixer_study.{key} must be a non-empty list of numbers > 0"
            )
        return tuple(values)

    def positive(key: str) -> float:
        value = _require(raw, key, (int, float), source)
        if isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{source}: mixer_study.{key} must be > 0, got {value!r}")
        return float(value)

    return MixerStudyConfig(
        mixers=choices("mixers", STUDY_MIXERS),
        optimizers=choices("optimizers", STUDY_OPTIMIZERS),
        layers=positives("layers", int),
        x_penalty_scales=tuple(float(c) for c in positives("x_penalty_scales", (int, float))),
        ramp_delta=positive("ramp_delta"),
        adapt_gradient_shift=positive("adapt_gradient_shift"),
        spsa=dict(_require(raw, "spsa", dict, source)),
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
