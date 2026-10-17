"""Running one experiment task (DECISIONS.md ADR-024).

An :class:`ExperimentContext` loads the frozen data snapshot, splits it (ADR-014), fits the
universe-level covariances (ADR-022), and draws the evaluation instances once per process.
:func:`run_task` then executes one task and returns a JSON-serializable record.

QAOA tasks use the protocol fixed by earlier ADRs: shots / budget / COBYLA radius from the
``qaoa`` section (ADR-020), ramp Delta, SPSA and ADAPT settings from ``mixer_study`` (ADR-021),
and the per-mixer optimizer / depth / penalty from ``experiment.mixer_settings``. m = 0 runs on
the exact NumPy simulator, m > 0 on the noisy density-matrix pipeline (ADR-023); the two are
identical at m = 0.

Out-of-sample check: the most probable *feasible* selection of the final distribution (the
answer one would deploy) is scored on 2021-2023 returns: realized variance and objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nisq_folio.baselines.brute_force import enumerate_objectives
from nisq_folio.baselines.greedy import greedy
from nisq_folio.baselines.markowitz import markowitz_rounding
from nisq_folio.baselines.simulated_annealing import AnnealingSettings, simulated_annealing
from nisq_folio.config import Config
from nisq_folio.data.prices import PriceMetadata, load_or_download
from nisq_folio.data.returns import annualized_moments, simple_returns, split_returns
from nisq_folio.denoise.rmt import denoised_covariance
from nisq_folio.experiments.grid import Task
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.mixers.variants import VariantSettings, run_variant
from nisq_folio.noise.density import ReferenceRates
from nisq_folio.noise.runner import run_noisy_variant
from nisq_folio.problem import PortfolioProblem
from nisq_folio.provenance import GitState
from nisq_folio.qaoa.metrics import SolutionLandscape
from nisq_folio.qaoa.optimizers import SpsaSettings
from nisq_folio.qubo.encoding import all_bitstrings
from nisq_folio.qubo.penalty import sufficient_penalty
from nisq_folio.seeding import child_rng


@dataclass
class ExperimentContext:
    """Everything a task needs that is shared across tasks (built once per process)."""

    config: Config
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    covariances: dict[str, pd.DataFrame | None]
    instances: dict[int, list[tuple[str, ...]]]
    rates: ReferenceRates
    data: PriceMetadata
    max_parallel_threads: int = 0

    @classmethod
    def build(
        cls, config: Config, git: GitState, max_parallel_threads: int = 0
    ) -> ExperimentContext:
        """Load data and derived inputs for ``config`` (no network if the cache exists)."""
        experiment, problem_cfg = config.require_experiment(), config.require_problem()
        u = config.universe
        prices, data = load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir, git)
        in_sample, out_of_sample = split_returns(simple_returns(prices), problem_cfg.in_sample_end)
        covariances: dict[str, pd.DataFrame | None] = {"none": None}
        if "rmt_clip" in experiment.denoising:
            covariances["rmt_clip"] = denoised_covariance(in_sample, rule="clip")[0]
        instances = {
            n: sample_instance_tickers(u.tickers, n, experiment.instances_per_size, config.seed)
            for n in experiment.sizes
        }
        needs_noise = any(m > 0 for m in experiment.noise_levels)
        rates = (
            ReferenceRates.from_backend(experiment.backend)
            if needs_noise
            else ReferenceRates(0.0, 0.0, 0.0)
        )
        return cls(
            config,
            in_sample,
            out_of_sample,
            covariances,
            instances,
            rates,
            data,
            max_parallel_threads,
        )

    def problem(self, task: Task) -> tuple[tuple[str, ...], PortfolioProblem]:
        """The task's tickers and problem (with the task's covariance estimator)."""
        tickers = self.instances[task.n][task.instance]
        p = self.config.require_problem()
        problem = build_problem(
            self.in_sample, tickers, task.n // 2, p.risk_aversion, self.covariances[task.denoising]
        )
        return tickers, problem

    def out_of_sample_score(
        self, tickers: tuple[str, ...], selection: list[int]
    ) -> dict[str, float]:
        """Realized variance and objective of an equal-weight selection on 2021-2023 data."""
        mu, sigma = annualized_moments(self.out_of_sample.loc[:, list(tickers)])
        w = np.zeros(len(tickers))
        w[selection] = 1.0 / len(selection)
        variance = float(w @ sigma.to_numpy() @ w)
        lam = self.config.require_problem().risk_aversion
        return {
            "realized_variance": variance,
            "realized_objective": lam * variance - float(mu.to_numpy() @ w),
        }


def variant_settings(config: Config, mixer: str) -> VariantSettings:
    """The fixed QAOA protocol for ``mixer`` (module docstring)."""
    qaoa, study = config.require_qaoa(), config.require_mixer_study()
    chosen = config.require_experiment().mixer_settings[mixer]
    return VariantSettings(
        mixer=mixer,
        optimizer=chosen.optimizer,
        layers=chosen.layers,
        shots=qaoa.shots,
        max_evaluations=qaoa.max_evaluations,
        rhobeg=qaoa.rhobeg,
        ramp_delta=study.ramp_delta,
        penalty_scale=chosen.penalty_scale,
        spsa=SpsaSettings(**study.spsa),
        adapt_gradient_shift=study.adapt_gradient_shift,
    )


def run_task(task: Task, ctx: ExperimentContext) -> dict[str, Any]:
    """Execute one task and return its result record (without provenance; see ``store``)."""
    start = time.perf_counter()
    tickers, problem = ctx.problem(task)
    rng = child_rng(ctx.config.seed, *task.seed_keys())
    record: dict[str, Any] = {
        "task": task.__dict__,
        "task_id": task.task_id,
        "tickers": list(tickers),
    }
    if task.kind == "classical":
        record.update(_classical(problem, tickers, rng, ctx))
    else:
        record.update(_qaoa(task, problem, tickers, rng, ctx))
    record["wall_seconds"] = time.perf_counter() - start
    return record


def _classical(
    problem: PortfolioProblem,
    tickers: tuple[str, ...],
    rng: np.random.Generator,
    ctx: ExperimentContext,
) -> dict[str, Any]:
    subsets, values = enumerate_objectives(problem)
    best = int(np.argmin(values))
    optimum = [int(i) for i in subsets[best]]
    g, mr = greedy(problem), markowitz_rounding(problem)
    annealing = {}
    for budget in ctx.config.require_experiment().classical_sa_budgets:
        sol = simulated_annealing(problem, rng, AnnealingSettings(n_steps=budget))
        annealing[str(budget)] = {
            "objective": sol.objective,
            "selection": list(sol.selection),
            "n_evaluations": sol.n_evaluations,
        }
    return {
        "f_star": float(values[best]),
        "f_worst": float(values.max()),
        "optimum": optimum,
        "greedy": {"objective": g.objective, "selection": list(g.selection)},
        "markowitz_rounding": {"objective": mr.objective, "selection": list(mr.selection)},
        "simulated_annealing": annealing,
        "optimum_out_of_sample": ctx.out_of_sample_score(tickers, optimum),
    }


def _qaoa(
    task: Task,
    problem: PortfolioProblem,
    tickers: tuple[str, ...],
    rng: np.random.Generator,
    ctx: ExperimentContext,
) -> dict[str, Any]:
    assert task.mixer is not None
    settings = variant_settings(ctx.config, task.mixer)
    landscape = SolutionLandscape.from_problem(problem)
    bound = sufficient_penalty(problem, ctx.config.require_qubo().penalty_margin)
    if task.noise_level == 0.0:
        run = run_variant(problem, bound, settings, rng, landscape)
    else:
        seed_transpiler = int(rng.integers(2**31))
        run = run_noisy_variant(
            problem,
            bound,
            settings,
            task.noise_level,
            ctx.rates,
            rng,
            landscape,
            backend=ctx.config.require_experiment().backend,
            seed_transpiler=seed_transpiler,
            max_parallel_threads=ctx.max_parallel_threads,
        )
    probs = run.distribution
    assert probs is not None
    feasible_probs = np.where(landscape.feasible, probs, -1.0)
    mode = int(np.argmax(feasible_probs))
    selection = [int(i) for i in np.flatnonzero(all_bitstrings(problem.n)[mode])]
    return {
        "run": run.as_dict(),
        "mode_feasible": {
            "selection": selection,
            "probability": float(probs[mode]),
            "objective": float(landscape.objective[mode]),
            "is_optimal": bool(landscape.optimal[mode]),
            "out_of_sample": ctx.out_of_sample_score(tickers, selection),
        },
    }
