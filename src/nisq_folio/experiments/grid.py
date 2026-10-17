"""Expansion of the experiment config into individual, independently resumable tasks (ADR-024).

Two kinds of task:
    ``classical``  one per (N, instance, denoising): exact optimum and the classical baselines
    ``qaoa``       one per (N, instance, denoising, mixer, noise level m)

Noise levels m > 0 are expanded only for N in ``noisy_sizes``; every other N runs m = 0 only.
Each task has a stable string id (its result file name) and a seed derived from its identity
alone, so results do not depend on task order, on parallelism, or on which other cells exist.
Tasks are ordered cheapest first (classical, then noiseless, then noisy by N), so partial runs
already contain complete low-cost cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from nisq_folio.config import DENOISING_LEVELS, STUDY_MIXERS, ExperimentConfig
from nisq_folio.seeding import STREAM_EXPERIMENT

_KIND_CODE = {"classical": 0, "qaoa": 1}


@dataclass(frozen=True, order=True)
class Task:
    """One unit of work. ``mixer`` is None and ``noise_level`` is 0 for classical tasks."""

    kind: str
    n: int
    instance: int
    denoising: str
    mixer: str | None = None
    noise_level: float = 0.0

    @property
    def task_id(self) -> str:
        """Stable identifier, used as the result file name."""
        base = f"{self.kind}__N{self.n:02d}__i{self.instance:03d}__{self.denoising}"
        if self.kind == "classical":
            return base
        return f"{base}__{self.mixer}__m{self.noise_level:g}"

    def seed_keys(self) -> tuple[int, ...]:
        """Integer keys for ``child_rng(root, *keys)``, derived only from the task identity.

        The noise level enters as round(m * 1e6), so m = 0.25 and m = 0.5 never collide.
        """
        mixer_code = STUDY_MIXERS.index(self.mixer) if self.mixer else 0
        return (
            STREAM_EXPERIMENT,
            _KIND_CODE[self.kind],
            self.n,
            self.instance,
            DENOISING_LEVELS.index(self.denoising),
            mixer_code,
            round(self.noise_level * 1_000_000),
        )


def expand_tasks(experiment: ExperimentConfig) -> list[Task]:
    """Return every task of the experiment, cheapest first (module docstring)."""
    classical, noiseless, noisy = [], [], []
    for n in experiment.sizes:
        levels = experiment.noise_levels if n in experiment.noisy_sizes else (0.0,)
        if 0.0 not in levels:
            levels = (0.0, *levels)  # the noiseless reference is always part of a cell's sweep
        for instance in range(experiment.instances_per_size):
            for denoising in experiment.denoising:
                classical.append(Task("classical", n, instance, denoising))
                for mixer in experiment.mixers:
                    for level in levels:
                        task = Task("qaoa", n, instance, denoising, mixer, float(level))
                        (noiseless if level == 0.0 else noisy).append(task)
    noisy.sort(key=lambda t: (t.n, t.noise_level, t.mixer or "", t.instance, t.denoising))
    return classical + noiseless + noisy
