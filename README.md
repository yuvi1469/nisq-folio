# NISQ-Folio

> **Status: scaffolding.** No finance or quantum logic is implemented yet. See
> [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope and milestones.

## Overview

NISQ-Folio is a reproducible benchmark of a compound hybrid quantum-classical pipeline for
**cardinality-constrained portfolio selection** (choose exactly K of N assets):

```
returns -> RMT covariance denoising -> QUBO encoding -> adaptive-mixer QAOA -> portfolio
```

It asks two questions: does this pipeline beat classical baselines, and at what level of
simulated NISQ noise does any advantage disappear (the *noise crossover point*)? Every
architectural and scientific choice is recorded in [DECISIONS.md](DECISIONS.md).

## Install

Requires [uv](https://docs.astral.sh/uv/). uv will fetch Python 3.12 automatically if needed.

```bash
git clone https://github.com/<user>/nisq-folio.git
cd nisq-folio
uv sync --locked
```

`--locked` installs exactly the versions in `uv.lock` and fails if the lock is out of date.

## Quickstart

```bash
uv run pytest
uv run ruff check .
```

_Experiment entry points will be documented here once implemented._

## Project structure

```
src/nisq_folio/
  data/         market data download, caching, returns and covariance inputs
  denoise/      random-matrix-theory covariance denoising
  qubo/         QUBO / Ising encoding of the cardinality-constrained problem
  mixers/       QAOA mixers, including adaptive mixers
  qaoa/         QAOA circuits and the classical angle optimizer
  noise/        NISQ noise models (Aer + fake IBM backends)
  baselines/    brute force, simulated annealing, greedy, Markowitz + rounding
  experiments/  factorial experiment runner (logs config + git commit per run)
  analysis/     statistics and figures
tests/          pytest suite (one module per package, minimum)
configs/        YAML configuration; all paths and seeds live here
notebooks/      exploration only (not a source of reported results)
dashboard/      Streamlit results dashboard
docs/           long-form documentation
results/        experiment outputs (gitignored, regenerable)
```

## Reproducing results

_To be written once the experiment runner exists._ The intended contract:

1. `uv sync --locked` to get the exact environment.
2. Run an experiment from a config in `configs/`.
3. Each run writes its resolved config and the git commit hash next to its results in `results/`.

## Citation

If you use this work, please cite it via [CITATION.cff](CITATION.cff) (GitHub shows a
"Cite this repository" button once the repo is public).

## License

[MIT](LICENSE) © 2026 Yuvaraj Guliani
