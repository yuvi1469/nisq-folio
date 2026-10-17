# NISQ-Folio

> **Status: Week 8 of 14.** Data pipeline, four validated classical baselines, RMT covariance
> denoising with an out-of-sample evaluation, QUBO/Ising encoding, a noisy-simulation benchmark,
> noiseless QAOA with X, XY-ring and adaptive (ADAPT) mixers, and verified gate-level circuits with
> a tunable noise model, and a resumable experiment runner are implemented. The pilot run is next. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope and milestones.

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
git clone https://github.com/yuvi1469/nisq-folio.git
cd nisq-folio
uv sync --locked
```

`--locked` installs exactly the versions in `uv.lock` and fails if the lock is out of date.

Everything runs on a laptop at zero cost: open-source packages, local simulation only (no paid
quantum hardware or cloud compute), and free public market data. On Windows, keep the clone
outside OneDrive-synced folders such as Desktop and Documents (see DECISIONS.md ADR-009).

## Quickstart

```bash
uv run pytest        # tests (no network access needed)
uv run ruff check .  # lint

# Download (first run) or load from cache the 30-stock universe, and print a summary:
uv run python -m nisq_folio.data --config configs/base.yaml

# Validate the classical baselines against the exact optimum (writes a run record to results/):
uv run python -m nisq_folio.baselines --config configs/base.yaml

# Noisy-simulation runtime benchmark (slow: see DECISIONS.md ADR-018 before running):
uv run python -m nisq_folio.noise --config configs/base.yaml

# Noiseless QAOA tuning study with the pre-registered selection rule (about 3 minutes):
uv run python -m nisq_folio.qaoa --config configs/base.yaml

# Noiseless mixer study (X vs XY ring vs ADAPT; about 15 minutes):
uv run python -m nisq_folio.mixers --config configs/base.yaml

# Out-of-sample evaluation of covariance estimators, classical only (about 20 seconds):
uv run python -m nisq_folio.denoise --config configs/base.yaml

# The factorial experiment: predict hours first, then run (resumable; safe to interrupt):
uv run python -m nisq_folio.experiments --config configs/base.yaml --estimate
uv run python -m nisq_folio.experiments --config configs/base.yaml --workers 3
```

The runtime benchmark includes noisy N=14 cases that can take hours on a laptop. To skip them,
remove 14 from `noise_benchmark.sizes` in a copy of the config.

The first data run downloads about 2,264 trading days for 30 tickers from Yahoo Finance into
`data/` (gitignored). Later runs read the checksummed cache. Add `--refresh` to re-download.

_Experiment entry points will be documented here once implemented._

## Project structure

```
src/nisq_folio/
  config.py     YAML config loading and validation
  problem.py    the cardinality-constrained objective every solver minimizes
  calibration.py  risk-aversion calibration rule
  seeding.py    reproducible per-component random streams
  provenance.py git commit / dirty-state capture for every artifact
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
