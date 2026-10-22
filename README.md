# NISQ-Folio

> **Status: Week 12 of 14.** The full pipeline, experiment runner, pre-registered analyses
> (ADR-026 noiseless, ADR-028 noisy), paper figures/tables and a results dashboard are
> implemented. Noiseless results of record: [docs/results/noiseless_main.md](docs/results/noiseless_main.md).
> The noisy grid is running; interim N = 8 results:
> [docs/results/noisy_n8_interim.md](docs/results/noisy_n8_interim.md). See
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
uv run python -m nisq_folio.experiments --config configs/main.yaml --estimate
uv run python -m nisq_folio.experiments --config configs/main.yaml --max-noise 0   # noiseless part
uv run python -m nisq_folio.experiments --config configs/main.yaml                 # everything else
```

The runtime benchmark includes noisy N=14 cases that can take hours on a laptop. To skip them,
remove 14 from `noise_benchmark.sizes` in a copy of the config.

The first data run downloads about 2,264 trading days for 30 tickers from Yahoo Finance into
`data/` (gitignored). Later runs read the checksummed cache. Add `--refresh` to re-download.

### Analysis, figures and dashboard

```bash
# Pre-registered noiseless tables T1-T6 (ADR-026):
uv run python -m nisq_folio.analysis --config configs/main.yaml --max-noise 0
# Pre-registered noisy analysis N1-N8 (ADR-028); complete cells only, safe while the grid runs:
uv run python -m nisq_folio.analysis.noise_report --config configs/main.yaml --superseded main
# Paper tables and figures (paper/generated/), then the PDF (Tectonic or any LaTeX):
uv run python paper/make_tables.py --config configs/main.yaml
uv run python paper/make_figures.py --config configs/main.yaml
# Read-only interactive dashboard at http://localhost:8501:
uv run streamlit run dashboard/app.py -- --config configs/main.yaml
```

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

Every reported number is generated by code from recorded runs; none is typed by hand.

1. **Environment:** `uv sync --locked` gives the exact package versions in `uv.lock`
   (Python 3.12, Qiskit 2.5.2, Qiskit Aer 0.17.2).
2. **Data:** `uv run python -m nisq_folio.data --config configs/base.yaml` downloads the price
   snapshot and verifies its SHA-256 (ADR-011). Never use `--refresh` to reproduce results:
   Yahoo values drift by about 1e-6 between downloads.
3. **Experiment of record:** `uv run python -m nisq_folio.experiments --config configs/main.yaml`.
   Use `--max-noise 0` for the noiseless part (minutes). The noisy grid takes hundreds of
   laptop-hours; it is resumable, so it can be stopped and restarted at any time.
4. **Analysis:** the commands under *Analysis, figures and dashboard* above.

**Provenance.** `results/experiments/<name>/manifest.json` holds the full config text (including
every file it `extends`), its SHA-256 fingerprint, and the data checksum. Each
`runs/<task>.json` records the git commit and dirty flag it was produced under. A directory refuses
results from a different config.

**Determinism.** All randomness derives from the root seed in the config through
`numpy.random.SeedSequence` streams keyed by task identity (ADR-004), so a single task re-run
alone gives the same result. Numerical libraries and thread scheduling can still introduce
last-digit differences across machines and operating systems.

**Decisions.** [DECISIONS.md](DECISIONS.md) lists every design choice (ADR-001 to ADR-029),
including the pre-registrations and their results, failures, and post-hoc changes.

## Citation

If you use this work, please cite it via [CITATION.cff](CITATION.cff) (GitHub shows a
"Cite this repository" button once the repo is public).

## License

[MIT](LICENSE) © 2026 Yuvaraj Guliani
