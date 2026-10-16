# NISQ-Folio — Project Plan

## 1. Research question

> Does a **compound pipeline** of RMT covariance denoising → QUBO encoding → adaptive-mixer QAOA
> beat classical baselines on cardinality-constrained portfolio selection? And as simulated NISQ
> noise increases, **where is the crossover point** at which any advantage disappears?

Sub-questions (one per pipeline component):
1. Does RMT denoising of the covariance matrix improve the quality of the selected portfolio
   (in-sample and out-of-sample) for both the quantum and classical solvers?
2. Do adaptive / constraint-preserving mixers improve QAOA's probability of returning feasible,
   near-optimal selections compared with the standard X mixer?
3. How do these effects interact, and how do they degrade with noise level and problem size N?

**Framing caveat (to keep the claims honest).** At N ≤ 14 the exact optimum is cheap to compute
by brute force (at most C(14,7) = 3,432 subsets). So "beat classical" cannot mean "find better
solutions than the exact optimum." Success is measured against the *heuristic* baselines and
as a gap-to-optimum, and is reported as evidence about trends, not as a practical speed-up. The
exact comparison metric is fixed in an ADR before any results are generated (Milestone W5).

## 2. Fixed scope

| Item | Value |
|---|---|
| Universe | 30 US stocks across 6 sectors, 5 each (ADR-010; list in `configs/base.yaml`) |
| Historical window | 2015-01-01 to 2023-12-31 (daily) |
| Problem sizes | N ∈ {8, 10, 12, 14} assets |
| Cardinality | K = N/2 (i.e. 4, 5, 6, 7) |
| Classical baselines | (1) brute force (exact), (2) simulated annealing, (3) greedy, (4) continuous Markowitz + rounding to K |
| Quantum backend | Qiskit Aer simulation, with noise models from fake IBM backends + parameterised noise levels |

Out of scope: real quantum hardware runs, anything that costs money (ADR-008), transaction costs, multi-period rebalancing, and
universes larger than 30 stocks.

## 3. Experiment design (factorial)

Factors:
- **Denoising**: e.g. none vs RMT (variants TBD)
- **Mixer**: e.g. X mixer vs constraint-preserving vs adaptive (set TBD)
- **Noise level**: noiseless plus a sweep of increasing noise (levels TBD)
- **N**: {8, 10, 12, 14}

Replication: **≥ 30 seeds per cell**. All seeds are derived from one root seed (ADR-004).

The exact factor levels are not yet fixed. They will be recorded in an ADR before
implementation of `experiments/`. The same N-specific problem instances (asset subsets drawn
from the universe) are shared across all cells, so comparisons are paired.

## 4. Compute-budget constraint

The full grid is approximately **4,300 optimizations**. Noisy QAOA simulation cost grows
steeply with qubit count: exact density-matrix simulation needs 4^N entries, about 4.3 GB at
N=14 (see ADR-007). **Decided in W4 from a measured benchmark (ADR-018, ADR-007 resolution):**
noisy optimization runs at N ∈ {8, 10, 12} with exact density-matrix simulation. One noisy
N=14 evaluation took 7.1 hours on this laptop. N=14 is therefore noiseless only. Noiseless runs
and classical baselines cover all N.

The number of noisy factor levels must fit an explicit laptop-hour budget. At about 100
evaluations per optimization, one 30-seed noisy cell costs roughly 1.6 h (N=8), 1.7 h (N=10)
and 6.6 h (N=12). The budget and the resulting noisy grid are fixed in an ADR before W12.

## 5. Milestones (14 weeks)

| Week | Milestone | Deliverable |
|---|---|---|
| W1 ✅ | Scaffolding, environment, CI | This repo; green CI |
| W2 ✅ | Data pipeline | `data/`: ticker list ADR, download + cache, returns, covariance; tests |
| W3 ✅ | Classical baselines | `baselines/`: all four solvers, validated against brute force on small N |
| W4 ✅ | QUBO + noise benchmark | `qubo/` with penalty-weight ADR; noisy-simulation runtime benchmark → finalize ADR-007 |
| W5 ✅ | QAOA core | `qaoa/` with X mixer, noiseless; metric definitions ADR |
| W6 ✅ | Mixers | `mixers/`: constraint-preserving + adaptive mixers |
| W7 | RMT denoising | `denoise/`: Marchenko–Pastur clipping; property tests (PSD, trace preservation) |
| W8 | Noise models | `noise/`: fake-backend + parameterised noise sweep |
| W9 | Experiment runner | `experiments/`: config-driven grid, provenance logging (config + git hash), resumable |
| W10 | Pilot run | Small grid end-to-end; sanity checks; fix factor levels |
| W11 | Full noiseless grid | All N, all seeds |
| W12 | Noisy grid | Restricted N per ADR-007 |
| W13 | Analysis + dashboard | `analysis/` statistics, crossover estimate with CIs; Streamlit dashboard |
| W14 | Write-up + release | Report, README reproduction guide, tagged release |

## 6. Success criteria

The project succeeds if it produces a **reproducible, honest answer** to the research
question, whether that answer is positive or negative. Concretely:

1. **Reproducibility:** a fresh clone + `uv sync --locked` + documented commands regenerates
   every reported number and figure. Every result file records its config and git commit hash.
2. **Correctness:** CI is green on Ubuntu and Windows. Core modules have ≥ 85% coverage with
   property-based tests (e.g. QUBO energy equals portfolio objective on feasible bitstrings;
   denoised covariance stays positive semidefinite).
3. **Statistical rigour:** every pipeline-vs-baseline comparison is reported with effect sizes
   and confidence intervals over ≥ 30 seeds, using paired designs where possible.
4. **A crossover answer:** an estimated noise level (with uncertainty) at which the pipeline's
   quality metric falls to the best heuristic baseline. If no advantage exists even noiselessly,
   that is reported as the finding.
5. **Traceability:** every non-trivial choice is documented in `DECISIONS.md`.
