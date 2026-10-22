---
title: 'NISQ-Folio: a pre-registered, reproducible benchmark of noisy QAOA for cardinality-constrained portfolio selection'
tags:
  - Python
  - quantum computing
  - QAOA
  - portfolio optimization
  - random matrix theory
  - reproducibility
authors:
  - name: Yuvaraj Guliani
    equal-contrib: true
    affiliation: 1
  - name: Kallol Das
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: S P Jain School of Global Management
    index: 1
date: 22 October 2026
bibliography: paper.bib
---

<!-- DRAFT. Not submittable before mid-April 2027: JOSS requires more than six months of public,
iterative development history (the repository became public on 13 October 2026). -->

# Summary

Choosing exactly $K$ of $N$ assets to balance expected return against risk is a hard
combinatorial problem that is often proposed as an early application of quantum computers. In
the usual recipe, the covariance matrix is estimated (and possibly cleaned), the problem is
written as a quadratic unconstrained binary optimization (QUBO) [@lucas2014; @markowitz1952], and
the quantum approximate optimization algorithm (QAOA) [@farhi2014qaoa] samples candidate
portfolios. Whether this recipe helps on today's noisy hardware is an empirical question, and
answers are easily distorted by hand-tuned settings, unpaired comparisons, or success metrics
that blind guessing can also meet.

NISQ-Folio is a Python package that runs this whole pipeline as a controlled experiment:
random-matrix-theory (RMT) covariance cleaning [@laloux1999; @bun2017], a QUBO with a provably
exact cardinality penalty, QAOA with the standard X mixer, the constraint-preserving XY mixer
[@hadfield2019; @wang2020xy] started from a Dicke state [@bartschi2019dicke], and an adaptive
mixer [@zhu2022adapt], under a tunable noise model scaled from IBM Heron-class error rates,
against exhaustive search, greedy selection, simulated annealing [@kirkpatrick1983] and a rounded
Markowitz relaxation. It is aimed at researchers who need a fair, reproducible baseline before
claiming a quantum effect in portfolio optimization.

# Statement of need

Published QAOA portfolio studies [@egger2020; @hodson2019; @brandhofer2022; @herman2023] differ in
problem definition, penalty choice, noise model, and metric, and rarely fix their analysis before
seeing results. NISQ-Folio provides three things that are hard to assemble from general-purpose
libraries:

1. **A fair success metric.** At simulable sizes a QAOA run draws more samples than there are
   bitstrings, so "the best sampled solution is optimal" is met by blind guessing. NISQ-Folio
   scores the final distribution exactly by its amplification over uniform random *feasible*
   guessing, $\mathcal{A} = p_{\mathrm{opt}} \binom{N}{K} / m$, where $m$ is the number of
   optimal selections. The noise crossover is the noise scale at which the median
   $\mathcal{A}$ falls to 1, reported with paired bootstrap intervals.
2. **Budget-matched classical baselines.** Every QAOA cell is paired by instance with exact,
   greedy, annealing and relaxation baselines on the same inputs, and sampling budgets are
   counted, including the shots an adaptive mixer spends on choosing operators.
3. **Pre-registration and provenance by construction.** Every scientific choice (risk aversion,
   penalty rule, tuning rule, noise levels, analysis) is a numbered decision record committed
   before the corresponding result exists. Every result file carries its configuration
   fingerprint, data checksum and git commit.

# State of the field

General QAOA frameworks such as Qiskit [@javadiabhari2024qiskit] and OpenQAOA [@sharma2022openqaoa]
provide circuits, optimizers and backends, and Qiskit's finance tooling formulates portfolio
problems. They are building blocks rather than experiments: they do not prescribe paired
instance designs, exact feasibility-aware metrics, penalty weights with guarantees, or
pre-registered analyses. NISQ-Folio builds on Qiskit and Qiskit Aer for transpilation and exact
density-matrix simulation instead of re-implementing them. It contributes the experimental layer
that is specific to the research question, which would not fit the scope of a general SDK.

# Software design

The package separates stages that are commonly entangled. `data` freezes a checksummed price
snapshot; `denoise` implements Marchenko–Pastur clipping [@marchenko1967] with a Ledoit–Wolf
reference [@ledoit2004]; `qubo` builds the encoding and the penalty
$P > \max_{m \ne K} (f_{\mathrm{ub}} - L_m)/(m-K)^2$, which guarantees that the QUBO minimum is
feasible without knowing the optimum; `qaoa` and `mixers` provide a fast NumPy statevector path
for noiseless runs; and `noise` provides gate-level circuits simulated as density matrices.

Several trade-offs follow from the research question:

- **Exact distributions over sampled estimates for scoring.** Optimization uses finite shots, as
  on hardware, but the final metric is computed from the exact output distribution. This removes
  scoring noise from the comparison, at the cost of limiting noisy runs to about 12 qubits.
- **One noise knob.** Noise is a single scale factor on calibrated median error rates
  (depolarizing gates, bit-flip readout), so the crossover is a single interpretable number. The
  model omits relaxation and crosstalk; this is a stated limitation, not a hidden one.
- **Deterministic routing.** Noisy circuits are transpiled onto a fixed connected $N$-qubit region
  of the device graph, which keeps memory bounded. A robustness analysis compares this with
  full-device routing.
- **Seeds as data.** All randomness derives from one root seed through `numpy.random.SeedSequence`
  streams keyed by task identity, so any task can be re-run alone and reproduce bit-for-bit.
  Noisy runs at zero noise reproduce the noiseless path to $10^{-15}$, which a test enforces.
- **Resumable, append-only experiments.** A grid of thousands of tasks runs for hundreds of
  laptop-hours. Results are written atomically one file per task, and a directory refuses results
  produced under a different configuration.

The test suite (about 400 tests, including Hypothesis property tests) runs on Linux and Windows
in continuous integration with a coverage floor of 85%. A read-only Streamlit dashboard and
generated paper tables and figures read the same result files.

# Research impact statement

NISQ-Folio produced the results of an accompanying research article (in preparation) on 2,400
paired tasks. In noiseless simulation, the XY and adaptive mixers beat random feasible guessing
at every size from 8 to 14 assets, while the penalized X mixer never does. Simulated annealing
nevertheless needs two to four orders of magnitude fewer samples. Under noise at 8 assets, both
constraint-preserving mixers lose their advantage below current hardware error rates, the deeper
adaptive circuit first; this reversed one of the study's own pre-registered predictions. All
configurations, run records and the decision log are public, so each number can be regenerated
from a single command.

# AI usage disclosure

Generative AI (Anthropic Claude Opus 5.5, model ID claude-opus-5-5, used through Claude Code, October 2026) was
used to write most of the source code, tests, documentation, analysis scripts and drafts of this
paper and the accompanying article, under the authors' direction. Every scientific decision was
made by the authors and recorded before results existed. Verification: all code passes the
automated test suite, linting and continuous integration; numerical results are regenerated by
scripts from recorded runs rather than typed by hand; and every reference was checked against its
DOI or arXiv record. The authors reviewed, edited and validated all AI-assisted outputs and take
full responsibility for them.

# Acknowledgements

This work received no external funding. All computation ran on a personal laptop using free and
open-source software.

# References
