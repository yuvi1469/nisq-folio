# Architecture & Scientific Decision Records

Each non-trivial architectural or scientific choice gets a numbered ADR. ADRs are never
deleted. A reversed decision gets a new ADR that supersedes the old one.

Status values: **Proposed**, **Accepted**, **Superseded by ADR-NNN**.

---

## ADR-001: Python 3.12 as the single supported interpreter

- **Status:** Accepted (2026-10-13)
- **Context:** The machine has Python 3.12.14, 3.13.14, and 3.14.4. The quantum stack is the
  constraint: `qiskit-aer` ships compiled wheels per Python version, and its latest release
  (0.17.2, 2025-09-17) is more than a year older than the current Qiskit (2.5.2, 2026-08-13).
- **Evidence:** A seeded smoke test (noisy 3-qubit GHZ on `AerSimulator.from_backend(FakeBrisbane())`,
  plus an Aer `EstimatorV2` expectation value) passed on 3.12, 3.13, and 3.14. All three gave
  identical seeded counts.
- **Decision:** Use Python 3.12 (`requires-python = ">=3.12,<3.13"`, `.python-version = 3.12`).
- **Reasoning:** All three work today. 3.12 is the most conservative choice for a 14-week
  project that must stay reproducible:
  (a) Aer appears to be in maintenance mode, so the newest Python versions are the most likely to
  lose wheel support in any future Aer release;
  (b) 3.12 has the longest track record across the scientific stack;
  (c) it receives security fixes until Oct 2028.
  Restricting to one minor version means `uv.lock` describes exactly one environment.
- **Consequences:** Moving to 3.13+ later requires a new ADR plus a re-run of the smoke test.

## ADR-002: Dependency pinning strategy and the verified quantum stack

- **Status:** Accepted (2026-10-13)
- **Decision:**
  - Quantum stack pinned exactly: `qiskit==2.5.2`, `qiskit-aer==0.17.2`,
    `qiskit-ibm-runtime==0.50.0`. These were verified together at runtime (ADR-001). Runtime 0.50.0
    requires `qiskit>=2.3,<3`, and Aer 0.17.2 requires `qiskit>=1.1.0`.
  - Other runtime dependencies use compatible-release pins (`~=X.Y.Z`, i.e. patch updates only).
  - `uv.lock` pins every package, including transitive ones, exactly, with hashes. CI installs
    with `uv sync --locked`.
- **Known conflict resolved:** `streamlit 1.65.0` requires `pyarrow <26` and excludes `25.0.0`.
  `pyarrow` is therefore pinned `~=25.0.1`, not the then-latest 26.0.0.
- **Risk:** Aer 0.17.2 predates Qiskit 2.5. Our smoke test passed, but it does not cover every
  Aer feature. If an Aer/Qiskit incompatibility appears, the fix is to lower the `qiskit` pin
  (not to edit tests). Record any such change in a new ADR.
- **Note:** `qiskit-ibm-runtime` is used only for its offline fake backends. No IBM account or
  token is needed or used.

## ADR-003: Tooling — uv, hatchling, src layout, ruff

- **Status:** Accepted (2026-10-13)
- **Decision:** `uv` for environment and lock management. `hatchling` as the build backend.
  A `src/` layout. `ruff` for both linting and formatting.
- **Reasoning:** uv gives a cross-platform lock file and fast, reproducible installs. The src
  layout ensures tests run against the installed package, not stray files in the working
  directory. The ruff rule set includes `D` (docstrings, Google convention), `ANN` (type hints),
  and `NPY` (flags legacy global `np.random.*` usage). Together these enforce CLAUDE.md rules
  1–2 mechanically.

## ADR-004: Explicit seeding via `numpy.random.SeedSequence`

- **Status:** Accepted (2026-10-13)
- **Decision:** `configs/base.yaml` holds one root seed. Each experiment cell/replicate derives
  its own child seed with `SeedSequence(root).spawn(...)`, or with an explicit
  `SeedSequence(entropy=root, spawn_key=(cell, rep))`. Child seeds are passed as arguments to
  every stochastic component: NumPy `Generator`s, simulated annealing, QAOA initial angles,
  `seed_simulator`, `seed_transpiler`, and the optimizer.
- **Reasoning:** Global RNG state makes results depend on call order. Spawned seeds are
  statistically independent and reproducible from (root, cell, replicate) alone.

## ADR-005: Enforce 85% coverage from day one

- **Status:** Accepted (2026-10-13)
- **Decision:** `fail_under = 85` (branch coverage, measured on `nisq_folio`) in `pyproject.toml`,
  enforced locally and in CI.
- **Reasoning:** It is trivially met while modules are empty, so the gate never has to be
  retrofitted onto existing untested code. If a specific non-core module (e.g. `dashboard` glue)
  needs exclusion later, record it in an ADR. Do not lower the threshold silently.

## ADR-006: CI on Ubuntu and Windows

- **Status:** Accepted (2026-10-13)
- **Decision:** GitHub Actions runs `uv sync --locked`, `ruff check`, `ruff format --check`, and
  `pytest` on `ubuntu-latest` and `windows-latest`, for pushes and PRs to `main`.
- **Reasoning:** Development happens on Windows, and reviewers will most likely reproduce on
  Linux. Testing both catches path and wheel differences early.

## ADR-007: Restrict noisy optimization to smaller N

- **Status:** Accepted (2026-10-15). See "ADR-007 resolution" below (noisy N ∈ {8, 10, 12}).
- **Context:** One qubit per asset, so N ∈ {8, 10, 12, 14} means 8–14 qubits. An exact
  density-matrix noisy simulation stores 4^N complex128 entries (16 bytes each):
  N=10 → ~16.8 MB, N=12 → ~268 MB, N=14 → ~4.3 GB per density matrix. Each cost evaluation
  inside a QAOA optimization loop needs such a simulation, so the full noisy grid
  (~4,300 optimizations in total) is unlikely to be feasible at N=14 on a laptop.
- **Options to evaluate:** (a) noisy runs only for N ∈ {8, 10} or {8, 10, 12};
  (b) Aer's trajectory (statevector + sampled noise) method, which costs 2^N memory but adds
  shot noise; (c) fewer seeds per noisy cell (must stay ≥ 30 per the plan).
- **To decide:** benchmark per-evaluation wall time for each N and method, then fix the
  restriction here.

## ADR-008: Zero-cost constraint

- **Status:** Accepted (2026-10-13)
- **Context:** The project must be completed entirely free of cost (user requirement).
- **Decision and how each cost is avoided:**
  - **Software:** every dependency is open source and installed from PyPI.
  - **Quantum execution:** local Qiskit Aer simulation only. `qiskit-ibm-runtime` is used only for
    its offline fake backends (noise models), which need no IBM account. Real quantum hardware is
    out of scope (PROJECT_PLAN.md §2). Changing that would need a new ADR showing it stays free.
  - **Compute:** the experiments run on the developer's laptop. No paid cloud compute. This is
    one reason the noisy grid must be restricted (ADR-007).
  - **CI:** the repository is public. GitHub Actions on standard GitHub-hosted runners is free
    for public repositories. The workflow also sets `concurrency` (cancels superseded runs) and
    `timeout-minutes` as a safeguard. If the repo is ever made private, Actions minutes become
    metered (and Windows runners count at a higher rate), so revisit this ADR first.
  - **Market data:** `yfinance`, a free, unofficial wrapper around Yahoo Finance. It has no
    service guarantee and may be rate-limited or change. Mitigations: download once, cache
    locally under `data/`, and record the download date and a checksum. Raw data is gitignored
    and not redistributed in the public repo. The code that regenerates it is.
  - **Dashboard:** runs locally with `streamlit run`. Public hosting, if wanted, would use a free
    tier; not committed to here.

## ADR-009: Keep the working copy outside OneDrive

- **Status:** Accepted (2026-10-13)
- **Context:** The repo was first created under `OneDrive\Desktop`. OneDrive syncs every file,
  including the ~100-package `.venv`. That is slow and can lock files. It also tagged uv's shared
  cache files. The next environment build then failed with
  `os error 396: The cloud operation cannot be performed on a file with incompatible hardlinks`,
  because uv installs packages by hardlinking from that cache.
- **Decision:** The working copy lives at `C:\Users\yuvra\dev\nisq-folio`, which is not
  synced by OneDrive. GitHub is the backup. After the move, the uv cache was cleared (`uv cache clean`)
  and the environment rebuilt from `uv.lock`.
- **Consequences:** Nothing in the code depends on the location (CLAUDE.md rule 3: no hardcoded
  paths). Anyone cloning on Windows should avoid OneDrive-synced folders (noted in README).

### ADR-006 amendment (2026-10-13): pinned runner images and tool versions

The first CI run passed, but GitHub announced that `ubuntu-latest` moves to Ubuntu 26 from
2026-10-19. It also warned that `actions/checkout@v4` / `setup-uv@v6` run on deprecated Node 20.
Since `*-latest` labels can change without any commit, CI now pins `ubuntu-24.04` and
`windows-2025`. It uses `actions/checkout@v7` and `astral-sh/setup-uv@v10.3.0` (both node24; setup-uv has no moving major tag, so it is pinned exactly),
and pins uv `0.12.17` to match local development. Upgrades happen deliberately, in a commit.

## ADR-010: The 30-stock universe

- **Status:** Accepted (2026-10-13). Proposed by Claude; the user delegated the choice. Easy to revise:
  the list lives only in `configs/base.yaml`, and the cache key changes automatically.
- **Decision:** 5 stocks in each of 6 GICS sectors:

  | Sector | Tickers |
  |---|---|
  | Information Technology | AAPL, MSFT, NVDA, CSCO, ORCL |
  | Health Care | JNJ, UNH, PFE, MRK, ABT |
  | Financials | JPM, BAC, WFC, GS, MS |
  | Energy | XOM, CVX, COP, SLB, EOG |
  | Consumer Staples | PG, KO, PEP, WMT, COST |
  | Industrials | CAT, UNP, HON, LMT, UPS |

- **Selection rules (applied by hand):** large, liquid US companies, continuously listed under the
  same ticker for the whole window (2015-01-01 to 2023-12-31). Excluded on purpose:
  - Tickers that changed during the window (e.g. FB→META in 2022).
  - Mergers that redefine the company (e.g. UTX→RTX in 2020).
  - Stocks moved between sectors by the March 2023 GICS reclassification (e.g. V, MA moved IT→Financials).
  - Six sectors with five names each keeps sectors balanced.
- **Verification (actual results, 2026-10-13):**
  - The download returned all 30 tickers, 2,264 trading days from 2015-01-02 to 2023-12-29, with
    zero missing values.
  - All 30 sector labels match Yahoo Finance's own sector field. Yahoo uses its own
    names (e.g. "Consumer Defensive" for Consumer Staples), not GICS, so this is a consistency
    check, not an official GICS confirmation.
- **Known limitation: survivorship and hindsight bias.** Every name was chosen in 2026 knowing that it
  survived and stayed large through 2023. Absolute returns of this universe are therefore biased upward
  compared with what an investor in 2015 could have picked. This matters less for our research
  question, because every solver sees the same inputs. But all results must be described as "on this
  universe", not as evidence about the market in general.

## ADR-011: Market data source, adjustment, caching and reproducibility

- **Status:** Accepted (2026-10-13)
- **Source:** Yahoo Finance via `yfinance` with `auto_adjust=True` (daily closes adjusted for splits and
  dividends). Free (ADR-008), but unofficial: no service guarantee, possible rate limits, and the API may
  change.
- **Window semantics:** yfinance treats `end` as exclusive. `fetch_yfinance` adds one day, so the config's
  `end_date` is inclusive.
- **Validation:** a download is rejected, not repaired, if it has any of:
  - wrong or missing tickers;
  - duplicate or unsorted dates;
  - dates outside the window;
  - any missing value;
  - any non-positive or non-finite price.

  Filling gaps would silently give assets different sample periods, which biases the covariance.
- **Returns:** simple returns r_t = P_t/P_{t-1} − 1, because portfolio simple return is linear in weights.
  μ and Σ are annualized with A = 252 (sample covariance, ddof = 1). Implemented in `data/returns.py`.
- **Cache and provenance:** prices are stored as `data/prices_<key>.parquet` with a JSON sidecar. The key is
  the first 12 hex characters of a SHA-256 over (tickers in order, start, end). The sidecar records the
  request, actual first and last dates, row count, download time (UTC), yfinance version, git commit,
  a dirty flag, and the SHA-256 of the Parquet file. The checksum is verified on every load.
- **Reproducibility caveat (stated honestly):** Yahoo rescales the whole adjusted history whenever a later
  dividend or split occurs, and it rounds prices. So a download made on a different day will not be
  byte-identical, and its SHA-256 will differ. Returns are invariant to a constant rescaling (property-tested in
  `test_returns_invariant_to_rescaling_prices`), so re-downloads should give the same returns up to
  rounding, but this is not guaranteed by the source. Raw Yahoo data is not redistributed in this public
  repo (data/ is gitignored). Reported results will cite the download date and checksum of the snapshot used.
- **Measured (2026-10-13):** two downloads made seconds apart were identical. But downloads a few
  minutes apart differed in about 80% of cells, by at most 1.1e-6 relative in price. The
  difference is not a constant rescale. This is noise in what Yahoo serves; our pipeline is
  deterministic: writing the same frame gives identical Parquet bytes. The effect downstream was
  ≤ 1.6e-6 in daily returns and ≈ 1.3e-7 in annualized μ and Σ entries, which is negligible next to
  estimation error. It does rule out bit-for-bit reproduction from a fresh download.
- **Consequences:**
  - (a) The cached snapshot is the frozen input. Do not run `--refresh` once experiments start.
  - (b) Every experiment logs the snapshot's SHA-256 along with its config and git commit.
  - (c) A later robustness check should confirm that conclusions (e.g. which subset is optimal) do not change
    under perturbations of this size. Snapshot used from Week 2 on: downloaded 2026-10-13T06:59:43Z at commit
    `8b297f9`, SHA-256 `25b7e5a7…4267a91`.
- **Not decided here:** an in-sample / out-of-sample split for evaluating selected portfolios. That goes in a
  later ADR, before results are generated.

## ADR-012: Property-based testing with Hypothesis, run deterministically

- **Status:** Accepted (2026-10-13)
- **Decision:** add `hypothesis~=6.169.0` to the dev dependency group (free and open source). `tests/conftest.py`
  loads a profile with `derandomize=True`, so every run on every machine generates the same examples.
- **Reasoning:** CLAUDE.md rule 9 asks for meaningful property-based tests. Examples so far:
  - Returns compound back to the total price change.
  - Returns are invariant to price rescaling.
  - Σ is symmetric positive semidefinite and equals 252 × `np.cov`.
  - μ and Σ permute together with the assets.

  Derandomization keeps rule 1 (fixed seeds): a failure in CI reproduces locally. No global RNG state is set.

## ADR-013: Config paths resolve relative to the config file

- **Status:** Accepted (2026-10-13)
- **Decision:** relative paths inside a YAML config are resolved against the directory containing that file,
  so `configs/base.yaml` uses `../data` and `../results`. They are never resolved against the current working directory.
- **Reasoning:** results must not depend on where a command was launched from (CLAUDE.md rule 3). This
  replaces the "relative to the repository root" comment in the original scaffold, which code could not
  enforce without guessing where the root is.

## ADR-014: In-sample / out-of-sample split

- **Status:** Accepted (2026-10-13). Proposed by Claude; the user delegated the choice.
- **Decision:** a single fixed split. μ and Σ are estimated on returns from **2015-01-02 to
  2020-12-31** (1,510 trading days). Portfolios that solvers select are evaluated on
  **2021-01-04 to 2023-12-29** (753 trading days), data that no solver ever sees. Configured as
  `problem.in_sample_end`. Implemented in `data/returns.py::split_returns`.
- **Reasoning:**
  - A fixed split is simple to explain and to reproduce.
  - A rolling window would multiply the already-tight compute budget (ADR-007).
  - The 6-year in-sample window gives about 1,500 observations for at most 14 assets, so Σ is
    well conditioned. RMT denoising (W7) is tested on this window.
  - The out-of-sample period includes the 2022 drawdown, so it is not a single easy regime.
- **Limitation:** one split gives one out-of-sample period. Out-of-sample results are evidence
  about that period only. A rolling-window robustness check can be added later if budget allows.

## ADR-015: Problem definition and risk aversion λ = 4.5

- **Status:** Accepted (2026-10-13)
- **Decision:** select exactly K = N/2 of N assets, held in equal weights w = x/K, and minimize

      f(x) = λ wᵀΣw − μᵀw = (λ/K²) xᵀΣx − (1/K) μᵀx,   x ∈ {0,1}ᴺ, Σx = K

  (`problem.py`). Every solver, classical or quantum, minimizes exactly this function, and the
  continuous Markowitz baseline minimizes the same expression without the equal-weight and
  cardinality constraints. So results are directly comparable.
- **Why equal weights:** a binary decision per asset is what a QUBO/QAOA encoding can represent
  with N qubits. Continuous weights would need several qubits per asset, which is not
  feasible at this scale.
- **Choosing λ, the rule:** for each instance, compute λ_bal = sd(portfolio return) / sd(portfolio
  variance) over all C(N, K) selections (`calibration.py`). At λ_bal, return and risk vary equally
  across the feasible set, so neither alone decides the optimum. Then set λ to the median of
  λ_bal over 200 seeded in-sample instances (50 per N), rounded to the nearest 0.5.
- **Measured (2026-10-13, in-sample data):**

  | N | λ_bal p10 | median | p90 |
  |---|---|---|---|
  | 8 | 2.76 | 4.04 | 8.15 |
  | 10 | 2.17 | 4.49 | 7.45 |
  | 12 | 2.95 | 5.99 | 8.31 |
  | 14 | 3.09 | 4.47 | 7.40 |
  | all | 2.76 | **4.59** | 7.73 |

  → **λ = 4.5**.
- **Reasoning:** at λ ≈ 1 the return term varies about 4× more than the risk term, and the problem
  collapses toward "pick the K highest-return assets", which is a trivial benchmark. The rule
  was chosen to make the trade-off non-degenerate, *not* to make any solver look good or bad;
  solver performance was not consulted when choosing it.
- **Interpretation:** in the textbook utility U = μ_p − (A/2)σ_p², λ = 4.5 corresponds to A = 9,
  which is at the high end of commonly used values (about 1–10). This is a modelling choice made
  for benchmark quality, not a claim about real investors.

## ADR-016: Classical baselines and their validation

- **Status:** Accepted (2026-10-13)
- **Implementations** (`baselines/`), all on the same `PortfolioProblem`:
  1. **Brute force:** vectorized enumeration of all C(N, K) subsets; ties go to the lexicographically
     first subset. Exact, so it is the reference for every other method.
  2. **Greedy:** add the asset with the smallest increase in f, with fixed K scaling:
     Δⱼ = a(2(Σx)ⱼ + Σⱼⱼ) − bμⱼ. Deterministic.
  3. **Simulated annealing:**
     - Swap moves (one asset in, one out), so every state stays feasible.
     - Metropolis acceptance; geometric cooling from T₀ to 10⁻³·T₀.
     - T₀ is set so an average uphill move is accepted with probability 0.8, estimated from
       100 random swaps.
     - O(1) incremental swap Δ.
     - Seeded from stream (STREAM_SIMULATED_ANNEALING, N, instance).
     - Default 2,000 steps.
  4. **Continuous Markowitz + rounding:** long-only simplex QP solved by SLSQP from equal
     weights, then keep the K largest weights (equal-weighted). Deterministic.
- **Tests:**
  - Brute force agrees with an independent scalar-loop exhaustive search.
  - Greedy agrees with an independent full-objective greedy.
  - The O(1) swap Δ equals the full recomputation.
  - Markowitz weights satisfy the KKT optimality conditions and match a closed-form case.
  - Every solver returns a feasible selection whose objective is ≥ the true optimum.
- **Validation on real data (run record `results/baselines_validation/20261013T072701Z_a92a5a8.json`,
  clean commit `a92a5a8`, data snapshot SHA-256 `25b7e5a7…`; 50 seeded instances per N, λ = 4.5).**
  Gap = (f − f*)/(f_worst − f*):

  | N | K | Greedy optimal | Greedy mean / max gap | Markowitz+round optimal | M+R mean / max gap | SA (2,000 steps) optimal |
  |---|---|---|---|---|---|---|
  | 8 | 4 | 92% | 0.0011 / 0.022 | 84% | 0.0073 / 0.20 | 100% |
  | 10 | 5 | 94% | 0.0002 / 0.006 | 78% | 0.0132 / 0.20 | 100% |
  | 12 | 6 | 92% | 0.0007 / 0.023 | 52% | 0.0212 / 0.20 | 100% |
  | 14 | 7 | 92% | 0.0001 / 0.003 | 32% | 0.0414 / 0.24 | 100% |

- **Simulated annealing optimal-hit rate vs step budget** (same 50 instances per N and the same
  seeds; 100 calibration evaluations come on top of each budget):

  | N | C(N,K) | 25 | 50 | 100 | 200 | 500 | 2,000 |
  |---|---|---|---|---|---|---|---|
  | 8 | 70 | 72% | 94% | 100% | 100% | 100% | 100% |
  | 10 | 252 | 50% | 70% | 94% | 100% | 100% | 100% |
  | 12 | 924 | 22% | 40% | 74% | 90% | 100% | 100% |
  | 14 | 3,432 | 6% | 26% | 64% | 86% | 98% | 100% |

- **What this means for the research question (stated honestly):**
  - At N ≤ 14, the classical heuristics are strong. Greedy is optimal in more than 90% of instances.
  - SA with 2,000 steps is always optimal. That budget is comparable to exhaustive enumeration
    at N = 14, but SA also reaches about 90% with 200 steps.
  - So "QAOA finds better solutions than classical" is not a realistic outcome at these sizes.
  - A meaningful comparison must match the **evaluation budget**, e.g. QAOA shots versus SA steps.
    It should be reported as quality versus budget, not as best-found solution.
  - This informs the metric ADR due in W5 (PROJECT_PLAN.md §1 framing caveat). It is recorded
    now so that the metric is fixed before any quantum results exist.
- **Markowitz + rounding gets worse as N grows** (84% → 32% optimal). Measured explanation, using
  the same instances and counting assets with continuous weight > 10⁻⁴:
  - **Cause 1, partial:** the continuous optimum holds fewer than K assets more often as N grows
    (6%, 18%, 28%, 50% of instances for N = 8…14). When that happens, rounding fills the remaining
    slots with assets that have essentially zero weight, and it is optimal in only 12–36% of those cases.
  - **Cause 2:** even when it holds ≥ K assets, its optimal rate also falls (87% → 52%). The continuous
    weights are unequal, so "largest K weights" is not the same as "best equal-weight K-subset".
  - This is a known weakness of relax-and-round. It is a fair baseline, not a bug: its continuous
    solution passes the KKT checks in the tests.

## ADR-017: QUBO encoding and the penalty weight P

- **Status:** Accepted (2026-10-15)
- **Encoding** (`qubo/encoding.py`): E(x) = f(x) + P(1ᵀx − K)² = xᵀQx + c, with
  Q = aΣ + P·11ᵀ + diag(−bμ − 2PK) and c = PK². The Ising form uses x = (1 − z)/2, and
  `IsingModel.to_sparse_pauli_op()` gives the Qiskit cost Hamiltonian.
  - Convention: qubit measured 1 ⇔ z = −1 ⇔ asset selected; little-endian, matching Qiskit.
  - Tests check that QUBO, Ising and the Pauli-operator diagonal agree on every bitstring.
    Measured disagreement on a real N = 10 instance is ≤ 3×10⁻¹⁵.
- **Penalty rule** (`qubo/penalty.py`): P = 1.01 · max over m ≠ K of (f_greedy − L_m)/(m − K)², where
  L_m = a·λ_min(Σ)·m − b·(sum of the m largest μᵢ) lower-bounds f over all selections of size m.
  - This **provably** makes the QUBO's global minimum equal the constrained optimum (proof in the
    module docstring). A property test checks it on random problems, including non-PSD Σ.
  - It uses only cheap quantities: the greedy solution, an eigenvalue, and a sort. It never uses
    the optimum.
- **Alternatives rejected:**
  - A single global lower bound gave a looser P (8–16× the minimum).
  - Calibrating P from the brute-force optimum (`minimal_exact_penalty`) would hand QAOA
    knowledge of the answer that no classical baseline gets. That function is kept for
    analysis only.
- **Measured (clean commit `11c5f23`, 50 instances per N, λ = 4.5):**

  | N | QUBO exact | P / P_exact (median, p10–p90) | P / (f_worst − f*) median |
  |---|---|---|---|
  | 8 | 50/50 | 6.85 (4.67–9.40) | 0.77 |
  | 10 | 50/50 | 8.32 (6.41–10.96) | 0.68 |
  | 12 | 50/50 | 10.38 (6.78–13.56) | 0.70 |
  | 14 | 50/50 | 11.20 (8.79–13.16) | 0.64 |

- **Known consequence (to be measured, not assumed):** P is about 7–11× larger than strictly
  necessary. One unit of constraint violation costs about 0.6–0.8× the whole feasible objective
  range. For X-mixer QAOA, which explores infeasible states, a large penalty can compress the
  differences between good feasible solutions relative to the overall energy scale. Its effect on
  QAOA quality is to be measured in W5. Constraint-preserving (XY) mixers stay in the feasible
  subspace, where the penalty is identically zero, so they are unaffected.

## ADR-018: Noisy-simulation runtime benchmark

- **Status:** Accepted (2026-10-15)
- **Setup** (`python -m nisq_folio.noise`; run record
  `results/noise_benchmark/20261015T100806Z_11c5f23.json`, clean commit `11c5f23`):
  - One seeded real instance per N; depth-p QAOA with the ADR-017 QUBO.
  - Reference device FakeKingston: IBM Heron r2, 156 qubits, median CZ error 1.8×10⁻³ from its
    stored calibration snapshot. The reference device and noise levels are revisited in W8.
  - Noisy circuits transpiled at optimization level 3 onto the device's heavy-hex coupling map.
  - Time for **one energy evaluation**: one run with 1,024 shots, median of 3.
  - Machine: Intel i7-1165G7, 4 cores / 8 threads, 15.7 GB RAM, Windows 11; qiskit 2.5.2 and
    qiskit-aer 0.17.2. Wall times are specific to this machine.
- **Results (seconds per evaluation):**

  | N | p | Noiseless | Noisy trajectories | Noisy density matrix | 2q gates after routing (logical) |
  |---|---|---|---|---|---|
  | 8 | 1 | 0.004 | 1.92 | 1.87 | 101 (28) |
  | 8 | 2 | 0.005 | 2.69 | 2.08 | 209 (56) |
  | 10 | 1 | 0.006 | 3.62 | 2.03 | 165 (45) |
  | 10 | 2 | 0.008 | 5.26 | 2.51 | 329 (90) |
  | 12 | 1 | 0.010 | 10.46 | 7.86 | 209 (66) |
  | 12 | 2 | 0.015 | 18.00 | 13.14 | 415 (132) |
  | 14 | 1 | 0.027 | 49.74 | **25,554** (7.1 h) | 333 (91) |
  | 14 | 2 | 0.031 | 134.3 | not run (p = 1 already over the limit) | — |

- **Observations:**
  - **Routing:** IBM's sparse heavy-hex connectivity multiplies the two-qubit gate count by
    about 3–4× relative to the all-to-all logical circuit. That raises both runtime and
    accumulated noise.
  - **N = 14, exact density matrix:** the density matrix needs 4.0 GiB (14 active qubits for this
    seed; other transpiler seeds gave 15 active qubits, which would need about 16 GiB). Only about
    5.6 GB was free, so the 7.1-hour time very likely includes memory pressure. This could not be
    confirmed after the fact. Either way it is unusable.
  - **Benchmark design flaw found and fixed:** a run cannot be interrupted once started, so the
    time cap can only skip *later* cases. Because of that, this single case made the benchmark
    take more than 7 hours.

## ADR-007 resolution (2026-10-15): noisy optimization restricted to N ∈ {8, 10, 12}

- **Status of ADR-007:** Accepted, based on ADR-018.
- **Decision:**
  - Noisy QAOA optimization runs at **N ∈ {8, 10, 12}**, using the **density-matrix** method.
    It is the fastest noisy method at these sizes, and its noise is exact, with no
    trajectory-sampling error.
  - **N = 14 runs noiseless only.**
  - Noiseless runs and classical baselines cover all N ∈ {8, 10, 12, 14}.
- **Why N = 14 is excluded:**
  - The exact density matrix is infeasible: 7.1 h per evaluation.
  - Trajectories take 50–134 s per evaluation. At about 100 evaluations per optimization (the
    optimizer is fixed in W5), that is 1.4–3.7 h per optimization, or **41–112 h per 30-seed
    cell**. That is unaffordable for any meaningful number of noisy cells.
- **Budget model for fixing the noisy grid (W8–W9):**
  total hours ≈ Σ over cells of (30 seeds × E evaluations × t_eval(N, p)) / 3600.
  - With E = 100, p = 1 and the times above, **one 30-seed noisy cell costs about 1.6 h at N = 8,
    1.7 h at N = 10 and 6.6 h at N = 12**. These are upper estimates. W8's parameterized noise
    model and running several small simulations in parallel processes may reduce them; to be
    measured, not assumed.
  - Example: 12 noisy cells per N (e.g. 2 denoising × 2 mixers × 3 noise levels) × 30 seeds would
    cost ≈ 19 + 20 + 79 ≈ **118 laptop-hours**.
  - The number of noisy factor levels must therefore be chosen to fit an explicit compute budget,
    recorded in an ADR before the noisy grid runs.
- **Consequence for the research question:** the noise crossover is estimated from three sizes
  (8, 10, 12). That is enough to see a trend in N, but extrapolation beyond N = 12 must be stated as
  extrapolation.

## ADR-019: Success metric, defined before any quantum results

- **Status:** Accepted (2026-10-16). Fixed before any QAOA tuning or evaluation results exist.
- **Problem with the obvious metric:** a QAOA run at N ≤ 14 draws about 10⁴–10⁵ samples
  (evaluations × 1,024 shots), while there are only 2^N ≤ 16,384 bitstrings and C(N, K) ≤ 3,432
  feasible selections. "Best solution ever sampled" would be found by blind random guessing
  with the same number of samples, so it cannot show any quantum effect.
- **Primary metric:** the **amplification over random feasible guessing** of QAOA's final output
  distribution, A = p_opt / (m / C(N, K)), where m is the number of optimal selections.
  - It is computed exactly from the final state, so it has no shot noise (`qaoa/metrics.py`).
  - A > 1 means QAOA puts more probability on the optimum than uniformly guessing a valid
    selection. Infeasible output mass counts against QAOA automatically.
- **Secondary metrics:**
  - p_feasible;
  - E[gap], the normalized gap, with infeasible samples scored as the worst gap 1;
  - A_uniform = p_opt·2^N/m, against uniformly random bitstrings, a weaker bar;
  - samples_to_90 = ⌈ln 0.1 / ln(1 − p_opt)⌉.
- **Definition of the noise crossover** (research question, part 2): for each N, the noise level
  at which the median A over evaluation instances falls to **1**. That is the point where QAOA's
  output is no better than random valid guessing. It is estimated by interpolating across the
  noise sweep, with bootstrap confidence intervals over instances (W13). A secondary crossover
  uses A_uniform = 1.
- **Comparison with classical baselines (research question, part 1):** budget-matched, reported as
  quality versus number of samples or evaluations.
  - QAOA's cost is (optimization shots) + samples_to_90.
  - Simulated annealing's cost is the steps needed to reach 90% optimal (ADR-016).
  - Greedy is deterministic and costs about N·K evaluations.
  - **Expectation stated in advance:** at N ≤ 14, ADR-016 measured SA at about 90% optimal in
    about 200 steps, far below QAOA's sampling cost. So QAOA is not expected to beat the classical
    baselines on budget at these sizes. The study's value lies in measuring *how much* quantum
    advantage over random guessing exists noiselessly, and *where noise erases it*.

## ADR-020: Noiseless QAOA protocol and tuning rule (pre-registered)

- **Status:** Accepted (2026-10-16). The selection rule was committed before the tuning study ran.
- **Simulation:**
  - Exact noiseless state computed in NumPy (`qaoa/statevector.py`). This works because H_C is
    diagonal; the result matches Qiskit's `Statevector` to 2×10⁻¹⁶, a check that is also a test.
  - H_C is divided by s = √(ΣJᵢⱼ² + Σhᵢ²), the energy standard deviation under uniformly random
    spins, so that γ = O(1) for every instance.
- **Optimization** (`qaoa/optimize.py`):
  - Each evaluation samples S = 1,024 shots, and the optimizer sees only the sample-mean QUBO
    energy, as on hardware.
  - COBYLA with at most 100 evaluations and rhobeg = 0.25. It may stop earlier; the *actual*
    evaluations and shots are recorded.
  - Linear-ramp initial angles with scale Δ.
  - The final distribution is scored exactly (ADR-019).
  - Randomness comes only from shot sampling: stream (STREAM_QAOA_SHOTS, N, instance, config).
- **Disclosure: why the grid includes penalty scales below 1.** On one exploratory instance (a
  random 10-stock subset; seed 12345, not part of any evaluation set):
  - With the provable penalty of ADR-017, X-mixer QAOA stayed **below** random feasible guessing
    (A ≈ 0.5–0.9), even with exact energies, p up to 4, and a p = 1 grid search.
  - With a 10× smaller penalty, which happened to be calibrated from the known optimum, it reached
    A ≈ 5.8 at p = 4.
  - So the penalty strength is the decisive setting. It must be chosen legitimately: on separate
    tuning instances, as a multiple of the provable bound, never from the optimum.
- **Tuning grid:**
  - p ∈ {1, 2, 3, 4}; penalty scale c ∈ {1, 0.5, 0.25, 0.1}, with P = c × the ADR-017 bound;
    Δ ∈ {0.75, 1.5}. That is 32 configurations.
  - c < 1 gives up the guarantee that the QUBO minimum is feasible. Infeasible output mass is
    already penalized by A, so this is allowed and is visible in p_feasible.
- **Tuning instances:** 10 per N, drawn from a separate stream (STREAM_TUNING_INSTANCES),
  excluding the first 50 evaluation instances per N. Settings are therefore never chosen on the
  instances they will be scored on.
- **Selection rule** (`qaoa/tuning.py`):
  - score = mean over N of ln(median over instances of A).
  - Among configurations within 0.1 of the best score, pick the smallest p (cheaper and less noisy
    later), then the largest c (closest to the guarantee), then the smallest Δ.
- **Scope:** X mixer only. The constraint-preserving and adaptive mixers (W6) get their own
  protocol ADR.

### ADR-020 results (2026-10-16): X-mixer QAOA does not beat random feasible guessing

- **Run record:** `results/qaoa_tuning/20261016T004038Z_ccb823e.json`. Clean commit `ccb823e`, which
  is the commit that pre-registered the rule. 32 configurations × 10 tuning instances × 4 sizes =
  1,280 noiseless runs, taking 2 min 43 s.
- **Selected by the rule:** p = 2, c = 0.1 (P = 0.1 × provable bound), Δ = 0.75. The best score
  belonged to p = 4, c = 0.1, Δ = 0.75 (−0.128 versus −0.148); p = 2 is within the 0.1
  tolerance, so the rule prefers the shallower circuit.
- **Selected configuration on the tuning instances:**

  | N | A median [IQR] | A_uniform median | p_feasible median | E[gap] median | evaluations / shots (median) |
  |---|---|---|---|---|---|
  | 8 | 0.89 [0.62, 0.99] | 3.24 | 0.24 | 0.80 | 42 / 43,008 |
  | 10 | 0.77 [0.69, 0.89] | 3.12 | 0.33 | 0.80 | 42 / 42,496 |
  | 12 | 0.82 [0.75, 1.01] | 3.62 | 0.34 | 0.78 | 38 / 38,912 |
  | 14 | 0.99 [0.85, 1.03] | 4.73 | 0.33 | 0.78 | 39 / 39,936 |

- **Findings (stated as measured):**
  1. **Noiseless X-mixer QAOA with a legitimately chosen penalty puts *less* probability on the
     optimum than uniformly guessing a valid selection** (median A < 1 at every N). Only 6% of all
     1,280 runs exceeded A = 1. The maximum was 5.82, so amplification is possible but not typical
     under this protocol.
  2. It does beat uniformly random *bitstrings* by 3–5×, mostly by learning feasibility: valid
     selections rise from 25% under uniform to about 33%.
  3. **Smaller penalties were consistently better** (c = 0.1 dominates the top of the ranking),
     which confirms the penalty-dominance concern of ADR-017.
  4. **Optimizer limitation:** COBYLA stopped before its 100-evaluation budget in **100%** of runs
     (median 47). With shot noise, its trust region shrinks early. The earlier exploratory probe
     used exact energies and 400 evaluations, and reached A ≈ 5.8 with a small penalty. So part of
     the shortfall is plausibly the optimizer under shot noise, not QAOA itself. This is a
     hypothesis, not yet tested.
- **Consequences:**
  - For the X mixer, the primary noise crossover (A = 1, ADR-019) is **already crossed without any
    noise**. Noisy X-mixer cells can still be measured against the secondary bar A_uniform = 1.
  - This makes the constraint-preserving and adaptive mixers (W6) central. They start in the
    feasible subspace, where A = 1 holds at zero angles, so any gain shows up directly in the
    primary metric.
  - **No post-hoc change was made.** Changing the optimizer after seeing these results would make
    the tuning rule meaningless. A stronger shot-noise-robust optimizer (e.g. COBYLA with a smaller
    final trust radius, or SPSA) may be adopted only through a new ADR, labelled post-hoc, and
    applied identically to every mixer. The ADR-020 X-mixer result above stays reported as is.

## ADR-021: Constraint-preserving and adaptive mixers, SPSA, and the mixer study (pre-registered)

- **Status:** Accepted (2026-10-16). The rule and grid were committed before the mixer study ran.
- **Mixers** (`mixers/`):
  - **XY ring:** a parity-ordered product of exp(−iβ(XᵢXⱼ + YᵢYⱼ)/2) over ring edges, one Trotter step
    per layer. Starts from the Dicke state |D_K^N⟩, the uniform superposition of all C(N, K)
    feasible selections. The cost layer uses the objective f only: the penalty is identically
    zero inside the feasible subspace.
    - Tests check each pair rotation against the dense matrix exponential of Qiskit's Pauli
      operators, and check that the Hamming-weight distribution is preserved.
    - At zero angles A = 1 exactly, so every gain shows up directly in the ADR-019 metric.
  - **ADAPT** (adaptive mixer), after the ADAPT-QAOA idea of Zhu et al. (Phys. Rev. Research, 2022;
    **reference to be verified before citing in the report**):
    - Starts from the Dicke state. The pool is {XY ring} ∪ {XYᵢⱼ : i < j}, all weight-preserving.
    - For each layer, every pool operator's energy gradient is estimated by a shot-based
      central difference with δ = π/8, at the layer's ramp γ.
    - The largest-|gradient| operator is appended, then all angles are re-optimized with an equal
      share of the remaining evaluation budget.
    - The selection shots (2·|pool|·S per layer, e.g. about 750k at N = 14 and p = 4) are recorded
      separately and **count in budget comparisons**.
  - **X:** as in ADR-020, with penalty scales c ∈ {1.0, 0.1}.
- **Optimizer (post-hoc, labelled as such):**
  - SPSA is added because COBYLA stopped early in 100% of ADR-020 runs. Gains are calibrated
    to a first step of 0.2 rad, with perturbation c = 0.1 and 5 calibration samples (10
    evaluations), and Spall's standard exponents 0.602 / 0.101.
  - SPSA spends exactly its budget. Both optimizers get the same 100-evaluation budget and
    are available to **every** mixer.
  - The ADR-020 X-mixer result stays reported unchanged.
- **Grid:** mixer {x, xy_ring, adapt} × optimizer {cobyla, spsa} × p {1, 2, 3, 4}, plus
  c {1.0, 0.1} for X only. That is 32 configurations.
  - Fixed settings: Δ = 0.75 (the ADR-020 selection), S = 1,024, 100 evaluations.
  - Same 10 tuning instances per N as ADR-020 (separate stream, evaluation instances excluded).
  - Shot and SPSA randomness comes from stream (STREAM_MIXER_STUDY, N, instance, config).
- **Selection rule, applied per mixer** (`mixers/study.py`):
  - The same score as ADR-020: mean over N of ln(median A).
  - Within 0.1 of that mixer's best score, choose the smallest p, then the largest c, then COBYLA
    before SPSA.
- **What the comparison means:** these are tuning-instance numbers, used to choose settings, and
  are optimistic by construction. The mixer comparison reported in the final results is made on
  the evaluation instances (W11).

### ADR-021 results (2026-10-16): constraint-preserving mixers beat random feasible guessing

- **Run record:** `results/mixer_study/20261016T032011Z_b8811f3.json`. Clean commit `b8811f3`, the
  pre-registration commit. 32 configurations × 10 tuning instances × 4 sizes = 1,280 runs.
- **Selected by the rule:**
  - **adapt** (COBYLA, p = 4)
  - **xy_ring** (SPSA, p = 1)
  - **x** (COBYLA, p = 2, c = 0.1). This is the same X choice as ADR-020: the best X score was
    p = 4, but p = 2 is within tolerance.
- **Selected configurations on the tuning instances** (median [IQR]; "A > 1" is the share of instances):

  | N | ADAPT A | A > 1 | E[gap] | total shots | XY ring A | A > 1 | E[gap] | X A | A > 1 | E[gap] |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 8 | 2.99 [1.75, 5.16] | 100% | 0.19 | 340k | 1.98 [0.98, 3.12] | 70% | 0.25 | 0.83 | 30% | 0.77 |
  | 10 | 4.89 [2.30, 8.86] | 100% | 0.21 | 479k | 1.54 [1.43, 4.20] | 80% | 0.29 | 0.75 | 10% | 0.80 |
  | 12 | 7.11 [4.67, 13.18] | 100% | 0.26 | 651k | 3.56 [2.12, 4.54] | 100% | 0.30 | 1.02 | 60% | 0.79 |
  | 14 | 7.05 [5.41, 8.86] | 100% | 0.28 | 856k | 3.78 [1.51, 4.32] | 80% | 0.32 | 0.76 | 10% | 0.80 |

  Total shots: XY ring 102k; X 37k–45k (COBYLA stops early). ADAPT = 102k optimization shots plus
  238k–754k operator-selection shots.
- **Findings (as measured):**
  1. **Both constraint-preserving mixers beat random feasible guessing noiselessly. The X mixer
     does not.** ADAPT is above 1 on every tuning instance at every N. Its expected gap is about
     3–4× smaller than X's.
  2. **ADAPT's advantage costs shots:** 3.3–8.4× more than XY ring in total, almost all of it on
     estimating pool gradients (2·|pool|·S per layer). Budget-matched comparisons (ADR-019) must
     charge this.
  3. **The XY ring is best at p = 1 and gets worse with depth** (median A at p = 4 is 0.4–1.4). The
     cause is not established. Candidates are Trotter error in the ring product, or a harder
     optimization landscape at depth; this is a hypothesis.
  4. **SPSA did not rescue the X mixer.** Its best X score with SPSA (−0.181) was below its best
     with COBYLA (−0.100). The ADR-020 hypothesis, that COBYLA's early stopping explains X's weak
     result, is **not supported**. X's weakness tracks the penalty and feasibility (p_feasible
     about 0.31–0.33), not the optimizer. SPSA was selected only for the XY ring.
  5. A rises with N partly because the random baseline 1/C(N, K) shrinks. A comparison across N
     should use E[gap] and p_opt alongside A.
- **Caveats:**
  - These are tuning-instance results, which are optimistic. The mixer comparison of record
    comes from the evaluation instances (W11).
  - **Noise implications for W8, not yet measured:** an X layer has no two-qubit mixer gates. An XY
    ring layer has N two-qubit XY gates. ADAPT's single-pair operators act on arbitrary pairs,
    which need routing on heavy-hex hardware. Constraint preservation also only holds
    noiselessly: noise leaks probability out of the feasible subspace, and the metric counts that
    as failure. The noisy study needs Dicke-state preparation and XY circuits, to be implemented
    and verified in W8.

## ADR-022: RMT covariance denoising and its out-of-sample evaluation (pre-registered)

- **Status:** Accepted (2026-10-16). The method, metrics and prediction were committed before the
  real-data evaluation ran.
- **Method** (`denoise/rmt.py`): Marchenko–Pastur eigenvalue clipping of the correlation matrix,
  after Laloux, Cizeau, Bouchaud & Potters (Phys. Rev. Lett. 83, 1467, 1999) and the review by Bun,
  Bouchaud & Potters (Physics Reports 666, 2017). **Both references are to be verified before
  citing in the report.**
  - With q = N/T and σ² = 1, keep the eigenvalues above λ₊ = (1 + √q)², and replace every other
    eigenvalue by their mean, which preserves the trace.
  - Keep the eigenvectors and rescale to unit diagonal.
  - Rebuild Σ = S·C_clean·S with the sample standard deviations, so only correlations change.
    μ is never denoised.
- **Level:** cleaning is done once, on the 30-asset universe's in-sample returns (T = 1,510,
  q = 0.020, band [0.738, 1.302]), and each instance uses its block. The alternative,
  instance-level cleaning, would have q ≈ 0.005–0.009. RMT is an N, T → ∞ theory, so the larger
  matrix is the more defensible place to apply it.
- **Factor levels for the experiment grid:** "none" = sample covariance; "RMT" = `rmt_clip` (the
  textbook rule above).
- **Reference estimators** (reported in this evaluation, not grid factors):
  - `rmt_band_only` averages only eigenvalues inside [λ₋, λ₊].
  - `ledoit_wolf` is linear shrinkage from scikit-learn.
- **Spectrum observed before evaluation:**
  - 3 eigenvalues lie above the band (15.1, 2.88, 1.33), 2 inside, and **25 below λ₋**.
  - Eigenvalues below λ₋ cannot be produced by noise of this q, so they indicate genuine
    (sector) structure.
  - `rmt_clip` therefore replaces 27 of 30 eigenvalues, which effectively makes it a
    3-factor model.
- **Synthetic check that changed the expectation:**
  - In a one-factor model with *equal* loadings (flat bulk), clipping estimated the population
    correlation better than the sample on 10 of 10 seeds (error ratio 0.31–0.85).
  - With *heterogeneous* loadings, where true eigenvalues lie below the band, clipping was
    **worse than the sample on 9 of 10 seeds**, and band-only was better on 8 of 10.
  - Both results are pinned as tests (`tests/test_denoise_rmt.py`).
- **Pre-registered prediction:** because the real spectrum looks like the heterogeneous case,
  `rmt_clip` may well **not** improve, and could worsen, out-of-sample results relative to the
  sample covariance. The evaluation decides; the method is not changed after seeing it.
- **Evaluation protocol** (`denoise/evaluation.py`, `python -m nisq_folio.denoise`). Classical only:
  - Instances: the first 50 evaluation instances per N (default stream; no tuning involved).
  - For each estimator, take the brute-force optimal selection x̂ with w = x̂/K, at λ = 4.5. λ is
    not re-calibrated per estimator.
  - Scored on 2021–2023 out-of-sample returns.
  - **Primary:** realized variance wᵀΣ_oos w, compared with the sample estimator, paired by instance.
  - **Secondary:** realized objective λ·v_oos − μ_oosᵀw; risk ratio v_oos/v_pred; share of
    instances with an identical selection; universe-level Frobenius distance between in-sample
    and out-of-sample correlations.
  - Statistics: median paired difference, share of instances improved, and a two-sided Wilcoxon
    signed-rank p over non-zero differences. Instances share assets and are not independent, so
    p-values are descriptive only.
- **Process incident (disclosed per CLAUDE.md rule 5):**
  - While writing these tests, I accidentally overwrote the Week 1 placeholder
    `tests/test_denoise.py` (an existing test, contrary to rule 4). It was caught before any
    commit and restored byte-for-byte from git. The new tests live in `tests/test_denoise_rmt.py`.
  - A check confirmed that no placeholder test has changed since commit `3c667dc`.

### ADR-022 results (2026-10-16): textbook RMT clipping does not help on this universe

- **Run record:** `results/denoise_evaluation/20261016T045617Z_01159b2.json`. Clean commit `01159b2`,
  the pre-registration commit. 50 evaluation instances per N, exact optima, λ = 4.5.
- **Universe level:**
  - `rmt_clip` kept 3 of 30 eigenvalues; `rmt_band_only` kept 28; Ledoit–Wolf shrinkage was 0.022.
  - Frobenius distance from the in-sample correlation to the out-of-sample correlation:
    sample 5.643, **rmt_clip 5.943 (worse)**, band-only 5.646, **Ledoit–Wolf 5.329 (best)**.
- **`rmt_clip` against the sample covariance, per N:**

  | N | Selection changed | Among changed: realized variance lower / median relative change | Among changed: realized objective lower | Wilcoxon p (variance, all instances) |
  |---|---|---|---|---|
  | 8 | 13/50 | 46% / +2.8% | 31% | 0.455 |
  | 10 | 15/50 | 53% / −0.3% | 47% | 0.389 |
  | 12 | 22/50 | 27% / +4.8% | 45% | 0.0074 |
  | 14 | 18/50 | 44% / +0.1% | 33% | 0.196 |

  Pooled over the 68 changed instances: realized variance was lower in 41% (median +1.8%) and the
  realized objective was lower in 40%.
- **References:**
  - Band-only and Ledoit–Wolf changed the selection in only 2–8% and 0–4% of instances, with no
    systematic effect.
  - Median risk ratio v_oos/v_pred was about 0.63–0.77 for every estimator: realized 2021–23 risk
    was below predicted. The in-sample window contains the 2020 crash, so this regime effect is
    shared by all estimators. It is slightly stronger for `rmt_clip`, at 0.63–0.73.
- **Conclusion, as measured and as predicted in advance:**
  - Textbook Marchenko–Pastur clipping does **not** improve the classical optimal selection on
    this universe. When it changes the answer, the answer is more often worse out of sample.
  - The likely mechanism is the one in the synthetic counter-example. That is an interpretation,
    not proven for the real data.
- **Consequences:**
  1. The denoising factor stays exactly as pre-registered (none vs `rmt_clip`). It is **not**
     swapped for a variant that looks better after the fact.
  2. The open research question is now narrower: does denoising change *QAOA's* behaviour? The
     3-factor cleaned matrix gives a different, smoother landscape, which may be easier or harder
     to optimize. That is measured in the W9–W11 grid on the same instances.
  3. The report must state that the pipeline's first stage does not help the classical answer
     here, so any pipeline benefit would have to come from its interaction with QAOA, not from
     better risk estimates.

## ADR-023: Gate-level circuits and the tunable noise model

- **Status:** Accepted (2026-10-16).
- **Circuits** (`noise/circuits.py`):
  - **Dicke state:** the split-and-cyclic-shift construction of Bärtschi & Eidenbenz (FCT 2019;
    **verify the reference before citing**). It is exact for every (N, K) with N ≤ 7 (test).
  - **XY gate:** exp(−iβ(XX + YY)/2) = `XXPlusYYGate(2β, 0)`, checked against the NumPy mixer.
  - **Cost layer:** RZZ / RZ with angles 2γJ/s and 2γh/s, using the same scale s as the noiseless
    simulator.
  - Whole circuits for X, XY-ring and ADAPT sequences match the NumPy state to 10⁻¹² (tests).
- **Noise model** (`noise/density.py`): one knob, the **noise level m**, scales the median error
  rates of the reference device FakeKingston (IBM Heron):
  - single-qubit error 2.36×10⁻⁴, CZ error 1.83×10⁻³, readout error 9.52×10⁻³;
  - m = 0 is noiseless and m = 1 is about today's hardware.
  - Gate errors are depolarizing with λ = r·d/(d − 1), so the average gate infidelity equals r.
    RZ gates are virtual and error-free.
  - Readout is an independent symmetric bit flip applied to the exact distribution.
  - **Not modelled:** thermal relaxation (T1/T2), crosstalk, coherent errors. This is a stated
    simplification. The crossover is therefore reported in units of "m × today's median
    depolarizing error", not as a full device emulation.
- **Execution:**
  - Circuits are transpiled once per mixer sequence onto Kingston's heavy-hex coupling map (basis
    cz/rz/sx/x, level 3, fixed seed), then bound per evaluation.
  - Simulation is an exact density matrix (`save_probabilities` on the routed final layout).
  - Shots are drawn from the exact noisy distribution with the run's seeded RNG, exactly as in the
    noiseless protocol (`noise/runner.py`).
- **Equivalence check, the key validation:** at m = 0, the noisy pipeline (transpile → route →
  density matrix → layout → readout) reproduces the NumPy simulator to 10⁻¹⁵ for single
  distributions. Whole optimizations give the same evaluations, the same ADAPT operator sequence,
  and |ΔA| ≈ 10⁻¹⁵ for all three mixers (tests).
  - **Bug found and fixed:** the density matrix returns about 10⁻¹⁶ residue in bins that are
    exactly zero in theory. NumPy's multinomial sampler consumes random numbers for non-zero bins,
    so the RNG stream drifted and optimizations diverged at m = 0 (ADAPT chose different
    operators).
  - Fix: set probabilities below 10⁻¹⁵ to 0. This is physically negligible (expected count
    < 10⁻¹² at 1,024 shots).
- **First noise effects (N = 6, p = 2, one XY sequence):** the circuit has 254 CZ after routing.
  p_feasible is 1.00 at m = 0, 0.62 at m = 1 and 0.34 at m = 4. Noise leaks probability out of the
  feasible subspace, and the metric counts that as failure.
- **Measured cost per noisy evaluation at m = 1** (one exact density-matrix run, this laptop):

  | N | X, p = 2 | XY ring, p = 1 | ADAPT, p = 4 |
  |---|---|---|---|
  | 8 | 0.17 s (222 CZ) | 0.30 s (408 CZ) | 0.54 s (808 CZ) |
  | 10 | 1.0 s (328 CZ) | 3.0 s (675 CZ) | 3.9 s (1,358 CZ) |
  | 12 | 18 s (533 CZ) | 44 s (1,053 CZ) | 217 s (1,927 CZ) |

  - The Dicke preparation and XY gates need many more CZ than the X mixer, so the constraint-
    preserving mixers pay **more noise and more runtime**.
  - **Machine variability:** identical test runs on this laptop took between 7 s and 441 s
    (thermal throttling or background load are likely; not diagnosed). Wall-time budgets need a
    safety factor.
- **Budget consequences (rough, for the W10 budget ADR; not decisions yet):**
  - **ADAPT under noise at N = 12 is infeasible here.** Operator selection alone needs
    2 × 67 × 4 = 536 noisy circuits per run, plus 100 evaluations at about 217 s. That is on the
    order of 20 h per run, or about 600 h per 30-seed cell.
  - At N = 10, ADAPT is about 20 min per run (about 10 h per cell); at N = 8, about 2–3 min per run.
  - X and XY ring are affordable at N = 8 and 10. At N = 12 they cost about 12 min (X) and about
    75 min (XY ring) per run.
  - Options to decide in W10, with the user's laptop-hour budget:
    - (a) noisy ADAPT only at N ≤ 10;
    - (b) a cheaper "transfer" protocol: choose the ADAPT operator sequence noiselessly, then
      optimize its angles under noise;
    - (c) fewer noise levels at N = 12.

## ADR-024: Experiment runner design

- **Status:** Accepted (2026-10-17). This is engineering only: no scientific setting is chosen here.
- **Cells and replicates:**
  - A cell is (denoising, mixer, noise level m, N).
  - The **replicates of every cell are the same first 30 evaluation instances per N** (default
    stream). Every comparison between cells is therefore *paired by instance*. This is how the
    plan's "30+ seeds per cell" is met.
  - One QAOA run per (cell, instance). Its seed comes from the task identity alone:
    (STREAM_EXPERIMENT, kind, N, instance, denoising, mixer, round(m·10⁶)).
- **Fixed protocol:** each mixer uses its ADR-021 selection, written explicitly in
  `experiment.mixer_settings`:
  - x: COBYLA, p = 2, c = 0.1;
  - xy_ring: SPSA, p = 1;
  - adapt: COBYLA, p = 4.

  Shots, budget, ramp, SPSA and ADAPT settings come from the `qaoa` and `mixer_study` sections.
- **Noise:** m > 0 is expanded only for `noisy_sizes` = {8, 10, 12} (ADR-007). m = 0 is always
  included as the reference. m = 0 runs on the NumPy simulator and m > 0 on the density-matrix
  pipeline; the two are identical at m = 0 (ADR-023).
- **Classical tasks per (N, instance, denoising):**
  - exact optimum and worst value, greedy, Markowitz + rounding;
  - SA at budgets {25, 50, 100, 200, 500, 2000}, for the budget-matched comparison of ADR-019;
  - out-of-sample score of the optimum.
- **Out-of-sample score of each QAOA run:** its most probable feasible selection (the answer one
  would deploy) is scored on 2021–23 returns.
- **Resumability and provenance** (`experiments/store.py`):
  - One JSON file per finished task, written atomically, so a crash never leaves a partial file.
  - Each file records its git state and config SHA-256 (CLAUDE.md rule 6).
  - A directory created under one config refuses results from another config.
  - Re-running skips finished tasks.
- **Parallelism:** `--workers W` uses W processes with Aer capped at 1 thread each.
  **Test-verified:** results with 2 workers are identical to serial results, and resuming does
  not redo or alter finished tasks.
- **Cost estimate** (`--estimate`, `experiments/cost.py`): evaluations per run × seconds per
  evaluation from the ADR-023 measurements. It is order-of-magnitude only, given this laptop's
  speed variability.
  - The current grid (m = 0 only) is **960 tasks, about 0.5 h predicted**.
  - Noisy levels and the noisy budget are fixed in W10 (ADR-025).
- **Bug found by the real smoke run and fixed (2026-10-17):**
  - `--workers > 1` crashed under `python -m` on Windows (`Can't get attribute '_worker_init'`).
    Spawned workers cannot unpickle functions defined in `__main__`.
  - The unit test called `main()` from pytest, so it never exercised that path. The workers now
    live in `experiments/parallel.py`.
  - A subprocess test (`tests/test_experiment_cli_subprocess.py`) now runs the real `python -m`
    command with 2 workers.
  - After the fix, a 260-task real-data smoke run (3 workers) completed in 32 s. Its results were
    discarded: uncommitted code, not results of record.
- **Note for the analysis (W13):** the smoke run showed that A > 1 does not imply that the modal
  (most probable) feasible selection is optimal. Both are reported. A measures probability mass on
  the optimum; the modal selection is what would be deployed.

## ADR-025: Noisy grid: pilot, noise levels and budget (rules pre-registered)

- **Status:** Rules accepted (2026-10-17). They were committed before the pilot ran. The resulting
  levels are recorded below after the pilot.
- **Budget (user decision, 2026-10-17):** 100 laptop-hours for the noisy grid, as about 10
  overnight runs of about 10 h each. The runner resumes, so each night continues where the
  previous one stopped (ADR-024).
- **Scope fixed now:**
  - Noisy ADAPT runs only at N ∈ {8, 10}. At N = 12 it is infeasible here, at about 20 h per run
    (ADR-023). Set via `experiment.noisy_exclusions`.
  - Noiseless ADAPT still runs at every N.
- **Pilot** (`python -m nisq_folio.experiments.pilot_run`). It uses tuning instances only, drawn
  from the separate stream with the evaluation instances excluded:
  1. **Timing:** seconds per noisy evaluation at m = 1 for every mixer and N ∈ {8, 10, 12}
     (not ADAPT at 12), with Aer on automatic threads. At N ∈ {8, 10} it is also timed with
     1 thread.
  2. **Bracketing:** full noisy runs at N = 8 for all three mixers at m ∈ {0.25, 1, 4}, plus
     the m = 0 reference, on 2 tuning instances, recording A.
- **Pre-registered rules** (`experiments/pilot.py`):
  - **Rule 1, upper level:**
    - For each mixer, the pilot crossing is the m where the median A falls to 1, interpolated
      in log m.
    - m_max is the smallest value of the ladder {0.125, 0.25, 0.5, 1, 2, 4, 8} that is ≥ 1 and
      ≥ every mixer's crossing.
    - If some mixer never falls to 1, m_max = 8.
  - **Rule 2, levels and budget:**
    - Start with every ladder value ≤ m_max.
    - While the estimated noisy-grid hours exceed 100, drop the smallest non-zero level,
      keeping at least 3.
    - If it is still over budget, **report the overrun** to the user rather than silently
      cutting cells.
  - **Rule 3, parallelism:**
    - Use 3 single-thread workers if 3·t_auto/t_1thread > 1.2 at both N = 8 and N = 10;
      otherwise use 1 worker.
    - The measured gain is applied only at the sizes where it was measured. N = 12 is assumed
      to gain nothing (conservative).
- **Why these rules:**
  - The sweep must reach past the point where QAOA stops beating random guessing, or no
    crossover can be estimated.
  - Low noise levels are the least informative about the crossover, so they are dropped first.
  - Pilot instances are never evaluation instances, so choosing levels from the pilot cannot
    bias the results of record.

### ADR-025 results (2026-10-17): pilot, levels {0.25, 0.5, 1}, full grid by user decision

- **Run record:** `results/pilot/20261017T013226Z_f29d883.json`. Clean commit `f29d883`, the
  pre-registration commit.
- **Timing (s per noisy evaluation at m = 1; automatic threads | 1 thread):**

  | N | X | XY ring | ADAPT |
  |---|---|---|---|
  | 8 | 0.07 \| 0.12 | 0.14 \| 0.28 | 0.26 \| 0.54 |
  | 10 | 0.52 \| 2.09 | 1.48 \| 6.13 | 1.86 \| 7.22 |
  | 12 | 12.4 | 29.2 | not run (excluded) |

  These are about 1.5–2× faster than the ADR-023 measurements on the same machine, consistent with
  its speed variability.
  - **Rule 3:** the 3-worker gain was 1.86 / 1.53 / 1.45 at N = 8, but 0.74 / 0.73 / 0.77 at N = 10.
    It is not above 1.2 everywhere, so **1 worker**.
- **Bracketing at N = 8** (median A over only 2 tuning instances, so noisy; e.g. ADAPT at m = 0.25
  exceeds its own m = 0 value):

  | Mixer | m = 0 | 0.25 | 1 | 4 | Crossing (A = 1) |
  |---|---|---|---|---|---|
  | X | 0.56 | 0.67 | 0.63 | 0.40 | ≤ 0.25 (already < 1 at m = 0) |
  | XY ring | 1.14 | 0.99 | 0.52 | 0.28 | ≈ 0.25 |
  | ADAPT | 1.37 | 2.88 | 0.53 | 0.27 | ≈ 0.76 |

  **At today's hardware noise (m = 1), every mixer was below random feasible guessing in the pilot.**
- **Rule 1:** m_max = 1. **Rule 2:** levels {0.125, 0.25, 0.5, 1}. Dropping the smallest gives
  {0.25, 0.5, 1}, the minimum of 3, which is still estimated at **233 h, over the 100 h budget**.
  This was reported to the user as Rule 2 requires.
- **Where the hours go:** noisy XY at N = 12, 147 h; ADAPT at N = 10, 44 h; X at N = 12, 28 h; the
  rest about 14 h.
- **Options offered:**
  - A (86 h): drop noisy XY at N = 12.
  - C (80 h): XY at N = 12 on 15 instances, and noisy runs without the denoising factor.
  - The full grid (233 h).
- **User decision (2026-10-17): the full grid, about 233 h, about 23 overnight runs.** The design is
  unchanged: 3 mixers (ADAPT noisy only at N ≤ 10), both denoising levels, 30 instances, and
  m ∈ {0, 0.25, 0.5, 1} at N ∈ {8, 10, 12}. That is 2,400 tasks, about 234 h estimated. The
  estimate is uncertain by roughly 2× because of machine variability.
- **Config of record:** `configs/main.yaml` extends `base.yaml` (new `extends` support, with mappings
  merged recursively). It changes only the noise levels and the pilot cost model.
  - `base.yaml` and its committed tests are untouched. Editing `base.yaml` would have broken two
    committed tests that pin its Week 9 values, and CLAUDE.md rule 4 forbids editing them.
  - The experiment fingerprint and recorded config text cover the whole chain, so a later change
    to `base.yaml` is still detected.

## ADR-026: Analysis of record (pre-registered before the Week 11 grid ran)

- **Status:** Accepted (2026-10-17). Tables, metrics and tests were committed before any
  evaluation-instance QAOA result existed.
- **Data:** the experiment of record `configs/main.yaml` (ADR-025). The noiseless results come from
  `--max-noise 0`, i.e. the evaluation instances, 30 per N.
- **Tables** (`analysis/summary.py`, `python -m nisq_folio.analysis`):
  - **T1, cells:** per (denoising, mixer, N): median A [IQR], share with A > 1, median E[gap],
    share of instances whose modal feasible selection is optimal, median total shots.
  - **T2, mixers:** paired by instance, at denoising = none. adapt/xy_ring, xy_ring/x and adapt/x,
    each as the geometric-mean ratio of A, the share of instances where the first is better, and
    a Wilcoxon p.
  - **T3, denoising:** the same paired statistics for rmt_clip/none, per (mixer, N).
  - **T4, classical:** share of instances solved optimally by greedy, Markowitz + rounding, and SA
    at each step budget.
  - **T5, budget:** median QAOA total shots (optimization plus ADAPT selection) against the
    smallest SA budget reaching ≥ 90% optimal.
  - **T6, deployment:** median ratio of the modal selection's realized 2021–23 variance to the true
    optimum's, plus the modal-optimal rate.
- **Statistics:**
  - Paired two-sided Wilcoxon signed-rank on per-instance differences of ln A, with zeros dropped.
  - The effect size is exp(median of the ln-differences).
  - **p-values are descriptive.** Instances share assets, so they are not independent, and no
    correction for multiple comparisons is applied.
- **Interpretation rule, fixed in advance:** "mixer M beats random feasible guessing at N" is stated
  only if its median A > 1 **and** A > 1 on at least two thirds of instances. Otherwise the result
  is reported as mixed or not supported.
