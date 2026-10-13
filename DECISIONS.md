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

- **Status:** Proposed. **Decision pending.** Needs a measured runtime benchmark (Milestone W4).
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
