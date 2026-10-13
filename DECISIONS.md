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
