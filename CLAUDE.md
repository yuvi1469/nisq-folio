# CLAUDE.md — standing rules for NISQ-Folio

These rules govern every session in this repository. They override convenience.

## Project context
Research-grade capstone: RMT covariance denoising -> QUBO -> adaptive-mixer QAOA for
cardinality-constrained portfolio selection under NISQ noise, benchmarked against classical
baselines. Clarity and maintainability matter more than cleverness. Scope and milestones:
`PROJECT_PLAN.md`. Decisions: `DECISIONS.md`.

Environment: Python 3.12, managed by `uv`. Always run tools through `uv run ...`. Change
dependencies only with `uv add` / `uv remove` (which update `uv.lock`), and record why in an ADR.

## Rules

1. **Seeds.** Fixed random seeds everywhere. Seeds are passed explicitly as arguments
   (e.g. a `numpy.random.Generator` or an integer seed), never set as global RNG state
   (no `np.random.seed`, no `random.seed`). Derive per-run seeds from the config's root seed
   with `numpy.random.SeedSequence`. This includes Qiskit/Aer (`seed_simulator`,
   `seed_transpiler`) and optimizer seeds.
2. **Types and docstrings.** Type hints and docstrings on every public function. Docstrings
   must state the equation being implemented where relevant.
3. **No hardcoded paths.** Use config files (YAML under `configs/`).
4. **Never edit or delete an existing test to make it pass.** If a test fails, report it to the
   user and investigate the actual cause.
5. **Never fabricate results, citations, or data.** If something is uncertain, say so explicitly.
6. **Provenance.** Every experiment run must log its config and the current git commit hash.
7. **Verify before finishing.** Before finishing any task, run all three and show the user the
   actual output:
   ```
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```
8. **ADRs.** Record every non-trivial architectural or scientific choice as a numbered ADR in
   `DECISIONS.md`, with the reasoning.
9. **Coverage.** Target 85% test coverage on core modules (enforced by `fail_under = 85` in
   `pyproject.toml`). Prioritize meaningful property-based tests over padding for the number.
