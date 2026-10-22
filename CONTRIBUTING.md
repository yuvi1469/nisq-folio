# Contributing to NISQ-Folio

Thank you for your interest. NISQ-Folio is a research benchmark, so contributions are welcome
as long as they keep the results reproducible.

## Reporting problems and asking questions

Open a GitHub issue at <https://github.com/yuvi1469/nisq-folio/issues>. For a bug, include the
command you ran, the config file, the full error output, your operating system, and the output
of `uv run python -c "import nisq_folio, qiskit, qiskit_aer; print(qiskit.__version__,
qiskit_aer.__version__)"`.

## Making changes

1. Fork the repository and create a branch.
2. `uv sync --locked`, then make your change.
3. Follow the standing rules in [CLAUDE.md](CLAUDE.md). The most important ones:
   - Pass random seeds explicitly; never set global RNG state.
   - Type hints and docstrings (with the equation being implemented) on public functions.
   - Paths and parameters go in YAML configs, not in code.
   - Never edit or delete an existing test to make it pass.
   - Record non-trivial scientific or architectural choices as a new ADR in
     [DECISIONS.md](DECISIONS.md). Changes to an analysis after seeing its results are labelled
     post hoc.
4. Before opening a pull request, run all three:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```
   CI runs the same checks on Linux and Windows.
5. Open a pull request describing what changed and why.

## Scope and support

The maintainers (Yuvaraj Guliani and Kallol Das) review issues and pull requests on a
best-effort basis. Changes that alter a reported result must come with a new ADR and must not
silently change the configs of record (`configs/base.yaml`, `configs/main.yaml`).

The project must stay usable at zero cost (ADR-008): please do not add dependencies on paid
services, data feeds, cloud compute or quantum hardware.
