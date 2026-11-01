# Release checklist: v1.0.0

The v1.0.0 release freezes the code, configs, decision log and paper sources that produced the
results of record, and archives them on Zenodo for a citable DOI (ADR-030). Everything here is free
(ADR-008). Work through the steps in order; each one says who does it.

## 1. Results are final

- [ ] The noisy grid `main_r2` has finished: 1,440 / 1,440 noisy tasks (dashboard, Progress tab).
- [ ] The Windows scheduled task is deleted (it was only for the run):
      `Unregister-ScheduledTask -TaskName 'NISQ-Folio main_r2' -Confirm:$false`.
- [ ] Final analysis: `uv run python -m nisq_folio.analysis.noise_report --config configs/main.yaml --superseded main`.
      Copy the report to `docs/results/noisy_final.md` with a provenance header (commit, date),
      as for the interim reports.
- [ ] ADR-028: add the final results section, including prediction 3 (crossover vs N) with N = 12.

## 2. Paper and documentation

- [ ] `uv run python paper/make_tables.py --config configs/main.yaml` and
      `uv run python paper/make_figures.py --config configs/main.yaml`; inspect every figure.
- [ ] Fill every `\pending{...}` in `paper/main.tex` (`grep -n 'pending{' paper/main.tex` must list
      only the macro definition), resolve the `\todo` on the AI-use wording, and build with
      `tectonic -X compile paper/main.tex` with no warnings.
- [ ] Both authors read the paper.
- [ ] README: status section says the grid is complete; results summary points to the final report.
- [ ] Update the decision-record count in `paper/main.tex` if new ADRs were added.

## 3. Version and metadata

- [ ] `pyproject.toml`: `version = "1.0.0"`, then `uv lock` (updates the project's own entry).
- [ ] `CITATION.cff`: `version: 1.0.0` and `date-released:` the release date. Add ORCIDs if you have
      them (free at orcid.org).
- [ ] `LICENSE`: decide whether the copyright line should name both authors.
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest` all pass; CI is green.

## 4. Zenodo (user, once)

- [ ] Sign in at <https://zenodo.org> with GitHub (free), open *GitHub* in the account menu, and
      switch the `nisq-folio` repository **on**. This must happen *before* the release is
      published, or Zenodo will not see it.
- [ ] Zenodo reads `CITATION.cff` for authors, title, licence and keywords; there is no separate
      `.zenodo.json`, so the two cannot disagree.

## 5. Tag and release

- [ ] `git tag -a v1.0.0 -m "v1.0.0: results of record"` on the commit with green CI, then
      `git push origin v1.0.0`.
- [ ] `gh release create v1.0.0 --title "v1.0.0" --notes-file <notes>`. Zenodo archives the release
      within minutes and mints two DOIs: one for this version and a concept DOI for all versions.

## 6. After the DOI exists

- [ ] Add the concept DOI to `CITATION.cff` (`doi:`), a DOI badge to `README.md`, and the DOI to the
      data-availability section of `paper/main.tex`. Commit (no new tag needed).
- [ ] **Decision for the authors: publish the run records.** `results/` is not in git, so the
      archive contains code but not the 2,400+ per-task result files the paper is computed from.
      They contain only tickers, QAOA parameters, metrics, seeds and git commits, no price data
      and no local paths (checked 2026-11-01; `manifest.json` has two local paths to strip).
      Reproducing them costs about two weeks of compute. Recommended: zip `results/experiments/
      main_r2/runs/`, `main_r2/manifest.json` (paths stripped) and the superseded `main/runs/`,
      upload as a separate free Zenodo *dataset* linked to the software DOI, and cite it in the
      paper.
