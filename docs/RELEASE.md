# Release checklist: v1.0.0

The v1.0.0 release freezes the code, configs, decision log and paper sources that produced the
results of record, and archives them on Zenodo for a citable DOI (ADR-030). Everything here is free
(ADR-008). Work through the steps in order; each one says who does it.

## 1. Results are final

- [x] The noisy grid `main_r2` has finished: 1,440 / 1,440 noisy tasks (2026-11-19).
- [x] The Windows scheduled task is deleted (it was only for the run):
      `Unregister-ScheduledTask -TaskName 'NISQ-Folio main_r2' -Confirm:$false`.
- [x] Final analysis: `uv run python -m nisq_folio.analysis.noise_report --config configs/main.yaml --superseded main`.
      Copy the report to `docs/results/noisy_final.md` with a provenance header (commit, date),
      as for the interim reports.
- [x] ADR-028: add the final results section, including prediction 3 (crossover vs N) with N = 12.

## 2. Paper and documentation

- [x] `uv run python paper/make_tables.py --config configs/main.yaml` and
      `uv run python paper/make_figures.py --config configs/main.yaml`; inspect every figure.
- [x] Fill every `\pending{...}` in `paper/main.tex` (`grep -n 'pending{' paper/main.tex` must list
      only the macro definition), resolve the `\todo` on the AI-use wording, and build with
      `tectonic -X compile --keep-logs paper/main.tex` with no warnings in the fresh `main.log`
      (without `--keep-logs` an old log is left in place). Done 2026-11-20: all markers
      filled, AI-use statement rewritten, fresh build has no layout warnings.
- [ ] Both authors read the paper.
- [x] README: status section says the grid is complete; results summary points to the final report.
- [x] Update the decision-record count in `paper/main.tex` if new ADRs were added (still 30).

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

## 7. arXiv preprint

- [ ] Build with `tectonic -X compile --keep-intermediates paper/main.tex`, then
      `uv run python tools/arxiv_bundle.py`. It refuses while any pending or todo marker remains or
      `main.bbl` is stale, and writes `paper/arxiv.zip` (main.tex, main.bbl, and exactly the tables
      and figures the paper reads). arXiv compiles with pdfLaTeX; the preamble and the
      bibliography are plain ASCII/pdfLaTeX-compatible, but we build with Tectonic (XeTeX), so
      **check arXiv's compiled preview PDF page by page** before submitting.
- [ ] Category: `quant-ph` primary, `q-fin.PM` (portfolio management) cross-list.
- [ ] Accounts and endorsement (user): both authors need free arXiv accounts. A first submission
      to a category usually needs an **endorsement** from someone who has published there; arXiv
      may waive it based on the submitter's history or affiliation, which you only learn when you
      start the submission. If asked, request it through arXiv's endorsement page from a
      colleague who has posted to `quant-ph`. Allow a few days.
- [ ] Licence: choose one at submission (CC BY 4.0 is the most open). Confirm the target journal's
      preprint policy allows it before choosing.
- [ ] Abstract: the arXiv form takes plain text (at most 1,920 characters); remove LaTeX markup.
- [ ] After posting: add the arXiv identifier to `README.md`, `CITATION.cff` (as
      `preferred-citation`) and the JOSS draft.
