# Submission checklist

Everything left between today and a submitted paper, in order. Each item says **who** does it,
**how**, and **why** it matters. Tick items off as you go.

Status on 2026-11-20: the research, code and paper are complete, and both authors have read and
approved the paper as final. The publication route is fixed by ADR-031: **Quantum Information
Processing (Springer), no-fee route, no arXiv preprint, no Zenodo.** The submission date is the
authors' choice.

**Project closed on 2026-11-20 as complete (release v1.0.0).** Submission is optional and may
happen later or not at all; the steps below remain valid if the authors decide to go ahead.

Who: **Y** = Yuvaraj, **K** = Kallol, **C** = Claude (on request).

---

## Done

- [x] Both authors read the paper and agreed on the interpretation and the AI-use statement.
- [x] Journal chosen: Quantum Information Processing, no-fee route.
- [x] LICENSE names both authors.
- [x] ORCID iDs: skipped.
- [x] arXiv preprint and Zenodo archive: not used (ADR-031).

## Before sharing publicly (LinkedIn, CV, portfolio)

- [ ] **Describe the work accurately: "completed study, to be submitted", not "published".**
  - **How:** use wording like "completed and prepared for submission to Quantum Information
    Processing". Link the GitHub repository and SHOWCASE.md.
  - **Why:** "published" means accepted by a journal after peer review. Claiming it early can be
    seen as misrepresentation by employers, the school or the journal, and it is easy to avoid.
    A completed, fully open study is already a strong thing to show.

- [ ] **Decide whether to post the PDF itself.**
  - **How:** check Springer's preprint policy (on the journal's "Submission guidelines" page)
    before posting the full paper. Posting the repository link and the showcase is always safe.
  - **Why:** some journals treat a publicly posted manuscript as prior publication; many Springer
    journals allow it, but confirm.

## Freeze the version (C, 10 minutes, any time)

- [x] **Tag v1.0.0 on GitHub.** Done 2026-11-20.
  - **How:** say "tag the release". C sets version 1.0.0 in `pyproject.toml`, `uv.lock` and
    `CITATION.cff`, runs the checks, commits, then (on your go-ahead) creates the tag.
  - **Why:** free, and it fixes the exact code the paper describes, so later changes cannot blur
    what was reviewed. Without Zenodo this is the only version marker the paper can point to.

## Submit to Quantum Information Processing (when you decide)

- [ ] **1. Check the journal's current guidelines.**
  - **How:** open the journal's page on springer.com, read "Submission guidelines": manuscript
    format and LaTeX template, length, AI policy, competing-interests and data statements.
    Choose the standard (subscription) route, which is free. Do **not** choose open access, which
    charges a fee.
  - **Why:** formats and policies change; the zero-cost rule (ADR-008) depends on the route.

- [ ] **2. Reformat to the journal's template (C, then Y and K check).**
  - **How:** tell C you are ready to submit. C moves the paper into Springer's LaTeX template
    without changing its content, rebuilds, and you both check the PDF.
  - **Why:** papers that ignore the format are often returned before review.

- [ ] **3. Cover letter (C drafts, you edit).**
  - **How:** one page: what the study does, the main finding, why it fits the journal, and that
    the analysis was pre-registered with all code public.
  - **Why:** editors decide quickly whether to send a paper to review.

- [ ] **4. Suggested reviewers, if asked.**
  - **How:** 3 to 5 researchers who work on QAOA mixers, quantum portfolio optimisation or NISQ
    noise, with no conflict of interest; the paper's references are a starting point. Verify
    names and emails yourselves.
  - **Why:** good suggestions speed up review.

- [ ] **5. Submit through the journal's online system (Y).**
  - **How:** upload the PDF and LaTeX sources, the cover letter, and the statements (AI use,
    competing interests, data and code availability pointing to the GitHub repository). Both
    authors with affiliations and emails as in the paper.

- [ ] **6. Tell C the submission date.**
  - **Why:** C records it and keeps the repository at v1.0.0 so the reviewed version stays
    reproducible. Later fixes become v1.0.1 and are noted.

## Later

- [ ] **Responding to reviewers:** C can help draft replies and run any extra analysis, recorded as
  a new, clearly labelled post hoc ADR, never replacing the pre-registered results.
- [ ] **JOSS software paper (ADR-029):** possible after the article is published, since without a
  preprint the article is the evidence of research use. Confirm Kallol's software role first.
