"""Package the paper sources for arXiv (not part of the research code).

Usage, after building the paper so that ``paper/main.bbl`` exists::

    tectonic -X compile --keep-intermediates paper/main.tex
    uv run python tools/arxiv_bundle.py            # refuses while pending/todo markers remain
    uv run python tools/arxiv_bundle.py --draft    # bundle anyway, for a test upload

arXiv compiles with pdfLaTeX and does not run BibTeX, so the bundle holds ``main.tex``, the
generated ``main.bbl``, and exactly the tables and figures that ``main.tex`` reads with its input
and includegraphics commands. The result is ``paper/arxiv.zip``.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
DEPENDENCY = re.compile(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}")
UNFINISHED = re.compile(r"\\(?:pending|todo)\{")


def dependencies(tex: str) -> list[str]:
    """Relative paths read by the input and includegraphics commands, in order, no repeats."""
    found = []
    for name in DEPENDENCY.findall(tex):
        name = name if Path(name).suffix else f"{name}.tex"
        if name not in found:
            found.append(name)
    return found


def unfinished(tex: str) -> list[int]:
    """Line numbers that use the pending or todo macros (definitions and comments excepted)."""
    return [
        i
        for i, line in enumerate(tex.splitlines(), start=1)
        if UNFINISHED.search(line) and "\\newcommand" not in line and not line.startswith("%")
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Write the arXiv bundle; returns 0 on success and 1 if the sources are not ready."""
    parser = argparse.ArgumentParser(prog="tools/arxiv_bundle.py")
    parser.add_argument("--paper", type=Path, default=PAPER, help="paper directory")
    parser.add_argument("--draft", action="store_true", help="allow \\pending and \\todo")
    args = parser.parse_args(argv)
    paper: Path = args.paper
    main_tex, bbl = paper / "main.tex", paper / "main.bbl"
    tex = main_tex.read_text(encoding="utf-8")

    problems = []
    if not args.draft and (lines := unfinished(tex)):
        problems.append(f"\\pending or \\todo still used on lines {lines}")
    if not bbl.exists():
        problems.append("main.bbl is missing: build with tectonic --keep-intermediates first")
    elif any(bbl.stat().st_mtime < p.stat().st_mtime for p in paper.glob("*.bib")):
        problems.append("main.bbl is older than the bibliography: rebuild the paper")
    files = ["main.tex", "main.bbl", *dependencies(tex)]
    problems += [f"missing {f}" for f in files[2:] if not (paper / f).exists()]
    if problems:
        print("not bundled:\n  " + "\n  ".join(problems))
        return 1

    out = paper / "arxiv.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(paper / f, f)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB): {', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
