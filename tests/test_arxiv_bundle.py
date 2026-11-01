"""tools/arxiv_bundle.py: dependency discovery, readiness checks, bundle contents."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "arxiv_bundle", Path(__file__).resolve().parent.parent / "tools" / "arxiv_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)

TEX = r"""\newcommand{\pending}[1]{#1}
% a comment that says \pending{x}
\input{generated/table.tex}
\includegraphics[width=0.5\linewidth]{generated/fig.pdf}
\input{generated/table}
\pending{still to do}
"""


def make_paper(root: Path, tex: str) -> Path:
    (root / "generated").mkdir(parents=True)
    (root / "main.tex").write_text(tex, encoding="utf-8")
    (root / "references.bib").write_text("", encoding="utf-8")
    (root / "main.bbl").write_text("", encoding="utf-8")
    (root / "generated" / "table.tex").write_text("", encoding="utf-8")
    (root / "generated" / "fig.pdf").write_bytes(b"%PDF")
    return root


def test_dependencies_and_unfinished_markers() -> None:
    assert bundle.dependencies(TEX) == ["generated/table.tex", "generated/fig.pdf"]
    assert bundle.unfinished(TEX) == [6]


def test_refuses_unfinished_text_unless_draft(tmp_path: Path) -> None:
    paper = make_paper(tmp_path, TEX)
    assert bundle.main(["--paper", str(paper)]) == 1
    assert not (paper / "arxiv.zip").exists()
    assert bundle.main(["--paper", str(paper), "--draft"]) == 0
    names = zipfile.ZipFile(paper / "arxiv.zip").namelist()
    assert names == ["main.tex", "main.bbl", "generated/table.tex", "generated/fig.pdf"]


@pytest.mark.parametrize("missing", ["main.bbl", "generated/fig.pdf"])
def test_refuses_missing_files(tmp_path: Path, missing: str) -> None:
    paper = make_paper(tmp_path, TEX.replace(r"\pending{still to do}", ""))
    (paper / missing).unlink()
    assert bundle.main(["--paper", str(paper)]) == 1
