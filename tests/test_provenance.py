"""Tests for nisq_folio.provenance."""

import re
import subprocess
from pathlib import Path

import pytest

from nisq_folio.provenance import git_state


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def make_repo(path: Path) -> Path:
    git(path, "init", "-q")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    (path / "file.txt").write_text("one\n", encoding="utf-8")
    git(path, "add", "file.txt")
    git(path, "commit", "-q", "-m", "init")
    return path


def test_clean_repo_reports_full_sha_and_not_dirty(tmp_path: Path) -> None:
    state = git_state(make_repo(tmp_path))
    assert state.commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}", state.commit)
    assert state.dirty is False


def test_modified_tracked_file_marks_dirty(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "file.txt").write_text("two\n", encoding="utf-8")
    assert git_state(repo).dirty is True


def test_untracked_files_do_not_mark_dirty(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert git_state(repo).dirty is False


def test_outside_a_repository_commit_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stop git searching parent directories, in case the temp dir sits inside some repo.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    state = git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is False
