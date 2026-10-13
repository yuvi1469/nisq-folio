"""Provenance helpers: record which code version produced an artifact (CLAUDE.md rule 6)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitState:
    """The git state of a working tree.

    Attributes:
        commit: Full SHA of ``HEAD``, or ``None`` if not inside a git repository.
        dirty: True if tracked files have uncommitted changes (then ``commit`` alone does not
            fully describe the code that ran).
    """

    commit: str | None
    dirty: bool


def git_state(directory: Path) -> GitState:
    """Return the git commit and dirty flag for the repository containing ``directory``.

    Args:
        directory: Any directory inside the working tree.

    Returns:
        The git state. ``commit`` is ``None`` if ``directory`` is not in a git repository
        or git is unavailable.
    """
    commit = _git(directory, "rev-parse", "HEAD")
    if commit is None:
        return GitState(commit=None, dirty=False)
    status = _git(directory, "status", "--porcelain", "--untracked-files=no")
    return GitState(commit=commit, dirty=bool(status))


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
