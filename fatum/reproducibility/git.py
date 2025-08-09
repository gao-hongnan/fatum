from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel


class GitInfo(BaseModel):
    commit: str | None = None
    short_commit: str | None = None
    branch: str | None = None
    is_dirty: bool = False

    @classmethod
    def current(cls, repo_path: Path | None = None) -> GitInfo:
        return get_git_info(repo_path)


@dataclass
class GitRepository:
    path: Path | None = None

    def _run_git(self, args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=self.path, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @cached_property
    def commit(self) -> str | None:
        return self._run_git(["rev-parse", "HEAD"])

    @cached_property
    def short_commit(self) -> str | None:
        return self._run_git(["rev-parse", "--short=7", "HEAD"])

    @cached_property
    def branch(self) -> str | None:
        return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    @cached_property
    def is_dirty(self) -> bool:
        """Whether the repository has uncommitted changes."""
        return bool(self._run_git(["status", "--porcelain"]))

    @cached_property
    def info(self) -> GitInfo:
        return GitInfo(
            commit=self.commit,
            short_commit=self.short_commit,
            branch=self.branch,
            is_dirty=self.is_dirty,
        )


def get_git_info(repo_path: Path | None = None) -> GitInfo:
    return GitRepository(repo_path).info
