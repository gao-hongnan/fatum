from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import Mock, patch

import pytest

from fatum.reproducibility.git import GitInfo, GitRepository, get_git_info


class TestGitRepository:
    @pytest.fixture
    def mock_subprocess_run(self) -> Iterator[Mock]:
        with patch("subprocess.run") as mock_run:
            yield mock_run

    @pytest.fixture
    def git_repo(self) -> GitRepository:
        return GitRepository(Path("/fake/repo"))

    def test_run_git_success(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = "  test output  \n"
        mock_subprocess_run.return_value = mock_result

        result = git_repo._run_git(["status"])

        assert result == "test output"
        mock_subprocess_run.assert_called_once_with(
            ["git", "status"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo"),
            check=True,
        )

    def test_run_git_subprocess_error(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        import subprocess

        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "git")

        result = git_repo._run_git(["status"])

        assert result is None

    def test_run_git_file_not_found(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_subprocess_run.side_effect = FileNotFoundError()

        result = git_repo._run_git(["status"])

        assert result is None

    def test_commit_property(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = "abc123def456789\n"
        mock_subprocess_run.return_value = mock_result

        commit = git_repo.commit

        assert commit == "abc123def456789"
        mock_subprocess_run.assert_called_with(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo"),
            check=True,
        )

    def test_short_commit_property(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = "abc123d\n"
        mock_subprocess_run.return_value = mock_result

        short_commit = git_repo.short_commit

        assert short_commit == "abc123d"
        mock_subprocess_run.assert_called_with(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo"),
            check=True,
        )

    def test_branch_property(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = "main\n"
        mock_subprocess_run.return_value = mock_result

        branch = git_repo.branch

        assert branch == "main"
        mock_subprocess_run.assert_called_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo"),
            check=True,
        )

    def test_is_dirty_clean_repo(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = ""
        mock_subprocess_run.return_value = mock_result

        is_dirty = git_repo.is_dirty

        assert is_dirty is False
        mock_subprocess_run.assert_called_with(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo"),
            check=True,
        )

    def test_is_dirty_with_changes(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = " M file.txt\n?? newfile.py\n"
        mock_subprocess_run.return_value = mock_result

        is_dirty = git_repo.is_dirty

        assert is_dirty is True

    def test_info_property(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        responses: dict[tuple[str, ...], str] = {
            ("rev-parse", "HEAD"): "abc123def456789",
            ("rev-parse", "--short=7", "HEAD"): "abc123d",
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature-branch",
            ("status", "--porcelain"): " M modified.txt",
        }

        def mock_run(cmd: list[str], **_kwargs: Any) -> Mock:
            git_args = tuple(cmd[1:])
            mock_result = Mock()
            mock_result.stdout = responses.get(git_args, "") + "\n"
            return mock_result

        mock_subprocess_run.side_effect = mock_run

        info = git_repo.info

        assert isinstance(info, GitInfo)
        assert info.commit == "abc123def456789"
        assert info.short_commit == "abc123d"
        assert info.branch == "feature-branch"
        assert info.is_dirty is True

    def test_cached_property_behavior(self, git_repo: GitRepository, mock_subprocess_run: Mock) -> None:
        mock_result = Mock()
        mock_result.stdout = "main\n"
        mock_subprocess_run.return_value = mock_result

        branch1 = git_repo.branch
        branch2 = git_repo.branch
        branch3 = git_repo.branch

        assert branch1 == branch2 == branch3 == "main"
        assert mock_subprocess_run.call_count == 1

    def test_none_path_uses_current_directory(self, mock_subprocess_run: Mock) -> None:
        repo = GitRepository(None)
        mock_result = Mock()
        mock_result.stdout = "test\n"
        mock_subprocess_run.return_value = mock_result

        result = repo._run_git(["test"])

        assert result == "test"
        mock_subprocess_run.assert_called_with(
            ["git", "test"],
            capture_output=True,
            text=True,
            cwd=None,
            check=True,
        )


class TestGitInfo:
    def test_default_values(self) -> None:
        info = GitInfo()

        assert info.commit is None
        assert info.short_commit is None
        assert info.branch is None
        assert info.is_dirty is False

    def test_with_values(self) -> None:
        info = GitInfo(
            commit="abc123",
            short_commit="abc",
            branch="main",
            is_dirty=True,
        )

        assert info.commit == "abc123"
        assert info.short_commit == "abc"
        assert info.branch == "main"
        assert info.is_dirty is True

    @patch("fatum.reproducibility.git.get_git_info")
    def test_current_method(self, mock_get_git_info: Mock) -> None:
        mock_info = GitInfo(commit="test123")
        mock_get_git_info.return_value = mock_info

        result = GitInfo.current(Path("/some/path"))

        assert result == mock_info
        mock_get_git_info.assert_called_once_with(Path("/some/path"))

    @patch("fatum.reproducibility.git.get_git_info")
    def test_current_method_no_path(self, mock_get_git_info: Mock) -> None:
        mock_info = GitInfo(branch="develop")
        mock_get_git_info.return_value = mock_info

        result = GitInfo.current()

        assert result == mock_info
        mock_get_git_info.assert_called_once_with(None)


class TestGetGitInfo:
    @patch("fatum.reproducibility.git.GitRepository")
    def test_get_git_info_with_path(self, mock_git_repo_class: Mock) -> None:
        mock_repo = Mock()
        mock_info = GitInfo(commit="xyz789")
        mock_repo.info = mock_info
        mock_git_repo_class.return_value = mock_repo

        result = get_git_info(Path("/test/repo"))

        assert result == mock_info
        mock_git_repo_class.assert_called_once_with(Path("/test/repo"))

    @patch("fatum.reproducibility.git.GitRepository")
    def test_get_git_info_no_path(self, mock_git_repo_class: Mock) -> None:
        mock_repo = Mock()
        mock_info = GitInfo(branch="master")
        mock_repo.info = mock_info
        mock_git_repo_class.return_value = mock_repo

        result = get_git_info()

        assert result == mock_info
        mock_git_repo_class.assert_called_once_with(None)
