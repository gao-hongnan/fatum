"""Unit tests for experiment tracker context managers and global functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from fatum.experiment import tracker
from fatum.experiment.storage import LocalStorage


@pytest.fixture
def temp_storage(tmp_path: Path) -> LocalStorage:
    """Create a LocalStorage instance with temp directory."""
    return LocalStorage(base_path=tmp_path / "experiments")


@pytest.fixture(autouse=True)
def cleanup_tracker() -> Iterator[None]:
    """Ensure tracker is clean before and after each test."""
    tracker.finish()
    yield
    tracker.finish()


class TestContextManagers:
    """Test experiment and run context managers."""

    def test_experiment_context_manager(self, tmp_path: Path) -> None:
        """Test experiment context manager."""
        with tracker.experiment(
            name="test_exp",
            base_path=tmp_path / "experiments",
            description="Test experiment",
        ) as exp:
            assert tracker.is_active()
            assert tracker.get_experiment() == exp
            assert exp.metadata.name == "test_exp"

        assert not tracker.is_active()
        assert tracker.get_experiment() is None

    def test_run_context_manager(self, tmp_path: Path) -> None:
        """Test run context manager within experiment."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            with tracker.run("test_run", tags=["tag1"]) as r:
                assert tracker.get_run() == r
                assert r.metadata.name == "test_run"
                assert r.metadata.tags == ["tag1"]

            assert tracker.get_run() is None

    def test_run_without_experiment_raises(self) -> None:
        """Test that run without experiment raises error."""
        with pytest.raises(RuntimeError, match="No active experiment"), tracker.run("test_run"):
            pass

    def test_nested_runs(self, tmp_path: Path) -> None:
        """Test multiple sequential runs."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments") as exp:
            with tracker.run("run1") as run1:
                tracker.log({"metric1": 0.1})

            with tracker.run("run2") as run2:
                tracker.log({"metric2": 0.2})

            assert len(exp._runs) == 2
            assert run1._completed
            assert run2._completed


class TestGlobalFunctions:
    """Test global tracker functions."""

    def test_log_metrics(self, tmp_path: Path) -> None:
        """Test logging metrics via global function."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"), tracker.run("test_run"):
            tracker.log({"loss": 0.5, "accuracy": 0.92})
            tracker.log({"val_loss": 0.3}, step=10)

            run = tracker.get_run()
            assert run is not None
            assert len(run._metrics) == 3

    def test_save_dict(self, tmp_path: Path) -> None:
        """Test saving dict via global function."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"), tracker.run("test_run"):
            config: dict[str, Any] = {"lr": 0.01, "batch_size": 32}
            tracker.save_dict(config, "config.json")

            exp_path = tmp_path / "experiments"
            config_files = list(exp_path.rglob("config.json"))
            assert len(config_files) == 1

    def test_save_text(self, tmp_path: Path) -> None:
        """Test saving text via global function."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"), tracker.run("test_run"):
            tracker.save_text("Training complete", "status.txt")

            exp_path = tmp_path / "experiments"
            status_files = list(exp_path.rglob("status.txt"))
            assert len(status_files) == 1
            assert status_files[0].read_text() == "Training complete"

    def test_save_file(self, tmp_path: Path) -> None:
        """Test saving file via global function."""
        source = tmp_path / "source.txt"
        source.write_text("source content")

        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"), tracker.run("test_run"):
            result = tracker.save(source, path="saved_file.txt")
            assert result is not None
            assert len(result) == 1

            exp_path = tmp_path / "experiments"
            saved_files = list(exp_path.rglob("saved_file.txt"))
            assert len(saved_files) == 1
            assert saved_files[0].read_text() == "source content"

    def test_save_with_category(self, tmp_path: Path) -> None:
        """Test saving with category via global function."""
        artifact = tmp_path / "artifact.pkl"
        artifact.write_bytes(b"model data")

        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"), tracker.run("test_run"):
            result = tracker.save(artifact, path="model.pkl", category="artifacts")
            assert result is not None
            assert len(result) == 1

            exp_path = tmp_path / "experiments"
            model_files = list(exp_path.rglob("model.pkl"))
            assert len(model_files) == 1
            assert "artifacts/model.pkl" in str(result[0])

    def test_global_functions_without_run(self, tmp_path: Path) -> None:
        """Test that global functions handle missing run gracefully."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            tracker.log({"metric": 0.5})
            tracker.save_dict({"test": "data"}, "test.json")
            tracker.save_text("text", "test.txt")

            exp_path = tmp_path / "experiments"
            assert not list(exp_path.rglob("test.json"))
            assert not list(exp_path.rglob("test.txt"))


class TestStartRunFinish:
    """Test start_run and finish functions."""

    def test_start_run_manual(self, tmp_path: Path) -> None:
        """Test manually starting and finishing runs."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            run = tracker.start_run("manual_run")
            assert tracker.get_run() == run

            tracker.log({"metric": 0.5})

            tracker.finish()
            assert tracker.get_run() is None
            assert run._completed

    def test_start_run_auto_completes_previous(self, tmp_path: Path) -> None:
        """Test that start_run completes previous run."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            run1 = tracker.start_run("run1")
            tracker.log({"metric1": 0.1})

            run2 = tracker.start_run("run2")
            assert run1._completed
            assert tracker.get_run() == run2

            tracker.finish()

    def test_finish_cleans_everything(self, tmp_path: Path) -> None:
        """Test finish cleans up both run and experiment."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments") as exp, tracker.run("test_run") as run:
            tracker.log({"metric": 0.5})

            tracker.finish()

            assert run._completed
            assert exp._completed
            assert tracker.get_experiment() is None
            assert tracker.get_run() is None


class TestGetters:
    """Test getter functions."""

    def test_get_experiment(self, tmp_path: Path) -> None:
        """Test getting active experiment."""
        assert tracker.get_experiment() is None

        with tracker.experiment("test_exp", base_path=tmp_path / "experiments") as exp:
            assert tracker.get_experiment() == exp

        assert tracker.get_experiment() is None

    def test_get_run(self, tmp_path: Path) -> None:
        """Test getting active run."""
        assert tracker.get_run() is None

        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            assert tracker.get_run() is None

            with tracker.run("test_run") as r:
                assert tracker.get_run() == r

            assert tracker.get_run() is None

    def test_is_active(self, tmp_path: Path) -> None:
        """Test checking if experiment is active."""
        assert not tracker.is_active()

        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            assert tracker.is_active()

        assert not tracker.is_active()


class TestCustomStorage:
    """Test using custom storage backend."""

    def test_experiment_with_custom_storage(self, temp_storage: LocalStorage) -> None:
        """Test providing custom storage to experiment."""
        with tracker.experiment(
            name="custom_storage_exp",
            storage=temp_storage,
        ) as exp:
            assert exp._storage == temp_storage

            with tracker.run("test_run"):
                tracker.save_dict({"test": "data"}, "test.json")

            saved_files = list(temp_storage.base_path.rglob("test.json"))
            assert len(saved_files) == 1


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_experiment_error_cleanup(self, tmp_path: Path) -> None:
        """Test that experiment cleans up on error."""
        with pytest.raises(ValueError), tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            raise ValueError("Test error")

        assert not tracker.is_active()
        assert tracker.get_experiment() is None

    def test_run_error_cleanup(self, tmp_path: Path) -> None:
        """Test that run cleans up on error."""
        with tracker.experiment("test_exp", base_path=tmp_path / "experiments"):
            run = None
            with pytest.raises(ValueError), tracker.run("error_run") as r:
                run = r
                raise ValueError("Test error")

            assert run is not None
            assert run._completed
            assert tracker.get_run() is None
