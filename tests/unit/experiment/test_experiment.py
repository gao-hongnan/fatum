"""Unit tests for Experiment and Run classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fatum.experiment.exceptions import StateError, ValidationError
from fatum.experiment.experiment import Experiment
from fatum.experiment.storage import LocalStorage
from fatum.experiment.types import (
    ExperimentID,
    ExperimentStatus,
    MetricKey,
    RunStatus,
    StorageCategories,
)


@pytest.fixture
def temp_storage(tmp_path: Path) -> LocalStorage:
    """Create a LocalStorage instance with temp directory."""
    return LocalStorage(base_path=tmp_path / "experiments")


@pytest.fixture
def experiment(temp_storage: LocalStorage) -> Experiment:
    """Create a test experiment."""
    return Experiment(
        name="test_experiment",
        storage=temp_storage,
        description="Test experiment for unit tests",
        tags=["test", "unit"],
    )


class TestExperiment:
    """Test Experiment class functionality."""

    def test_experiment_creation(self, temp_storage: LocalStorage) -> None:
        """Test creating an experiment with metadata."""
        exp = Experiment(
            name="my_experiment",
            storage=temp_storage,
            description="Test description",
            tags=["tag1", "tag2"],
        )

        assert exp.metadata.name == "my_experiment"
        assert exp.metadata.description == "Test description"
        assert exp.metadata.tags == ["tag1", "tag2"]
        assert exp.metadata.status == ExperimentStatus.RUNNING
        assert exp.id.startswith("my_experiment_")

    def test_experiment_with_custom_id(self, temp_storage: LocalStorage) -> None:
        """Test creating experiment with custom ID."""
        exp = Experiment(
            name="test",
            id="custom_id_123",
            storage=temp_storage,
        )

        assert exp.id == ExperimentID("custom_id_123")

    def test_start_run(self, experiment: Experiment) -> None:
        """Test starting a new run."""
        run = experiment.start_run(name="test_run", tags=["run_tag"])

        assert run.metadata.name == "test_run"
        assert run.metadata.tags == ["run_tag"]
        assert run.metadata.experiment_id == experiment.id
        assert run.metadata.status == RunStatus.RUNNING
        assert run.id in experiment._runs

    def test_run_context_manager(self, experiment: Experiment) -> None:
        """Test run context manager."""
        with experiment.run(name="context_run") as run:
            assert run.metadata.status == RunStatus.RUNNING
            run.log_metric(MetricKey("test_metric"), 0.5)

        assert run._completed
        status = cast(RunStatus, run.metadata.status)
        assert status == RunStatus.COMPLETED

    def test_run_context_manager_with_error(self, experiment: Experiment) -> None:
        """Test run context manager handles errors."""
        run = None
        with pytest.raises(ValueError), experiment.run(name="error_run") as r:
            run = r
            assert run.metadata.status == RunStatus.RUNNING
            raise ValueError("Test error")

        assert run is not None
        assert run._completed
        status = cast(RunStatus, run.metadata.status)
        assert status == RunStatus.FAILED

    def test_experiment_context_manager(self, temp_storage: LocalStorage) -> None:
        """Test experiment context manager."""
        with Experiment(name="context_exp", storage=temp_storage) as exp:
            assert exp.metadata.status == ExperimentStatus.RUNNING
            exp.start_run("test_run")

        assert exp._completed
        status = cast(ExperimentStatus, exp.metadata.status)
        assert status == ExperimentStatus.COMPLETED

    def test_experiment_completion(self, experiment: Experiment) -> None:
        """Test completing an experiment."""
        assert experiment.metadata.status == ExperimentStatus.RUNNING

        experiment.complete()
        assert experiment._completed
        status = cast(ExperimentStatus, experiment.metadata.status)
        assert status == ExperimentStatus.COMPLETED

        experiment.complete()
        assert experiment._completed


class TestRun:
    """Test Run class functionality."""

    def test_log_metric(self, experiment: Experiment, temp_storage: LocalStorage) -> None:
        """Test logging metrics."""
        run = experiment.start_run("metric_run")

        run.log_metric(MetricKey("loss"), 0.5, step=0)
        run.log_metric(MetricKey("accuracy"), 0.92, step=1)

        assert len(run._metrics) == 2
        assert run._metrics[0].key == MetricKey("loss")
        assert run._metrics[0].value == 0.5
        assert run._metrics[1].key == MetricKey("accuracy")
        assert run._metrics[1].value == 0.92

        metrics_dir = (
            temp_storage.base_path / experiment.id / StorageCategories.RUNS / run.id / StorageCategories.METRICS
        )
        assert metrics_dir.exists()
        assert len(list(metrics_dir.glob("*.json"))) == 2

    def test_log_metrics_batch(self, experiment: Experiment) -> None:
        """Test logging multiple metrics at once."""
        run = experiment.start_run("batch_metrics")

        metrics = {"loss": 0.3, "accuracy": 0.95, "f1_score": 0.88}
        run.log_metrics(metrics, step=10)

        assert len(run._metrics) == 3

    def test_save_file(self, experiment: Experiment, tmp_path: Path) -> None:
        """Test saving a single file."""
        run = experiment.start_run("file_run")

        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")

        keys = run.save(test_file, path="my_file.txt")

        assert len(keys) == 1
        assert "my_file.txt" in str(keys[0])

    def test_save_directory(self, experiment: Experiment, tmp_path: Path) -> None:
        """Test saving a directory."""
        run = experiment.start_run("dir_run")

        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file2.txt").write_text("content2")

        keys = run.save(test_dir, path="my_dir")

        assert len(keys) == 2
        assert any("my_dir/file1.txt" in str(k) for k in keys)
        assert any("my_dir/subdir/file2.txt" in str(k) for k in keys)

    def test_save_dict(self, experiment: Experiment, temp_storage: LocalStorage) -> None:
        """Test saving dictionary as JSON."""
        run = experiment.start_run("dict_run")

        data: dict[str, Any] = {"config": {"lr": 0.01, "batch_size": 32}}
        key = run.save_dict(data, "config.json", indent=2)

        saved_path = temp_storage.base_path / key
        assert saved_path.exists()

        loaded = json.loads(saved_path.read_text())
        assert loaded == data

    def test_save_text(self, experiment: Experiment, temp_storage: LocalStorage) -> None:
        """Test saving text content."""
        run = experiment.start_run("text_run")

        text = "Training log:\nEpoch 1: loss=0.5\nEpoch 2: loss=0.3"
        key = run.save_text(text, "training.log")

        saved_path = temp_storage.base_path / key
        assert saved_path.exists()
        assert saved_path.read_text() == text

    def test_save_with_category(self, experiment: Experiment, tmp_path: Path, temp_storage: LocalStorage) -> None:
        """Test saving with category prefix."""
        run = experiment.start_run("category_run")

        source = tmp_path / "model.pkl"
        source.write_bytes(b"model data")

        keys = run.save(source, path="best_model.pkl", category="models")

        assert len(keys) == 1
        saved_path = temp_storage.base_path / keys[0]
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"model data"
        assert "models/best_model.pkl" in str(keys[0])

    def test_complete_run(self, experiment: Experiment, temp_storage: LocalStorage) -> None:
        """Test completing a run."""
        run = experiment.start_run("complete_run")

        assert run.metadata.status == RunStatus.RUNNING

        run.complete()

        assert run._completed
        status = cast(RunStatus, run.metadata.status)
        assert status == RunStatus.COMPLETED
        assert run.metadata.ended_at is not None

        metadata_path = (
            temp_storage.base_path
            / experiment.id
            / StorageCategories.RUNS
            / run.id
            / StorageCategories.METADATA
            / "run.json"
        )
        assert metadata_path.exists()

    def test_run_state_errors(self, experiment: Experiment) -> None:
        """Test operations on completed run raise errors."""
        run = experiment.start_run("state_run")
        run.complete()

        with pytest.raises(StateError):
            run.log_metric(MetricKey("test"), 0.5)

        with pytest.raises(StateError):
            run.save_dict({"test": "data"}, "test.json")

        with pytest.raises(StateError):
            run.save_text("test", "test.txt")

        with pytest.raises(StateError):
            run.save(Path("."), path="test")

    def test_invalid_artifact_source(self, experiment: Experiment) -> None:
        """Test saving non-existent artifact raises error."""
        run = experiment.start_run("invalid_run")

        with pytest.raises(ValidationError) as exc_info:
            run.save("/non/existent/path")

        assert exc_info.value.field == "source"
        assert "does not exist" in str(exc_info.value)


class TestValidation:
    """Test validation and error handling."""

    def test_invalid_experiment_id(self, temp_storage: LocalStorage) -> None:
        """Test invalid experiment ID is rejected."""
        with pytest.raises(ValueError, match="Invalid experiment ID"):
            Experiment(
                name="test",
                id="/invalid/id",
                storage=temp_storage,
            )

    def test_invalid_metric_key(self, experiment: Experiment) -> None:
        """Test invalid metric key is rejected."""
        run = experiment.start_run("validation_run")

        with pytest.raises(ValueError, match="Invalid metric key"):
            run.log_metric(MetricKey("invalid key with spaces!"), 0.5)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_multiple_runs(self, experiment: Experiment) -> None:
        """Test multiple runs in an experiment."""
        runs = []
        for i in range(3):
            with experiment.run(name=f"run_{i}") as run:
                run.log_metric(MetricKey("metric"), float(i))
                runs.append(run)

        assert len(experiment._runs) == 3
        for run in runs:
            assert run._completed

    def test_nested_contexts(self, temp_storage: LocalStorage) -> None:
        """Test nested experiment and run contexts."""
        with Experiment(name="nested", storage=temp_storage) as exp:
            with exp.run("run1") as run1:
                run1.log_metric(MetricKey("m1"), 0.1)

            with exp.run("run2") as run2:
                run2.log_metric(MetricKey("m2"), 0.2)

        assert exp._completed
        assert run1._completed
        assert run2._completed

    def test_run_without_name(self, experiment: Experiment) -> None:
        """Test creating run without explicit name."""
        run = experiment.start_run()

        assert run.metadata.name.startswith("run_")
        assert run.id.startswith("run_")
