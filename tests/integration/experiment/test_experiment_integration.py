"""Integration tests for the experiment tracking system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fatum.experiment import experiment, run
from fatum.experiment.storage import LocalStorage
from fatum.experiment.types import StorageCategories


class TestExperimentIntegration:
    """Integration tests for complete experiment workflows."""

    def test_full_experiment_workflow(self, tmp_path: Path) -> None:
        """Test a complete experiment workflow end-to-end."""
        base_path = tmp_path / "experiments"

        with experiment(
            name="integration_test",
            base_path=base_path,
            description="Full integration test",
            tags=["test", "integration"],
        ) as exp:
            with run("training", tags=["train"]) as r:
                r.save_dict(
                    {"learning_rate": 0.01, "batch_size": 32, "epochs": 10},
                    "hyperparameters.json",
                )

                for epoch in range(3):
                    r.log_metrics(
                        {
                            "loss": 1.0 / (epoch + 1),
                            "accuracy": epoch * 0.3,
                        },
                        step=epoch,
                    )

                model_path = tmp_path / "model.pkl"
                model_path.write_bytes(b"trained model data")
                r.save(model_path, path="models/final_model.pkl")

            with run("evaluation", tags=["eval"]) as r:
                r.log_metrics(
                    {
                        "test_accuracy": 0.92,
                        "test_f1": 0.89,
                        "test_precision": 0.91,
                        "test_recall": 0.87,
                    }
                )

                report = """Evaluation Report
                ==================
                Test Accuracy: 92%
                F1 Score: 0.89
                Precision: 0.91
                Recall: 0.87
                """
                r.save_text(report, "reports/evaluation.txt")

            with run("inference", tags=["inference", "production"]) as r:
                predictions: dict[str, Any] = {
                    "predictions": [0.95, 0.12, 0.88, 0.67],
                    "labels": ["cat", "dog", "cat", "cat"],
                }
                r.save_dict(predictions, "predictions.json", indent=2)

        exp_dir = base_path / exp.id

        exp_metadata_path = exp_dir / StorageCategories.METADATA / "experiment.json"
        assert exp_metadata_path.exists()
        exp_metadata = json.loads(exp_metadata_path.read_text())
        assert exp_metadata["name"] == "integration_test"
        assert exp_metadata["status"] == "completed"

        runs_dir = exp_dir / "runs"
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 3

        training_runs = [d for d in run_dirs if "training" in (d / StorageCategories.METADATA / "run.json").read_text()]
        assert len(training_runs) == 1
        training_dir = training_runs[0]

        training_run_metadata = json.loads((training_dir / StorageCategories.METADATA / "run.json").read_text())
        assert "train" in training_run_metadata["tags"]

        hyperparam_path = training_dir / "hyperparameters.json"
        assert hyperparam_path.exists()
        hyperparams = json.loads(hyperparam_path.read_text())
        assert hyperparams["learning_rate"] == 0.01

        metrics_dir = training_dir / StorageCategories.METRICS
        assert metrics_dir.exists()
        metric_files = list(metrics_dir.glob("*.json"))
        assert len(metric_files) == 6

        model_path = training_dir / "models" / "final_model.pkl"
        assert model_path.exists()
        assert model_path.read_bytes() == b"trained model data"

        eval_runs = [d for d in run_dirs if "evaluation" in (d / StorageCategories.METADATA / "run.json").read_text()]
        assert len(eval_runs) == 1
        eval_dir = eval_runs[0]

        report_path = eval_dir / "reports" / "evaluation.txt"
        assert report_path.exists()
        assert "Test Accuracy: 92%" in report_path.read_text()

    def test_experiment_with_custom_storage(self, tmp_path: Path) -> None:
        """Test experiment with custom storage backend."""
        storage_path = tmp_path / "custom_storage"
        custom_storage = LocalStorage(base_path=storage_path)

        with (
            experiment(
                name="custom_storage_test",
                storage=custom_storage,
            ),
            run("test_run") as r,
        ):
            r.save_dict({"test": "data"}, "test.json")
            r.log_metrics({"metric": 0.5})

        assert storage_path.exists()
        exp_files = list(storage_path.rglob("*.json"))
        assert len(exp_files) > 0
        assert any("test.json" in str(f) for f in exp_files)

    def test_experiment_error_recovery(self, tmp_path: Path) -> None:
        """Test that experiment handles errors gracefully."""
        with pytest.raises(ValueError), experiment("error_test", base_path=tmp_path / "experiments"):
            with run("run1") as r:
                r.log_metrics({"metric": 0.5})

            with run("run2") as r:
                r.log_metrics({"metric": 0.3})
                raise ValueError("Simulated error")

        exp_dir = tmp_path / "experiments"
        run_files = list(exp_dir.rglob("run.json"))
        assert len(run_files) >= 1

        for run_file in run_files:
            run_data = json.loads(run_file.read_text())
            if run_data["name"] == "run1":
                assert run_data["status"] == "completed"
            elif run_data["name"] == "run2":
                assert run_data["status"] == "failed"

    def test_nested_experiment_contexts(self, tmp_path: Path) -> None:
        """Test running multiple experiments sequentially."""
        base_path = tmp_path / "experiments"

        with experiment("exp1", base_path=base_path) as exp1, run("exp1_run") as r:
            r.log_metrics({"exp1_metric": 0.1})

        with experiment("exp2", base_path=base_path) as exp2, run("exp2_run") as r:
            r.log_metrics({"exp2_metric": 0.2})

        assert (base_path / exp1.id).exists()
        assert (base_path / exp2.id).exists()

        exp1_metrics = list((base_path / exp1.id).rglob("*exp1_metric.json"))
        exp2_metrics = list((base_path / exp2.id).rglob("*exp2_metric.json"))
        assert len(exp1_metrics) == 1
        assert len(exp2_metrics) == 1

    def test_large_artifact_handling(self, tmp_path: Path) -> None:
        """Test handling of large artifacts and directories."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        for i in range(5):
            (artifacts_dir / f"file_{i}.txt").write_text(f"Content {i}" * 100)

        subdir = artifacts_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content")

        with experiment("large_test", base_path=tmp_path / "experiments"), run("artifact_run") as r:
            keys = r.save(artifacts_dir, path="all_artifacts")

            assert len(keys) == 6

            for i in range(5):
                expected_key = f"all_artifacts/file_{i}.txt"
                assert any(expected_key in str(k) for k in keys)

            assert any("all_artifacts/subdir/nested.txt" in str(k) for k in keys)

    def test_concurrent_runs_data_isolation(self, tmp_path: Path) -> None:
        """Test that data from different runs is properly isolated."""
        with experiment("isolation_test", base_path=tmp_path / "experiments") as exp:
            run_data = {}

            for i in range(3):
                with run(f"run_{i}") as r:
                    config = {"run_id": i, "value": i * 10}
                    r.save_dict(config, "config.json")
                    r.log_metrics({"metric": float(i)})
                    run_data[r.id] = config

        exp_dir = tmp_path / "experiments" / exp.id
        for run_id, expected_config in run_data.items():
            config_path = exp_dir / "runs" / run_id / "config.json"  # Default run container
            assert config_path.exists()

            loaded_config = json.loads(config_path.read_text())
            assert loaded_config == expected_config

    def test_experiment_metadata_persistence(self, tmp_path: Path) -> None:
        """Test that experiment metadata is properly persisted."""
        exp_id = None

        with experiment(
            name="metadata_test",
            base_path=tmp_path / "experiments",
            description="Testing metadata",
            tags=["meta", "test"],
        ) as exp:
            exp_id = exp.id

            with run("test_run") as r:
                r.log_metrics({"metric": 0.5})

        metadata_path = tmp_path / "experiments" / exp_id / StorageCategories.METADATA / "experiment.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["name"] == "metadata_test"
        assert metadata["description"] == "Testing metadata"
        assert "meta" in metadata["tags"]
        assert "test" in metadata["tags"]
        assert metadata["status"] == "completed"
        assert "created_at" in metadata
        assert "updated_at" in metadata
