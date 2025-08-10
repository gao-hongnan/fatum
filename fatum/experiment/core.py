from __future__ import annotations

import json
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Self

from fatum.experiment.constants import (
    DEFAULT_EXPERIMENTS_DIR,
    EXPERIMENT_METADATA_FILE,
    JSON_INDENT,
    METRICS_SUMMARY_FILE,
    PARAMETERS_FILE,
    RUN_METADATA_FILE,
    TEMP_FILE_MODE,
)
from fatum.experiment.exceptions import NotFoundError, StateError, ValidationError
from fatum.experiment.storage import LocalFileStorage, StorageProtocol
from fatum.experiment.types import (
    Artifact,
    ArtifactKey,
    ExperimentConfig,
    ExperimentID,
    ExperimentMetadata,
    ExperimentStatus,
    FilePath,
    Metric,
    MetricKey,
    Parameter,
    ParamKey,
    RunID,
    RunMetadata,
    RunStatus,
    StorageCategories,
    StorageKey,
)
from fatum.reproducibility.git import get_git_info


class Run:
    """
    Represents a single run within an experiment.

    A run tracks metrics, parameters, and artifacts for a specific execution
    of an experiment. Multiple runs can be created within a single experiment
    to compare different configurations or iterations.

    Examples
    --------
    >>> with experiment.run(name="baseline") as run:
    ...     run.log_param("learning_rate", 0.01)
    ...     run.log_metric("accuracy", 0.95)
    ...     run.log_artifact("model.pkl")
    """

    def __init__(
        self,
        run_id: RunID,
        experiment: Experiment,
        name: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.id = run_id
        self.experiment = experiment
        self.metadata = RunMetadata(
            id=run_id,
            experiment_id=experiment.id,
            name=name,
            tags=tags or [],
        )
        self._metrics: list[Metric] = []
        self._parameters: dict[ParamKey, Parameter] = {}
        self._artifacts: dict[ArtifactKey, Artifact] = {}
        self._completed = False

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a metric value for this run."""
        if self._completed:
            raise StateError(self.metadata.status, "log metric")

        metric = Metric(key=MetricKey(key), value=value, step=step)
        self._metrics.append(metric)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter for this run."""
        if self._completed:
            raise StateError(self.metadata.status, "log parameter")

        param = Parameter(key=ParamKey(key), value=value)
        self._parameters[param.key] = param

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_artifact(self, source: FilePath, artifact_name: str | None = None) -> ArtifactKey:
        """Save an artifact file for this run."""
        if self._completed:
            raise StateError(self.metadata.status, "log artifact")

        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "File does not exist")

        name = artifact_name or source_path.name
        artifact_key = ArtifactKey(f"{StorageCategories.ARTIFACTS}/{name}")
        storage_key = StorageKey(f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}/{artifact_key}")

        self.experiment._storage.save(storage_key, source_path)

        artifact = Artifact(
            key=artifact_key,
            storage_key=storage_key,
            path=source_path,
            size_bytes=source_path.stat().st_size if source_path.is_file() else 0,
        )
        self._artifacts[artifact_key] = artifact
        return artifact_key

    def complete(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Complete the run and save all data."""
        if self._completed:
            return

        self.metadata = self.metadata.model_copy(update={"status": status, "ended_at": datetime.now()})
        self._completed = True

        self._save_run_data()

    def _save_run_data(self) -> None:
        """Save run metadata, metrics, and parameters to organized paths."""
        run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"

        metadata_dict = self.metadata.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(mode=TEMP_FILE_MODE, suffix=".json", delete=False) as tmp:
            json.dump(metadata_dict, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{run_base}/{StorageCategories.METADATA}/{RUN_METADATA_FILE}")
            self.experiment._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

        if self._metrics:
            metrics_dict = [m.model_dump(mode="json") for m in self._metrics]
            with tempfile.NamedTemporaryFile(mode=TEMP_FILE_MODE, suffix=".json", delete=False) as tmp:
                json.dump(metrics_dict, tmp, indent=JSON_INDENT)
                tmp_path = Path(tmp.name)

            try:
                storage_key = StorageKey(f"{run_base}/{StorageCategories.METRICS}/{METRICS_SUMMARY_FILE}")
                self.experiment._storage.save(storage_key, tmp_path)
            finally:
                tmp_path.unlink()

        if self._parameters:
            params_dict = {k: p.model_dump(mode="json") for k, p in self._parameters.items()}
            with tempfile.NamedTemporaryFile(mode=TEMP_FILE_MODE, suffix=".json", delete=False) as tmp:
                json.dump(params_dict, tmp, indent=JSON_INDENT)
                tmp_path = Path(tmp.name)

            try:
                storage_key = StorageKey(f"{run_base}/{StorageCategories.PARAMETERS}/{PARAMETERS_FILE}")
                self.experiment._storage.save(storage_key, tmp_path)
            finally:
                tmp_path.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,  # noqa: ARG002
        exc_tb: TracebackType | None,  # noqa: ARG002
    ) -> None:
        if exc_type is not None:
            self.complete(RunStatus.FAILED)
        else:
            self.complete(RunStatus.COMPLETED)


class Experiment:
    """
    Main experiment tracking class with flexible file organization.

    This class provides two approaches for saving files:

    1. **System files** (fixed paths): Internal metadata, run data, etc.
       are saved to predefined locations for consistency.

    2. **User files** (flexible paths): Users have complete control over
       file organization within their experiment directory.

    Parameters
    ----------
    name : str
        Name of the experiment (required if not using config)
    storage : StorageProtocol | None
        Storage backend to use (defaults to LocalFileStorage)
    description : str
        Human-readable description of the experiment
    tags : list[str] | None
        Tags for categorizing the experiment
    run_id_prefix : str
        Prefix for generated run IDs (default: "run")
    config : ExperimentConfig | None
        Configuration object (alternative to individual parameters)

    Examples
    --------
    Basic usage with flexible file organization:

    >>> experiment = Experiment(name="model_training")
    >>>
    >>> # Users control the full path
    >>> experiment.save_dict({"lr": 0.01}, "configs/hyperparams.json")
    >>> experiment.save_text(log_content, "logs/training.log")
    >>> experiment.save_file("model.pkl", "models/best_model.pkl")
    >>>
    >>> # Create custom directory structure
    >>> experiment.save_dict(results, "evaluation/metrics.json")
    >>> experiment.save_text(summary, "reports/summary.md")

    Directory structure created:
    ```
    model_training_abc123/
    ├── metadata/
    │   └── experiment.json      # System file (fixed location)
    ├── configs/
    │   └── hyperparams.json     # User file (user-defined path)
    ├── logs/
    │   └── training.log         # User file (user-defined path)
    ├── models/
    │   └── best_model.pkl       # User file (user-defined path)
    ├── evaluation/
    │   └── metrics.json         # User file (user-defined path)
    └── reports/
        └── summary.md           # User file (user-defined path)
    ```
    """

    def __init__(
        self,
        name: str = "",
        storage: StorageProtocol | None = None,
        description: str = "",
        tags: list[str] | None = None,
        run_id_prefix: str = "run",
        config: ExperimentConfig | None = None,
    ) -> None:
        if config:
            self._init_from_config(config)
        else:
            if not name:
                raise ValidationError("name", name, "Name is required when not using config")
            self.id = ExperimentID(f"{name}_{uuid.uuid4().hex[:8]}")
            self.metadata = ExperimentMetadata(
                id=self.id,
                name=name,
                description=description,
                tags=tags or [],
                git_info=get_git_info().model_dump() if get_git_info() else {},
            )
            self._storage = storage or LocalFileStorage(Path(DEFAULT_EXPERIMENTS_DIR))
            self._run_id_prefix = run_id_prefix

        self._runs: dict[RunID, Run] = {}
        self._artifacts: dict[ArtifactKey, Artifact] = {}
        self._completed = False

        self._save_metadata()

    def _init_from_config(self, config: ExperimentConfig) -> None:
        self.id = ExperimentID(f"{config.name}_{uuid.uuid4().hex[:8]}")
        self.metadata = ExperimentMetadata(
            id=self.id,
            name=config.name,
            description=config.description,
            tags=config.tags,
            git_info=get_git_info().model_dump() if get_git_info() else {},
        )

        # Use run_id_prefix from config
        self._run_id_prefix = config.run_id_prefix

        if config.storage_type == "local":
            self._storage = LocalFileStorage(config.storage_path)
        else:
            raise ValidationError("storage_type", config.storage_type, "Unsupported storage type")

    def start_run(self, name: str = "", tags: list[str] | None = None) -> Run:
        run_id = RunID(f"{self._run_id_prefix}_{uuid.uuid4().hex[:8]}" if self._run_id_prefix else uuid.uuid4().hex[:8])
        run = Run(run_id, self, name, tags)
        self._runs[run_id] = run
        return run

    @contextmanager
    def run(self, name: str = "", tags: list[str] | None = None) -> Iterator[Run]:
        run = self.start_run(name, tags)
        try:
            yield run
        finally:
            run.complete()

    def log_artifact(self, source: FilePath, artifact_name: str | None = None) -> ArtifactKey:
        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "File does not exist")

        name = artifact_name or source_path.name
        artifact_key = ArtifactKey(f"{StorageCategories.ARTIFACTS}/{name}")
        storage_key = StorageKey(f"{self.id}/{artifact_key}")

        self._storage.save(storage_key, source_path)

        artifact = Artifact(
            key=artifact_key,
            storage_key=storage_key,
            path=source_path,
            size_bytes=source_path.stat().st_size if source_path.is_file() else 0,
        )
        self._artifacts[artifact_key] = artifact
        return artifact_key

    def load_artifact(self, artifact_key: ArtifactKey) -> Path:
        if artifact_key not in self._artifacts:
            raise NotFoundError("artifact", artifact_key)

        artifact = self._artifacts[artifact_key]
        return self._storage.load(artifact.storage_key)

    def save_dict(self, data: dict[str, Any], path: str, **tempfile_kwargs: Any) -> StorageKey:
        """
        Save a dictionary as JSON to a user-specified path within the experiment.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to save as JSON
        path : str
            Relative path within the experiment directory (e.g., "configs/model.json")
        **tempfile_kwargs : Any
            Additional kwargs for tempfile.NamedTemporaryFile
            (e.g., dir="/fast/ssd/tmp", prefix="exp_")

        Returns
        -------
        StorageKey
            The storage key for the saved file

        Examples
        --------
        >>> experiment.save_dict({"lr": 0.01}, "hyperparameters/config.json")
        >>> experiment.save_dict({"accuracy": 0.95}, "results/metrics.json")
        >>> # Use custom temp directory for large files
        >>> experiment.save_dict(huge_data, "data.json", dir="/mnt/ssd/tmp")
        """
        tempfile_kwargs.setdefault("mode", TEMP_FILE_MODE)
        tempfile_kwargs.setdefault("suffix", ".json")
        tempfile_kwargs.setdefault("delete", False)

        with tempfile.NamedTemporaryFile(**tempfile_kwargs) as tmp:
            json.dump(data, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{path}")
            self._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_text(self, text: str, path: str, **tempfile_kwargs: Any) -> StorageKey:
        """
        Save text content to a user-specified path within the experiment.

        Parameters
        ----------
        text : str
            Text content to save
        path : str
            Relative path within the experiment directory (e.g., "logs/training.log")
        **tempfile_kwargs : Any
            Additional kwargs for tempfile.NamedTemporaryFile
            (e.g., dir="/fast/ssd/tmp", prefix="log_")

        Returns
        -------
        StorageKey
            The storage key for the saved file

        Examples
        --------
        >>> experiment.save_text("Training started...", "logs/training.log")
        >>> experiment.save_text(readme_content, "README.md")
        >>> # Use custom temp directory
        >>> experiment.save_text(large_log, "debug.log", dir="/tmp", prefix="debug_")
        """
        tempfile_kwargs.setdefault("mode", TEMP_FILE_MODE)
        tempfile_kwargs.setdefault("delete", False)

        with tempfile.NamedTemporaryFile(**tempfile_kwargs) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{path}")
            self._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_file(self, source: Path | str, relative_path: str) -> StorageKey:
        """
        Save a file to a user-specified path within the experiment.

        Parameters
        ----------
        source : Path | str
            Source file path to copy
        relative_path : str
            Relative path within the experiment directory where the file should be saved

        Returns
        -------
        StorageKey
            The storage key for the saved file

        Examples
        --------
        >>> experiment.save_file("/tmp/model.pkl", "models/trained_model.pkl")
        >>> experiment.save_file("data.csv", "datasets/train.csv")
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "File does not exist")

        storage_key = StorageKey(f"{self.id}/{relative_path}")
        self._storage.save(storage_key, source_path)
        return storage_key

    def _save_metadata(self) -> None:
        metadata_dict = self.metadata.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(mode=TEMP_FILE_MODE, suffix=".json", delete=False) as tmp:
            json.dump(metadata_dict, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{StorageCategories.METADATA}/{EXPERIMENT_METADATA_FILE}")
            self._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

    def complete(self) -> None:
        if self._completed:
            return

        self.metadata = self.metadata.model_copy(
            update={"status": ExperimentStatus.COMPLETED, "updated_at": datetime.now()}
        )
        self._save_metadata()
        self._completed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,  # noqa: ARG002
        exc_tb: TracebackType | None,  # noqa: ARG002
    ) -> None:
        if exc_type is not None:
            self.metadata = self.metadata.model_copy(update={"status": ExperimentStatus.FAILED})
        self.complete()


def create_experiment(config: ExperimentConfig) -> Experiment:
    return Experiment(config=config)
