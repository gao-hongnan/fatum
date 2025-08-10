from __future__ import annotations

import json
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Self

from fatum.experiment.constants import JSON_INDENT, PARAMETERS_FILE, RUN_METADATA_FILE
from fatum.experiment.exceptions import NotFoundError, StateError, ValidationError
from fatum.experiment.protocols import StorageBackend
from fatum.experiment.storage import LocalStorage
from fatum.experiment.types import (
    Artifact,
    ArtifactKey,
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
    """Represents a single run within an experiment.

    A run tracks metrics, parameters, and artifacts for a specific execution
    of an experiment. Metrics are stored in JSONL format for efficient append
    operations and easy querying.

    Parameters
    ----------
    run_id : RunID
        Unique identifier for the run
    experiment : Experiment
        Parent experiment this run belongs to
    name : str, optional
        Human-readable name for the run
    tags : list[str] | None, optional
        Tags for categorizing the run

    Examples
    --------
    >>> with experiment.run("training") as run:
    ...     run.log_param("learning_rate", 0.001)
    ...     run.log_param("batch_size", 32)
    ...
    ...     for epoch in range(10):
    ...         run.log_metric("loss", loss_value, step=epoch)
    ...         run.log_metric("accuracy", acc_value, step=epoch)
    ...
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
        """Log a metric value for this run.

        Metrics are immediately saved as individual files through the storage backend
        for durability and consistency. Each metric is saved as a separate JSON file.

        Parameters
        ----------
        key : str
            Name of the metric (e.g., "loss", "accuracy", "f1_score")
        value : float
            Numeric value of the metric
        step : int, optional
            Training step or epoch number, defaults to 0

        Examples
        --------
        >>> run.log_metric("loss", 0.523)
        >>> run.log_metric("accuracy", 0.95, step=100)

        Creates files like:
        - metrics/step_000000_loss.json
        - metrics/step_000100_accuracy.json
        """
        if self._completed:
            raise StateError(self.metadata.status, "log metric")

        metric = Metric(key=MetricKey(key), value=value, step=step)
        self._metrics.append(metric)

        metric_filename = f"step_{step:06d}_{key}.json"
        storage_key = StorageKey(
            f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}/{StorageCategories.METRICS}/{metric_filename}"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metric.model_dump(mode="json"), tmp, indent=2)
            tmp_path = Path(tmp.name)

        try:
            self.experiment._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter for this run."""
        if self._completed:
            raise StateError(self.metadata.status, "log parameter")

        param = Parameter(key=ParamKey(key), value=value)
        self._parameters[param.key] = param

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_artifact(self, source: FilePath, artifact_name: str | None = None) -> ArtifactKey:
        """Save an artifact file for this run.

        Artifacts are saved using the experiment's storage backend, which can be
        local filesystem or cloud storage (S3, GCS, etc.).

        Parameters
        ----------
        source : FilePath
            Path to the file to save as an artifact
        artifact_name : str | None, optional
            Name for the artifact, defaults to source filename

        Returns
        -------
        ArtifactKey
            Key that can be used to retrieve the artifact later

        Examples
        --------
        >>> # Save model checkpoint
        >>> run.log_artifact("checkpoint.pt")
        >>>
        >>> # Save with custom name
        >>> run.log_artifact("/tmp/model.pkl", artifact_name="best_model.pkl")
        """
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
        """Save run metadata, metrics, and parameters."""
        run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"

        metadata_dict = self.metadata.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metadata_dict, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{run_base}/{StorageCategories.METADATA}/{RUN_METADATA_FILE}")
            self.experiment._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

        if self._parameters:
            params_dict = {k: p.model_dump(mode="json") for k, p in self._parameters.items()}
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
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
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.complete(RunStatus.FAILED)
        else:
            self.complete(RunStatus.COMPLETED)


class Experiment:
    """Main experiment tracking class with hybrid storage architecture.

    This class implements a pragmatic hybrid storage approach:
    - **Metrics/Metadata**: Always stored locally in JSONL/JSON format for fast append operations
    - **Artifacts**: Can use custom storage backends (S3, GCS, etc.) for scalability

    This design allows efficient metric logging (no network latency) while enabling
    cloud storage for large artifacts without double I/O overhead.

    Parameters
    ----------
    name : str
        Name of the experiment (used to generate unique ID)
    base_path : str | Path, optional
        Base directory for metrics and metadata (always local), defaults to "./experiments"
    storage : StorageBackend | None, optional
        Optional storage backend for artifacts (defaults to LocalStorage).
        Users can implement custom backends with just save() and load() methods.
    description : str, optional
        Human-readable description of the experiment
    tags : list[str] | None, optional
        Tags for categorizing and filtering experiments
    run_id_prefix : str, optional
        Prefix for generated run IDs, defaults to "run"

    Examples
    --------
    Basic usage with local storage:

    >>> exp = Experiment("model_training")
    >>> # Creates: ./experiments/model_training_abc123/
    >>>
    >>> with exp.run("epoch_1") as run:
    ...     run.log_metric("loss", 0.5)
    ...     run.log_metric("accuracy", 0.92)

    Using custom S3 storage for artifacts:

    >>> from my_storage import S3Storage
    >>> exp = Experiment(
    ...     "distributed_training",
    ...     base_path="./metrics",  # Metrics stay local
    ...     storage=S3Storage("ml-bucket"),  # Artifacts go to S3
    ...     tags=["production", "gpu"]
    ... )
    >>> exp.log_artifact("model.pkl")  # Uploads directly to S3

    Directory structure created:

    ```
    experiments/
    └── model_training_abc123/
        ├── metadata/
        │   └── experiment.json       # Experiment metadata
        ├── runs/
        │   └── run_001/
        │       ├── metadata/
        │       │   └── run.json      # Run metadata
        │       ├── metrics/
        │       │   └── metrics.jsonl # Append-only metrics log
        │       └── parameters/
        │           └── parameters.json
        └── artifacts/                # Large files (can be remote)
    ```

    Notes
    -----
    - Metrics use JSONL format for efficient append operations and easy querying
    - The hybrid approach eliminates double I/O for cloud deployments
    - Users can query metrics with standard tools: jq, DuckDB, pandas
    - Storage backends only need 2 methods: save() and load()
    """

    def __init__(
        self,
        name: str,
        base_path: str | Path = "./experiments",
        storage: StorageBackend | None = None,
        description: str = "",
        tags: list[str] | None = None,
        run_id_prefix: str = "run",
    ) -> None:
        """Initialize experiment with hybrid storage."""
        self.id = ExperimentID(f"{name}_{uuid.uuid4().hex[:8]}")
        self.metadata = ExperimentMetadata(
            id=self.id,
            name=name,
            description=description,
            tags=tags or [],
            git_info=get_git_info().model_dump() if get_git_info() else {},
        )

        self._base_path = Path(base_path).expanduser().resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

        self._storage = storage or LocalStorage(base_path)

        self._run_id_prefix = run_id_prefix

        self._runs: dict[RunID, Run] = {}
        self._artifacts: dict[ArtifactKey, Artifact] = {}
        self._completed = False

        self._save_metadata()

    def start_run(self, name: str = "", tags: list[str] | None = None) -> Run:
        """Start a new run."""
        run_id = RunID(f"{self._run_id_prefix}_{uuid.uuid4().hex[:8]}" if self._run_id_prefix else uuid.uuid4().hex[:8])
        run = Run(run_id, self, name, tags)
        self._runs[run_id] = run
        return run

    @contextmanager
    def run(self, name: str = "", tags: list[str] | None = None) -> Iterator[Run]:
        """Context manager for creating and managing runs.

        Automatically completes the run when the context exits, marking it as
        completed or failed based on whether an exception occurred.

        Parameters
        ----------
        name : str, optional
            Name for the run
        tags : list[str] | None, optional
            Tags for the run

        Yields
        ------
        Run
            The created run object

        Examples
        --------
        >>> with exp.run("training") as run:
        ...     run.log_param("lr", 0.001)
        ...     for epoch in range(10):
        ...         run.log_metric("loss", train(model))
        >>> # Run automatically completed

        >>> # Handles failures gracefully
        >>> with exp.run("evaluation") as run:
        ...     raise ValueError("Something went wrong")
        >>> # Run marked as failed
        """
        run = self.start_run(name, tags)
        try:
            yield run
        finally:
            run.complete()

    def log_artifact(self, source: FilePath, artifact_name: str | None = None) -> ArtifactKey:
        """Log an artifact at the experiment level.

        Experiment-level artifacts are shared across all runs. These are saved
        using the configured storage backend (local or cloud).

        Parameters
        ----------
        source : FilePath
            Path to the file to save as an artifact
        artifact_name : str | None, optional
            Name for the artifact, defaults to source filename

        Returns
        -------
        ArtifactKey
            Key that can be used to retrieve the artifact later

        Examples
        --------
        >>> # Save dataset
        >>> exp.log_artifact("data/train.csv", "training_data.csv")
        >>>
        >>> # Save large model file (goes to S3 if configured)
        >>> exp.log_artifact("model.bin")  # Direct S3 upload, no double I/O
        """
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
        """Load an artifact."""
        if artifact_key not in self._artifacts:
            raise NotFoundError("artifact", artifact_key)

        artifact = self._artifacts[artifact_key]
        return self._storage.load(artifact.storage_key)

    def save_dict(self, data: dict[str, Any], path: str, **tempfile_kwargs: Any) -> StorageKey:
        """Save a dictionary as JSON to the experiment directory.

        This method is useful for saving configuration, hyperparameters, or any
        structured data that needs to be preserved with the experiment.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to save (must be JSON-serializable)
        path : str
            Relative path within experiment directory (e.g., "config.json")
        **tempfile_kwargs : Any
            Additional arguments for tempfile creation (encoding, prefix, etc.)

        Returns
        -------
        StorageKey
            Storage key for the saved file

        Examples
        --------
        >>> # Save configuration
        >>> config = {"model": "transformer", "layers": 12}
        >>> exp.save_dict(config, "config.json")
        >>>
        >>> # Save to subdirectory
        >>> exp.save_dict(results, "evaluation/results.json")

        Creates:
        ```
        experiment_abc123/
        ├── config.json
        └── evaluation/
            └── results.json
        ```
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", **tempfile_kwargs) as tmp:
            json.dump(data, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{path}")
            # Use storage backend for user-created files
            self._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_text(self, text: str, path: str, **tempfile_kwargs: Any) -> StorageKey:
        """Save text content to a file in the experiment directory.

        Useful for saving logs, notes, or any text-based information.

        Parameters
        ----------
        text : str
            Text content to save
        path : str
            Relative path within experiment directory (e.g., "notes.txt")
        **tempfile_kwargs : Any
            Additional arguments for tempfile creation

        Returns
        -------
        StorageKey
            Storage key for the saved file

        Examples
        --------
        >>> # Save training log
        >>> exp.save_text("Training completed successfully", "training.log")
        >>>
        >>> # Save model description
        >>> exp.save_text(model_summary, "model_architecture.txt")
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, **tempfile_kwargs) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{path}")
            self._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_file(self, source: Path | str, relative_path: str) -> StorageKey:
        """Save a file to a specific path within the experiment directory.

        This method uses the storage backend for file operations, enabling
        direct uploads to cloud storage without double I/O.

        Parameters
        ----------
        source : Path | str
            Source file path
        relative_path : str
            Relative path within experiment directory

        Returns
        -------
        StorageKey
            Storage key for the saved file

        Examples
        --------
        >>> # Save plot
        >>> exp.save_file("plot.png", "visualizations/loss_curve.png")
        >>>
        >>> # Save checkpoint (goes directly to S3 if configured)
        >>> exp.save_file("checkpoint.pt", "checkpoints/epoch_10.pt")
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "File does not exist")

        storage_key = StorageKey(f"{self.id}/{relative_path}")
        self._storage.save(storage_key, source_path)
        return storage_key

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata_dict = self.metadata.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metadata_dict, tmp, indent=JSON_INDENT)
            tmp_path = Path(tmp.name)

        try:
            storage_key = StorageKey(f"{self.id}/{StorageCategories.METADATA}/experiment.json")
            self._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

    def complete(self) -> None:
        """Mark the experiment as completed and update metadata.

        This method updates the experiment status to 'completed' and records
        the completion timestamp. The metadata file is overwritten with the
        updated information.

        Notes
        -----
        - Safe to call multiple times (idempotent)
        - Metadata overwrite is intentional to update status
        - All other experiment data is preserved

        Examples
        --------
        >>> exp = Experiment("training")
        >>> # ... run experiments ...
        >>> exp.complete()
        >>> # metadata/experiment.json now shows status: "completed"
        """
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
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.metadata = self.metadata.model_copy(update={"status": ExperimentStatus.FAILED})
        self.complete()
