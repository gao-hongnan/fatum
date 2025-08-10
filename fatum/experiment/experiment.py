from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Self

from fatum.experiment.exceptions import NotFoundError, StateError, ValidationError
from fatum.experiment.protocols import StorageBackend
from fatum.experiment.storage import LocalStorage
from fatum.experiment.types import (
    RUN_METADATA_FILE,
    Artifact,
    ArtifactKey,
    ExperimentID,
    ExperimentMetadata,
    ExperimentStatus,
    Metric,
    MetricKey,
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
            json.dump(metric.model_dump(mode="json"), tmp)
            tmp_path = Path(tmp.name)

        try:
            self.experiment._storage.save(storage_key, tmp_path)
        finally:
            tmp_path.unlink()

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def save_artifacts(self, source: Path | str, name: str | None = None) -> list[StorageKey]:
        """Save artifacts (file or directory) to this run.

        Handles both single files and directories (recursively).
        Files are saved using the configured storage backend (local or cloud).

        Parameters
        ----------
        source : Path | str
            File or directory to save
        name : str | None
            Name in storage (defaults to source name)

        Returns
        -------
        list[StorageKey]
            List of saved storage keys
        """
        if self._completed:
            raise StateError(self.metadata.status, "save artifacts")

        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "Source does not exist")

        saved_keys = []
        run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"

        if source_path.is_file():
            artifact_name = name or source_path.name
            storage_key = StorageKey(f"{run_base}/{StorageCategories.ARTIFACTS}/{artifact_name}")
            self.experiment._storage.save(storage_key, source_path)
            saved_keys.append(storage_key)
        else:
            name = name or source_path.name

            for root, _, files in os.walk(source_path):
                for file in files:
                    file_path = Path(root) / file
                    relative = file_path.relative_to(source_path)
                    storage_key = StorageKey(f"{run_base}/{StorageCategories.ARTIFACTS}/{name}/{relative.as_posix()}")
                    self.experiment._storage.save(storage_key, file_path)
                    saved_keys.append(storage_key)

        return saved_keys

    def save_dict(self, data: dict[str, Any], path: str, **json_kwargs: Any) -> StorageKey:
        """Save a dictionary as JSON to this run.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to save as JSON
        path : str
            Relative path within the run directory
        **json_kwargs
            Keyword arguments passed directly to json.dump()

        Returns
        -------
        StorageKey
            The storage key where the data was saved

        Examples
        --------
        >>> # Default compact JSON
        >>> run.save_dict({"model": "gpt-4"}, "config.json")

        >>> # Pretty printed with indentation
        >>> run.save_dict({"model": "gpt-4"}, "config.json", indent=2)

        >>> # Custom formatting
        >>> run.save_dict(results, "results.json", indent=4, sort_keys=True)
        """
        if self._completed:
            raise StateError(self.metadata.status, "save dict")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp, **json_kwargs)
            tmp_path = Path(tmp.name)

        try:
            run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"
            storage_key = StorageKey(f"{run_base}/{path}")
            self.experiment._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_text(self, text: str, path: str) -> StorageKey:
        """Save text content to a file in this run."""
        if self._completed:
            raise StateError(self.metadata.status, "save text")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"
            storage_key = StorageKey(f"{run_base}/{path}")
            self.experiment._storage.save(storage_key, tmp_path)
            return storage_key
        finally:
            tmp_path.unlink()

    def save_file(self, source: Path | str, relative_path: str) -> StorageKey:
        """Save a file to a specific path within this run."""
        if self._completed:
            raise StateError(self.metadata.status, "save file")

        source_path = Path(source)
        if not source_path.exists():
            raise ValidationError("source", str(source), "File does not exist")

        run_base = f"{self.experiment.id}/{StorageCategories.RUNS}/{self.id}"
        storage_key = StorageKey(f"{run_base}/{relative_path}")
        self.experiment._storage.save(storage_key, source_path)
        return storage_key

    async def asave_artifacts(self, source: Path, name: str) -> None:
        """Async version of save_artifacts."""
        raise NotImplementedError("Async save_artifacts not yet implemented")

    async def asave_dict(self, data: dict[str, Any], name: str) -> None:
        """Async version of save_dict."""
        raise NotImplementedError("Async save_dict not yet implemented")

    async def asave_file(self, source: Path, name: str) -> None:
        """Async version of save_file."""
        raise NotImplementedError("Async save_file not yet implemented")

    async def asave_text(self, text: str, name: str) -> None:
        """Async version of save_text."""
        raise NotImplementedError("Async save_text not yet implemented")

    def complete(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Complete the run and save all data."""
        if self._completed:
            return

        self.metadata = self.metadata.model_copy(update={"status": status, "ended_at": datetime.now()})
        self.save_dict(self.metadata.model_dump(mode="json"), f"{StorageCategories.METADATA}/{RUN_METADATA_FILE}")
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
        id: str | None = None,
        base_path: str | Path = "./experiments",
        storage: StorageBackend | None = None,
        description: str = "",
        tags: list[str] | None = None,
        run_id_prefix: str = "run",
    ) -> None:
        """Initialize experiment with hybrid storage."""
        self.id = ExperimentID(id) if id else ExperimentID(f"{name}_{uuid.uuid4().hex[:8]}")
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
        self._default_run: Run | None = None

        self._save_metadata()

    def start_run(self, name: str | None = None, tags: list[str] | None = None) -> Run:
        """Start a new run."""
        if name is None:
            import time

            name = f"run_{int(time.time())}"

        run_id = RunID(f"{self._run_id_prefix}_{uuid.uuid4().hex[:8]}" if self._run_id_prefix else uuid.uuid4().hex[:8])
        run = Run(run_id, self, name, tags)
        self._runs[run_id] = run
        return run

    def get_or_create_default_run(self) -> Run:
        """Get or create the default run for this experiment."""
        if self._default_run is None:
            self._default_run = self.start_run("default")
        return self._default_run

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

    def save_artifacts(self, source: Path | str, name: str | None = None) -> list[StorageKey]:
        """Proxy to default run's save_artifacts for convenience."""
        return self.get_or_create_default_run().save_artifacts(source, name)

    def load_artifact(self, artifact_key: ArtifactKey) -> Path:
        """Load an artifact."""
        if artifact_key not in self._artifacts:
            raise NotFoundError("artifact", artifact_key)

        artifact = self._artifacts[artifact_key]
        return self._storage.load(artifact.storage_key)

    def log_metrics(self, data: dict[str, float], step: int | None = None) -> None:
        """Proxy to default run's log_metrics for convenience."""
        self.get_or_create_default_run().log_metrics(data, step or 0)

    def save_dict(self, data: dict[str, Any], path: str, **json_kwargs: Any) -> StorageKey:
        """Proxy to default run's save_dict for convenience."""
        return self.get_or_create_default_run().save_dict(data, path, **json_kwargs)

    def save_text(self, text: str, path: str) -> StorageKey:
        """Proxy to default run's save_text for convenience."""
        return self.get_or_create_default_run().save_text(text, path)

    def save_file(self, source: Path | str, relative_path: str) -> StorageKey:
        """Proxy to default run's save_file for convenience."""
        return self.get_or_create_default_run().save_file(source, relative_path)

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata_dict = self.metadata.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metadata_dict, tmp)
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
