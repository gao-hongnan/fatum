from __future__ import annotations

import atexit
import contextvars
from pathlib import Path
from typing import Any

from fatum.experiment.experiment import Experiment, Run
from fatum.experiment.protocols import StorageBackend

# NOTE: Context variables for async safety (better than thread-local)
_active_experiment: contextvars.ContextVar[Experiment | None] = contextvars.ContextVar(
    "_active_experiment", default=None
)
_active_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("_active_run", default=None)


def init(
    name: str,
    id: str | None = None,
    base_path: str | Path = "./experiments",
    storage: StorageBackend | None = None,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Experiment:
    """
    Initialize a global experiment.

    Parameters
    ----------
    name : str
        Experiment name (required)
    base_path : str | Path
        Base directory for metrics and metadata (default: "./experiments")
    storage : StorageBackend | None
        Optional storage backend for artifacts (defaults to LocalStorage)
    config : dict, optional
        Configuration dictionary to save
    **kwargs : Any
        Additional arguments passed to Experiment constructor

    Returns
    -------
    Experiment
        The initialized experiment

    Examples
    --------
    Simple local storage:
    >>> from fatum import experiment
    >>> experiment.init("my_experiment")
    >>> experiment.log({"accuracy": 0.95})

    With custom storage backend:
    >>> from my_storage import S3Storage
    >>> experiment.init(
    ...     name="alignment",
    ...     storage=S3Storage("ml-bucket"),
    ...     config={"lr": 0.01, "batch_size": 32},
    ...     tags=["production", "v2"]
    ... )
    """
    finish()

    exp = Experiment(
        name=name,
        id=id,
        base_path=base_path,
        storage=storage,
        **kwargs,
    )

    _active_experiment.set(exp)

    # Auto-create default run (like W&B)
    run = exp.start_run()
    _active_run.set(run)

    if config:
        run.save_dict(config, "config.json")

    atexit.register(finish)

    return exp


def start_run(name: str | None = None, tags: list[str] | None = None) -> Run:
    """
    Start a new run within the active experiment.

    Parameters
    ----------
    name : str | None
        Optional name for the run
    tags : list[str] | None
        Optional tags for the run

    Returns
    -------
    Run
        The newly created run

    Examples
    --------
    >>> experiment.init("hyperparameter_search")
    >>> experiment.start_run("lr_0.01")
    >>> experiment.log({"loss": 0.5})
    >>> experiment.finish()
    """
    exp = _active_experiment.get()
    if not exp:
        raise RuntimeError("No active experiment. Call init() first.")

    # NOTE: End current run if there is one
    if (current_run := _active_run.get()) and not current_run._completed:
        current_run.complete()

    # NOTE: Start new run
    run = exp.start_run(name, tags)
    _active_run.set(run)
    return run


def finish() -> None:
    """Finish the active run and experiment, then clean up."""
    run = _active_run.get()
    if run and not run._completed:
        run.complete()
    _active_run.set(None)

    exp = _active_experiment.get()
    if exp and not exp._completed:
        exp.complete()
    _active_experiment.set(None)


def log(data: dict[str, Any], step: int | None = None) -> None:
    """
    Log metrics to the active run.

    Parameters
    ----------
    data : dict
        Dictionary of metrics to log
    step : int, optional
        Step number for this log entry

    Examples
    --------
    >>> experiment.log({"loss": 0.23, "accuracy": 0.95})
    >>> experiment.log({"val_loss": 0.18}, step=100)
    """
    run = _active_run.get()
    if run:
        run.log_metrics(data, step or 0)


def save_dict(data: dict[str, Any], path: str) -> None:
    """
    Save dictionary to the active experiment.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to save as JSON
    path : str
        Relative path within the experiment directory

    Examples
    --------
    >>> experiment.save_dict({"model": "gpt-4"}, "configs/model.json")
    """
    run = _active_run.get()
    if run:
        run.save_dict(data, path)


def save_text(text: str, path: str) -> None:
    """
    Save text to the active experiment.

    Parameters
    ----------
    text : str
        Text content to save
    path : str
        Relative path within the experiment directory

    Examples
    --------
    >>> experiment.save_text("Training complete", "logs/status.txt")
    """
    run = _active_run.get()
    if run:
        run.save_text(text, path)


def save_file(source: Path | str, path: str) -> None:
    """
    Save file to the active experiment.

    Parameters
    ----------
    source : Path | str
        Source file path
    path : str
        Relative path within the experiment directory

    Examples
    --------
    >>> experiment.save_file("model.pkl", "artifacts/model.pkl")
    """
    run = _active_run.get()
    if run:
        run.save_file(source, path)


def save_artifacts(source: Path | str, name: str | None = None) -> list[Any] | None:
    """
    Save artifacts (file or directory) to the active experiment.

    Parameters
    ----------
    source : Path | str
        Source file or directory path
    name : str, optional
        Name for the artifact (defaults to source name)

    Examples
    --------
    >>> experiment.save_artifacts("model.pkl")
    >>> experiment.save_artifacts("/path/to/data", "training_data")
    >>> experiment.save_artifacts("checkpoints/")  # Saves directory recursively
    """
    run = _active_run.get()
    if run:
        return run.save_artifacts(source, name)
    return None


def get_experiment() -> Experiment | None:
    """
    Get the active experiment (for advanced usage).

    Returns
    -------
    Experiment | None
        The active experiment or None if no experiment is active

    Examples
    --------
    >>> exp = experiment.get_experiment()
    >>> if exp:
    ...     print(f"Active experiment: {exp.id}")
    """
    return _active_experiment.get()


def get_run() -> Run | None:
    """
    Get the active run (for advanced usage).

    Returns
    -------
    Run | None
        The active run or None if no run is active

    Examples
    --------
    >>> run = experiment.get_run()
    >>> if run:
    ...     print(f"Active run: {run.id}")
    """
    return _active_run.get()


def is_active() -> bool:
    """
    Check if an experiment is active.

    Returns
    -------
    bool
        True if an experiment is currently active

    Examples
    --------
    >>> if experiment.is_active():
    ...     experiment.log({"status": "running"})
    """
    return _active_experiment.get() is not None
