from __future__ import annotations

import atexit
import contextvars
from pathlib import Path
from typing import Any

from fatum.experiment.experiment import Experiment
from fatum.experiment.protocols import StorageBackend

# NOTE: Context variable for async safety (better than thread-local)
_active_experiment: contextvars.ContextVar[Experiment | None] = contextvars.ContextVar(
    "_active_experiment", default=None
)


def init(
    name: str,
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
        base_path=base_path,
        storage=storage,
        **kwargs,
    )

    if config:
        exp.save_dict(config, "config.json")

    _active_experiment.set(exp)

    atexit.register(finish)

    return exp


def finish() -> None:
    """Finish the active experiment and clean up."""
    exp = _active_experiment.get()
    if exp and not exp._completed:
        exp.complete()
    _active_experiment.set(None)


def log(data: dict[str, Any], step: int | None = None) -> None:
    """
    Log metrics/parameters.

    Parameters
    ----------
    data : dict
        Dictionary of metrics/parameters to log
    step : int, optional
        Step number for this log entry

    Examples
    --------
    >>> experiment.log({"loss": 0.23, "accuracy": 0.95})
    >>> experiment.log({"val_loss": 0.18}, step=100)
    """
    exp = _active_experiment.get()
    if not exp:
        return

    if step is not None:
        filename = f"metrics/step_{step:06d}.json"
    else:
        import time

        filename = f"metrics/log_{int(time.time() * 1000)}.json"

    exp.save_dict(data, filename)


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
    exp = _active_experiment.get()
    if exp:
        exp.save_dict(data, path)


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
    exp = _active_experiment.get()
    if exp:
        exp.save_text(text, path)


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
    exp = _active_experiment.get()
    if exp:
        exp.save_file(source, path)


def save_artifact(source: Path | str, artifact_name: str | None = None) -> None:
    """
    Save artifact to the active experiment.

    Parameters
    ----------
    source : Path | str
        Source file or directory path
    artifact_name : str, optional
        Name for the artifact (defaults to source filename)

    Examples
    --------
    >>> experiment.save_artifact("model.pkl")
    >>> experiment.save_artifact("/path/to/data", "training_data")
    """
    exp = _active_experiment.get()
    if exp:
        exp.log_artifact(source, artifact_name)


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
