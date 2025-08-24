from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from fatum.experiment.experiment import Experiment, Run
from fatum.experiment.protocols import StorageBackend

# NOTE: Context variables for async safety (better than thread-local)
_active_experiment: contextvars.ContextVar[Experiment | None] = contextvars.ContextVar(
    "_active_experiment", default=None
)
_active_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("_active_run", default=None)


@contextmanager
def experiment(
    name: str,
    storage: StorageBackend | None = None,
    id: str | None = None,
    **kwargs: Any,
) -> Iterator[Experiment]:
    """
    Context manager for creating and managing experiments.

    Parameters
    ----------
    name : str
        Experiment name (required)
    storage : StorageBackend | None
        Storage backend (defaults to LocalStorage("./experiments") if not provided)
    id : str | None
        Optional experiment ID (auto-generated if not provided)
    **kwargs : Any
        Additional arguments passed to Experiment constructor

    Yields
    ------
    Experiment
        The experiment instance

    Examples
    --------
    Single-run experiment:
    >>> with experiment.experiment("training") as exp:
    ...     with experiment.run() as r:
    ...         r.save_dict({"lr": 0.01}, "config.json")
    ...         r.log_metrics({"loss": 0.5})

    Multi-run experiment:
    >>> with experiment.experiment("hyperparam_search") as exp:
    ...     for lr in [0.001, 0.01]:
    ...         with experiment.run(f"lr_{lr}") as r:
    ...             r.save_dict({"lr": lr}, "config.json")
    ...             r.log_metrics({"loss": 0.5})
    """
    finish()

    if storage is None:
        from fatum.experiment.storage import LocalStorage

        storage = LocalStorage("./experiments")

    exp = Experiment(
        name=name,
        storage=storage,
        id=id,
        **kwargs,
    )

    _active_experiment.set(exp)

    try:
        yield exp
    finally:
        exp.complete()
        _active_experiment.set(None)


@contextmanager
def run(name: str | None = None, tags: list[str] | None = None) -> Iterator[Run]:
    """
    Context manager for creating and managing runs within the active experiment.

    Parameters
    ----------
    name : str | None
        Optional name for the run
    tags : list[str] | None
        Optional tags for the run

    Yields
    ------
    Run
        The run instance

    Examples
    --------
    >>> with experiment.experiment("training") as exp:
    ...     with experiment.run("epoch_1") as r:
    ...         r.log_metrics({"loss": 0.5})
    """
    exp = _active_experiment.get()
    if not exp:
        raise RuntimeError("No active experiment. Use experiment() context manager first.")

    r = exp.start_run(name, tags)
    _active_run.set(r)

    try:
        yield r
    except Exception:
        from fatum.experiment.types import RunStatus

        r.complete(RunStatus.FAILED)
        raise
    else:
        r.complete()
    finally:
        _active_run.set(None)


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
    >>> with experiment.experiment("hyperparameter_search") as exp:
    ...     with experiment.run("lr_0.01") as r:
    ...         r.log_metrics({"loss": 0.5})
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
