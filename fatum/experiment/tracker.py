from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from fatum.experiment.experiment import Experiment, Run
from fatum.experiment.protocols import Storage

# NOTE: Context variables for async safety (better than thread-local)
_active_experiment: contextvars.ContextVar[Experiment | None] = contextvars.ContextVar(
    "_active_experiment", default=None
)
_active_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("_active_run", default=None)


@contextmanager
def experiment(
    name: str,
    storage: Storage | None = None,
    id: str | None = None,
    **kwargs: Any,
) -> Iterator[Experiment]:
    """Context manager for creating and managing experiments.

    Parameters
    ----------
    name : str
        Experiment name (required)
    storage : Storage | None
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
    >>> with experiment("training") as exp:
    ...     with run() as r:
    ...         r.storage.log_metric("loss", 0.5)

    Multi-run experiment:
    >>> with experiment("hyperparam_search") as exp:
    ...     for lr in [0.001, 0.01]:
    ...         with run(f"lr_{lr}") as r:
    ...             r.storage.log_param("lr", lr)
    ...             r.storage.log_metric("loss", 0.5)
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
        _active_experiment.set(None)


@contextmanager
def run(name: str | None = None, tags: list[str] | None = None) -> Iterator[Run]:
    """Context manager for creating and managing runs within the active experiment.

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
    >>> with experiment("training") as exp:
    ...     with run("epoch_1") as r:
    ...         r.storage.log_metric("loss", 0.5)
    """
    exp = _active_experiment.get()
    if not exp:
        raise RuntimeError("No active experiment. Use experiment() context manager first.")

    # NOTE: Use the experiment's run() context manager
    with exp.run(name or "") as r:
        _active_run.set(r)
        try:
            yield r
        finally:
            _active_run.set(None)


def finish() -> None:
    """Clean up active run and experiment."""
    _active_run.set(None)
    _active_experiment.set(None)


def get_experiment() -> Experiment | None:
    """Get the active experiment (for advanced usage).

    Returns
    -------
    Experiment | None
        The active experiment or None if no experiment is active

    Examples
    --------
    >>> exp = get_experiment()
    >>> if exp:
    ...     print(f"Active experiment: {exp.id}")
    """
    return _active_experiment.get()


def get_run() -> Run | None:
    """Get the active run (for advanced usage).

    Returns
    -------
    Run | None
        The active run or None if no run is active

    Examples
    --------
    >>> r = get_run()
    >>> if r:
    ...     print(f"Active run: {r.id}")
    """
    return _active_run.get()


def is_active() -> bool:
    """Check if an experiment is active.

    Returns
    -------
    bool
        True if an experiment is currently active

    Examples
    --------
    >>> if is_active():
    ...     print("Experiment is active")
    """
    return _active_experiment.get() is not None
