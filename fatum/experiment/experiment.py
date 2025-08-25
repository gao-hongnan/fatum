"""Experiment tracking with pure lifecycle management.

This module implements the simplest possible design where Run and Experiment
are pure lifecycle managers. Storage handles ALL operations.

Design Philosophy:
    - Run/Experiment ONLY manage lifecycle (context management)
    - Storage handles ALL I/O operations
    - Users access storage directly via run.storage
    - Storage can have ANY methods it wants
    - No artificial separation between experiment and run storage

Examples
--------
>>> # Storage can have ANY methods
>>> class MyStorage:
...     def initialize(self, run_id: str, experiment_id: str) -> None:
...         # Setup for run
...         pass
...     def finalize(self, status: str) -> None:
...         # Cleanup for run
...         pass
...     # Add ANY methods you want!
...     def log_metric(self, key: str, value: float) -> None: ...
...     def save_model(self, model: Any) -> None: ...
...     def custom_operation(self) -> None: ...
>>>
>>> storage = MyStorage()
>>> exp = Experiment("test", storage=storage)
>>>
>>> with exp.run("training") as run:
...     # Direct access to ALL storage methods!
...     run.storage.log_metric("loss", 0.5)
...     run.storage.save_model(model)
...     run.storage.custom_operation()
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Self

from fatum.experiment.protocols import Storage
from fatum.experiment.types import (
    ExperimentID,
    RunID,
)


class Run:
    """A run is just a lifecycle manager - nothing more.

    The Run class is a context manager that calls storage.initialize() on enter
    and storage.finalize() on exit. That's it. All actual operations are done
    through storage, which users access directly.

    Parameters
    ----------
    run_id : RunID
        Unique identifier for the run
    storage : Storage
        Storage backend that handles ALL operations
    experiment_id : ExperimentID
        Parent experiment identifier

    Attributes
    ----------
    storage : Storage
        Direct access to storage - can have ANY methods!

    Examples
    --------
    >>> with exp.run("training") as run:
    ...     # Storage can have ANY methods - you're not limited!
    ...     run.storage.log_metric("loss", 0.5)
    ...     run.storage.save_checkpoint(model, optimizer)
    ...     run.storage.whatever_method_you_defined()
    """

    def __init__(
        self,
        run_id: RunID,
        storage: Storage,
        experiment_id: ExperimentID,
    ) -> None:
        self.id = run_id
        self.experiment_id = experiment_id
        self.storage = storage  # Direct access to storage!
        self._completed = False

    def __enter__(self) -> Self:
        """Enter context - initialize storage."""
        self.storage.initialize(str(self.id), str(self.experiment_id))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context - finalize storage."""
        status = "failed" if exc_type else "completed"
        if not self._completed:
            self.storage.finalize(status)
            self._completed = True


class Experiment:
    """An experiment is just a lifecycle manager.

    The Experiment class creates Run instances and passes them the storage.
    That's its only job. All actual operations are handled by storage.

    Parameters
    ----------
    name : str
        Name of the experiment
    storage : Storage
        Storage backend for ALL operations
    id : str | None, optional
        Unique identifier (auto-generated if not provided)

    Examples
    --------
    >>> storage = MyCustomStorage()
    >>> exp = Experiment("training", storage=storage)
    >>>
    >>> with exp.run() as run:
    ...     run.storage.do_anything()  # Storage handles everything!
    """

    def __init__(
        self,
        name: str,
        storage: Storage,
        id: str | None = None,
    ) -> None:
        self.id = ExperimentID(id) if id else ExperimentID(f"{name}_{uuid.uuid4().hex[:8]}")
        self.name = name
        self._storage = storage
        self._completed = False

    @contextmanager
    def run(self, name: str = "") -> Iterator[Run]:
        """Create and manage a run.

        Parameters
        ----------
        name : str, optional
            Name for the run (auto-generated if not provided)

        Yields
        ------
        Run
            Run instance with direct storage access

        Examples
        --------
        >>> with exp.run("epoch_1") as run:
        ...     run.storage.log_metric("loss", 0.5)
        ...     # Storage can have ANY methods!
        """
        run_id = RunID(name) if name else RunID(f"run_{uuid.uuid4().hex[:8]}")
        run = Run(run_id, self._storage, self.id)

        try:
            yield run
        except Exception:  # noqa: TRY203
            raise

    def __enter__(self) -> Self:
        """Enter experiment context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit experiment context."""
        self._completed = True
