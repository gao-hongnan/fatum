from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from fatum.experiment.types import StorageKey


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends.

    All storage implementations must provide these methods to work
    with the experiment tracking system.
    """

    def save(self, key: StorageKey, source: Path) -> None:
        """Save a file to storage.

        Parameters
        ----------
        key : StorageKey
            Storage key (path in storage)
        source : Path
            Local file path to save
        """
        ...

    def load(self, key: StorageKey) -> Path:
        """Load a file from storage.

        For remote storage, this downloads the file to a temporary location.
        For local storage, returns the actual file path.

        Parameters
        ----------
        key : StorageKey
            Storage key (path in storage)

        Returns
        -------
        Path
            Local path where file can be accessed
        """
        ...

    def get_uri(self, key: StorageKey) -> str:
        """Get URI/location of artifact without downloading.

        Parameters
        ----------
        key : StorageKey
            Storage key

        Returns
        -------
        str
            URI for the artifact:
            - Local: file:///absolute/path/to/artifact
            - S3: s3://bucket/key
            - GCS: gs://bucket/key
            - HTTP: https://storage.example.com/key
        """
        ...

    def list_keys(self, prefix: StorageKey | None = None) -> list[StorageKey]:
        """List all keys with given prefix.

        Parameters
        ----------
        prefix : StorageKey
            Key prefix to filter by

        Returns
        -------
        list[StorageKey]
            List of storage keys
        """
        ...

    def exists(self, key: StorageKey) -> bool:
        """Check if a key exists in storage.

        Parameters
        ----------
        key : StorageKey
            Storage key to check

        Returns
        -------
        bool
            True if key exists
        """
        ...

    async def asave(self, key: StorageKey, source: Path) -> None:
        """Async save to storage."""
        raise NotImplementedError("Async save not implemented")

    async def aload(self, key: StorageKey) -> Path:
        """Async load from storage."""
        raise NotImplementedError("Async load not implemented")
