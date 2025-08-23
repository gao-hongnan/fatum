from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends.

    All storage implementations must provide these methods to work
    with the experiment tracking system.
    """

    def save(self, key: str, source: Path) -> None:
        """Save a file to storage.

        Parameters
        ----------
        key : str
            Storage key (path in storage)
        source : Path
            Local file path to save
        """
        ...

    def load(self, key: str) -> Path:
        """Load a file from storage.

        For remote storage, this downloads the file to a temporary location.
        For local storage, returns the actual file path.

        Parameters
        ----------
        key : str
            Storage key (path in storage)

        Returns
        -------
        Path
            Local path where file can be accessed
        """
        ...

    def get_uri(self, key: str) -> str:
        """Get URI/location of artifact without downloading.

        Parameters
        ----------
        key : str
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

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with given prefix.

        Parameters
        ----------
        prefix : str
            Key prefix to filter by

        Returns
        -------
        list[str]
            List of storage keys
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Parameters
        ----------
        key : str
            Storage key to check

        Returns
        -------
        bool
            True if key exists
        """
        ...

    async def asave(self, key: str, source: Path) -> None:
        """Async save to storage."""
        raise NotImplementedError("Async save not implemented")

    async def aload(self, key: str) -> Path:
        """Async load from storage."""
        raise NotImplementedError("Async load not implemented")
