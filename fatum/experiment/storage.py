from __future__ import annotations

import shutil
from pathlib import Path


class LocalStorage:
    """Local filesystem storage backend.

    Parameters
    ----------
    base_path : str | Path
        Base directory for storage (default: "./experiments")
    """

    def __init__(self, base_path: str | Path = "./experiments") -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, source: Path) -> None:
        """Save file or directory to local storage."""
        dest = self.base_path / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if source.is_file():
            shutil.copy2(source, dest)
        else:
            shutil.copytree(source, dest, dirs_exist_ok=True)

    def load(self, key: str) -> Path:
        """Load file or directory from local storage."""
        path = self.base_path / key
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return path

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with given prefix.

        Parameters
        ----------
        prefix : str
            Key prefix to filter by

        Returns
        -------
        list[str]
            List of storage keys (relative paths)
        """
        search_path = self.base_path / prefix if prefix else self.base_path

        if not search_path.exists():
            return []

        keys = []
        for item in search_path.rglob("*"):
            if item.is_file():
                relative = item.relative_to(self.base_path)
                keys.append(str(relative))

        return sorted(keys)

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
        path = self.base_path / key
        return path.exists()

    def get_uri(self, key: str) -> str:
        """Get file:// URI for local storage path.

        Parameters
        ----------
        key : str
            Storage key

        Returns
        -------
        str
            file:// URI for the local file
        """
        path = (self.base_path / key).resolve()
        return path.as_uri()

    async def asave(self, key: str, source: Path) -> None:
        """Async save to storage (not implemented for local storage)."""
        raise NotImplementedError("Async operations not implemented for LocalStorage")

    async def aload(self, key: str) -> Path:
        """Async load from storage (not implemented for local storage)."""
        raise NotImplementedError("Async operations not implemented for LocalStorage")
