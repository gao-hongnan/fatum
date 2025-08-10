from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def save(self, key: str, source: Path) -> None: ...
    def load(self, key: str) -> Path: ...

    async def asave(self, key: str, source: Path) -> None:
        """Async save to storage."""
        raise NotImplementedError("Async save not implemented")

    async def aload(self, key: str) -> Path:
        """Async load from storage."""
        raise NotImplementedError("Async load not implemented")
