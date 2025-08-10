from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def save(self, key: str, source: Path) -> None: ...
    def load(self, key: str) -> Path: ...
