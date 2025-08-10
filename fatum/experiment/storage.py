from __future__ import annotations

import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, base_path: str | Path = "./experiments") -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, source: Path) -> None:
        dest = self.base_path / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if source.is_file():
            shutil.copy2(source, dest)
        else:
            shutil.copytree(source, dest, dirs_exist_ok=True)

    def load(self, key: str) -> Path:
        path = self.base_path / key
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return path
