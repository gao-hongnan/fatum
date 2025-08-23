"""Unit tests for LocalStorage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from fatum.experiment.storage import LocalStorage
from fatum.experiment.types import StorageKey


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Create LocalStorage instance with temp directory."""
    return LocalStorage(base_path=tmp_path / "storage")


class TestLocalStorage:
    """Test LocalStorage implementation."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test storage initialization creates base directory."""
        base_path = tmp_path / "test_storage"
        assert not base_path.exists()

        storage = LocalStorage(base_path=base_path)

        assert base_path.exists()
        assert base_path.is_dir()
        assert storage.base_path == base_path.resolve()

    def test_save_file(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test saving a file."""
        source = tmp_path / "source.txt"
        source.write_text("test content")

        key = StorageKey("path/to/file.txt")
        storage.save(key, source)

        saved_path = storage.base_path / "path" / "to" / "file.txt"
        assert saved_path.exists()
        assert saved_path.read_text() == "test content"

    def test_save_directory(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test saving a directory."""
        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file2.txt").write_text("content2")

        key = StorageKey("saved_dir")
        storage.save(key, source_dir)

        saved_dir = storage.base_path / "saved_dir"
        assert saved_dir.exists()
        assert (saved_dir / "file1.txt").read_text() == "content1"
        assert (saved_dir / "subdir" / "file2.txt").read_text() == "content2"

    def test_load_existing_file(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test loading an existing file."""
        source = tmp_path / "source.txt"
        source.write_text("load test")
        key = StorageKey("test/load.txt")
        storage.save(key, source)

        loaded_path = storage.load(key)

        assert loaded_path.exists()
        assert loaded_path.read_text() == "load test"
        assert loaded_path == storage.base_path / "test" / "load.txt"

    def test_load_missing_file(self, storage: LocalStorage) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Key not found"):
            storage.load(StorageKey("non/existent.txt"))

    def test_exists(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test checking if keys exist."""
        assert not storage.exists(StorageKey("missing.txt"))

        source = tmp_path / "test.txt"
        source.write_text("exists test")
        key = StorageKey("existing.txt")
        storage.save(key, source)

        assert storage.exists(key)

    def test_list_keys_empty(self, storage: LocalStorage) -> None:
        """Test listing keys in empty storage."""
        keys = storage.list_keys()
        assert keys == []

    def test_list_keys_with_files(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test listing keys with files."""
        for i in range(3):
            source = tmp_path / f"file{i}.txt"
            source.write_text(f"content{i}")
            storage.save(StorageKey(f"data/file{i}.txt"), source)

        keys = storage.list_keys()
        assert len(keys) == 3
        assert StorageKey("data/file0.txt") in keys
        assert StorageKey("data/file1.txt") in keys
        assert StorageKey("data/file2.txt") in keys

    def test_list_keys_with_prefix(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test listing keys with prefix filter."""
        source = tmp_path / "test.txt"
        source.write_text("test")

        storage.save(StorageKey("models/model1.pkl"), source)
        storage.save(StorageKey("models/model2.pkl"), source)
        storage.save(StorageKey("data/file1.txt"), source)
        storage.save(StorageKey("data/file2.txt"), source)

        model_keys = storage.list_keys(StorageKey("models"))
        assert len(model_keys) == 2
        assert all("models" in str(k) for k in model_keys)

        data_keys = storage.list_keys(StorageKey("data"))
        assert len(data_keys) == 2
        assert all("data" in str(k) for k in data_keys)

    def test_list_keys_nested(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test listing keys with nested directory structure."""
        source = tmp_path / "test.txt"
        source.write_text("test")

        storage.save(StorageKey("a/b/c/file1.txt"), source)
        storage.save(StorageKey("a/b/file2.txt"), source)
        storage.save(StorageKey("a/file3.txt"), source)

        keys = storage.list_keys(StorageKey("a"))
        assert len(keys) == 3

        keys = storage.list_keys(StorageKey("a/b"))
        assert len(keys) == 2
        assert StorageKey("a/b/c/file1.txt") in keys
        assert StorageKey("a/b/file2.txt") in keys

    def test_get_uri(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test getting file URI."""
        source = tmp_path / "test.txt"
        source.write_text("uri test")
        key = StorageKey("test/file.txt")
        storage.save(key, source)

        uri = storage.get_uri(key)

        assert uri.startswith("file:///")
        assert "test/file.txt" in uri
        expected_path = (storage.base_path / "test" / "file.txt").resolve()
        assert uri == expected_path.as_uri()

    def test_path_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that paths are properly expanded."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))

        storage = LocalStorage(base_path="~/experiments")

        assert storage.base_path.is_absolute()
        assert str(home) in str(storage.base_path)

    def test_concurrent_saves(self, storage: LocalStorage, tmp_path: Path) -> None:
        """Test that parent directories are created safely."""
        source = tmp_path / "test.txt"
        source.write_text("test")

        storage.save(StorageKey("dir1/file.txt"), source)
        storage.save(StorageKey("dir2/file.txt"), source)
        storage.save(StorageKey("dir1/subdir/file.txt"), source)

        assert storage.exists(StorageKey("dir1/file.txt"))
        assert storage.exists(StorageKey("dir2/file.txt"))
        assert storage.exists(StorageKey("dir1/subdir/file.txt"))
