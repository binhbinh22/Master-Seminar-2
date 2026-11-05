"""Interchangeable byte stores for staged datasets, independent of validation logic."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol


class ByteStore(Protocol):
    def read_bytes(self, path: str | Path) -> bytes: ...
    def write_bytes(self, path: str | Path, value: bytes) -> None: ...


class FileStore:
    """Persist bytes using the same paths accepted by pathlib."""
    def read_bytes(self, path: str | Path) -> bytes:
        return Path(path).read_bytes()

    def write_bytes(self, path: str | Path, value: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)


class MemoryStore:
    """Own an isolated snapshot; missing keys behave like missing files."""
    def __init__(self, values: Mapping[str | Path, bytes] | None = None):
        self._values = {self._key(k): bytes(v) for k, v in (values or {}).items()}

    @staticmethod
    def _key(path: str | Path) -> str:
        return Path(path).as_posix()

    def read_bytes(self, path: str | Path) -> bytes:
        try:
            return self._values[self._key(path)]
        except KeyError:
            raise FileNotFoundError(str(path)) from None

    def write_bytes(self, path: str | Path, value: bytes) -> None:
        self._values[self._key(path)] = bytes(value)

    @classmethod
    def snapshot(cls, directory: str | Path) -> MemoryStore:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        return cls({p: p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()})


def dataset_store(backend: str, directory: str | Path) -> ByteStore:
    """Memory mode imports one snapshot, then all loader reads use that snapshot."""
    if backend == 'file':
        return FileStore()
    if backend == 'memory':
        return MemoryStore.snapshot(directory)
    raise ValueError('storage backend must be file or memory')
