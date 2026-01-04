"""File-based vector storage adapter.

Implements VectorStoragePort using the local filesystem.
Vectors are stored as .npy files, index data as .bin files.

Usage:
    storage = FileVectorStorage("/path/to/data")
    storage.save_vector("v1", np.array([1.0, 2.0, 3.0]))
    vec = storage.load_vector("v1")

Directory structure:
    data_dir/
        vectors/
            v1.npy
            v2.npy
        indices/
            hnsw.bin
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class FileVectorStorage:
    """File-based implementation of VectorStoragePort.

    Stores vectors as numpy .npy files and index data as binary files.
    Suitable for single-node deployments with moderate data sizes.

    Attributes:
        data_dir: Root directory for all storage
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize file storage.

        Args:
            data_dir: Root directory for storage
        """
        self._data_dir = Path(data_dir)
        self._vectors_dir = self._data_dir / "vectors"
        self._indices_dir = self._data_dir / "indices"

        # Create directories if they don't exist
        self._vectors_dir.mkdir(parents=True, exist_ok=True)
        self._indices_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return self._data_dir

    def _vector_path(self, vector_id: str) -> Path:
        """Get file path for a vector."""
        # Sanitize ID to be filesystem-safe
        safe_id = vector_id.replace("/", "_").replace("\\", "_")
        return self._vectors_dir / f"{safe_id}.npy"

    def _index_path(self, index_name: str) -> Path:
        """Get file path for an index."""
        safe_name = index_name.replace("/", "_").replace("\\", "_")
        return self._indices_dir / f"{safe_name}.bin"

    def save_vector(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Persist a vector to disk.

        Args:
            vector_id: Unique identifier
            vector: Vector data to store
        """
        path = self._vector_path(vector_id)
        np.save(path, vector.astype(np.float32))

    def load_vector(self, vector_id: str) -> NDArray[np.float32] | None:
        """Load a vector from disk.

        Args:
            vector_id: Unique identifier

        Returns:
            Vector data, or None if not found
        """
        path = self._vector_path(vector_id)
        if not path.exists():
            return None
        return np.load(path).astype(np.float32)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from disk.

        Args:
            vector_id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        path = self._vector_path(vector_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_vectors(self) -> list[str]:
        """List all stored vector IDs.

        Returns:
            List of vector IDs
        """
        ids = []
        for path in self._vectors_dir.glob("*.npy"):
            # Convert filename back to ID
            ids.append(path.stem)
        return ids

    def save_index(self, index_data: bytes, index_name: str) -> None:
        """Persist serialized index data to disk.

        Args:
            index_data: Serialized index bytes
            index_name: Name/identifier for the index
        """
        path = self._index_path(index_name)
        path.write_bytes(index_data)

    def load_index(self, index_name: str) -> bytes | None:
        """Load serialized index data from disk.

        Args:
            index_name: Name/identifier for the index

        Returns:
            Serialized index bytes, or None if not found
        """
        path = self._index_path(index_name)
        if not path.exists():
            return None
        return path.read_bytes()

    def clear(self) -> None:
        """Delete all stored data.

        Warning: This permanently deletes all vectors and indices!
        """
        for path in self._vectors_dir.glob("*.npy"):
            path.unlink()
        for path in self._indices_dir.glob("*.bin"):
            path.unlink()

    def __len__(self) -> int:
        """Number of stored vectors."""
        return len(list(self._vectors_dir.glob("*.npy")))
