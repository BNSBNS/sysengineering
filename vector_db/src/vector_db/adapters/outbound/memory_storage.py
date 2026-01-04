"""In-memory vector storage adapter.

A simple in-memory implementation of VectorStoragePort for testing
and development purposes. Data is not persisted across restarts.

Usage:
    storage = InMemoryVectorStorage()
    storage.save_vector("v1", np.array([1.0, 2.0, 3.0]))
    vec = storage.load_vector("v1")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class InMemoryVectorStorage:
    """In-memory implementation of VectorStoragePort.

    Stores vectors and index data in dictionaries. Useful for
    testing and development. Not suitable for production use.
    """

    def __init__(self) -> None:
        """Initialize empty storage."""
        self._vectors: dict[str, NDArray[np.float32]] = {}
        self._indices: dict[str, bytes] = {}

    def save_vector(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Store a vector in memory.

        Args:
            vector_id: Unique identifier
            vector: Vector data to store
        """
        self._vectors[vector_id] = vector.copy()

    def load_vector(self, vector_id: str) -> NDArray[np.float32] | None:
        """Load a vector from memory.

        Args:
            vector_id: Unique identifier

        Returns:
            Vector data, or None if not found
        """
        vec = self._vectors.get(vector_id)
        return vec.copy() if vec is not None else None

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from memory.

        Args:
            vector_id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        if vector_id in self._vectors:
            del self._vectors[vector_id]
            return True
        return False

    def list_vectors(self) -> list[str]:
        """List all stored vector IDs.

        Returns:
            List of vector IDs
        """
        return list(self._vectors.keys())

    def save_index(self, index_data: bytes, index_name: str) -> None:
        """Store serialized index data in memory.

        Args:
            index_data: Serialized index bytes
            index_name: Name/identifier for the index
        """
        self._indices[index_name] = index_data

    def load_index(self, index_name: str) -> bytes | None:
        """Load serialized index data from memory.

        Args:
            index_name: Name/identifier for the index

        Returns:
            Serialized index bytes, or None if not found
        """
        return self._indices.get(index_name)

    def clear(self) -> None:
        """Clear all stored data."""
        self._vectors.clear()
        self._indices.clear()

    def __len__(self) -> int:
        """Number of stored vectors."""
        return len(self._vectors)
