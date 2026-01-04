"""Outbound port for vector storage.

This protocol defines the interface for persisting vectors
and index data to external storage systems.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VectorStoragePort(Protocol):
    """Protocol for vector storage operations.

    This is the contract that outbound adapters implement
    for persistent storage of vectors and index data.
    """

    def save_vector(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Persist a single vector.

        Args:
            vector_id: Unique identifier
            vector: Vector data to store
        """
        ...

    def load_vector(self, vector_id: str) -> NDArray[np.float32] | None:
        """Load a single vector.

        Args:
            vector_id: Unique identifier

        Returns:
            Vector data, or None if not found
        """
        ...

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from storage.

        Args:
            vector_id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    def list_vectors(self) -> list[str]:
        """List all stored vector IDs.

        Returns:
            List of vector IDs
        """
        ...

    def save_index(self, index_data: bytes, index_name: str) -> None:
        """Persist serialized index data.

        Args:
            index_data: Serialized index bytes
            index_name: Name/identifier for the index
        """
        ...

    def load_index(self, index_name: str) -> bytes | None:
        """Load serialized index data.

        Args:
            index_name: Name/identifier for the index

        Returns:
            Serialized index bytes, or None if not found
        """
        ...
