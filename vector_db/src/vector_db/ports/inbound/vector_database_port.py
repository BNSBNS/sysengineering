"""Inbound port for vector database operations.

This protocol defines the interface that the application layer
exposes to inbound adapters (REST API, gRPC, CLI, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.value_objects import SearchResult


@runtime_checkable
class VectorDatabasePort(Protocol):
    """Protocol for vector database operations.

    This is the contract that the application layer implements
    and inbound adapters depend on.
    """

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        ...

    def __len__(self) -> int:
        """Number of vectors in the database."""
        ...

    def insert(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert a vector into the database.

        Args:
            vector_id: Unique identifier for the vector
            vector: The vector data, shape (dim,)
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector, shape (dim,)
            k: Number of neighbors to return

        Returns:
            List of SearchResult with vector_id and distance
        """
        ...

    def get(self, vector_id: str) -> NDArray[np.float32] | None:
        """Retrieve a vector by ID.

        Args:
            vector_id: The vector's unique identifier

        Returns:
            The vector data, or None if not found
        """
        ...

    def contains(self, vector_id: str) -> bool:
        """Check if a vector exists in the database.

        Args:
            vector_id: The vector's unique identifier

        Returns:
            True if the vector exists
        """
        ...

    def stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        ...
