"""Vector entity for storing embedding vectors with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.value_objects.vector_id import VectorId


@dataclass(slots=True)
class Vector:
    """A vector entity representing an embedding with associated metadata.

    Attributes:
        id: Unique identifier for the vector
        data: The embedding vector as a numpy array
        metadata: Optional key-value metadata for filtering
        created_at: Timestamp when the vector was created
    """

    id: VectorId
    data: NDArray[np.float32]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate and normalize the vector data."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float32)
        elif self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

        if self.data.ndim != 1:
            raise ValueError(f"Vector must be 1-dimensional, got {self.data.ndim}")

        if len(self.data) == 0:
            raise ValueError("Vector cannot be empty")

    @property
    def dim(self) -> int:
        """Return the dimensionality of the vector."""
        return len(self.data)

    @property
    def norm(self) -> float:
        """Return the L2 norm of the vector."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Vector":
        """Return a new vector with unit L2 norm."""
        norm = self.norm
        if norm == 0:
            return self
        return Vector(
            id=self.id,
            data=self.data / norm,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
        )

    def __len__(self) -> int:
        """Return the dimensionality."""
        return self.dim

    def __eq__(self, other: object) -> bool:
        """Vectors are equal if they have the same ID."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)


@dataclass(frozen=True, slots=True)
class VectorWithDistance:
    """A vector paired with its distance from a query.

    Used as a return type for search operations.
    """

    vector: Vector
    distance: float

    def __lt__(self, other: "VectorWithDistance") -> bool:
        """Enable sorting by distance (ascending)."""
        return self.distance < other.distance
