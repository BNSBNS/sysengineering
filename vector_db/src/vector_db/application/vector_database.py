"""VectorDatabase application service.

This is the main entry point for the vector database, coordinating
domain services (HNSW, IVF, PQ) and providing a unified interface.

Usage:
    from vector_db.application import VectorDatabase, IndexType

    # Create database with HNSW index
    db = VectorDatabase(dim=128, index_type=IndexType.HNSW)

    # Insert vectors
    db.insert("vec1", np.random.randn(128).astype(np.float32))

    # Search
    results = db.search(query_vector, k=10)

Reference:
    - design.md for architectural decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.services import (
    HNSWIndex,
    HNSWParams,
    IVFIndex,
    IVFParams,
    ProductQuantizer,
    PQParams,
)
from vector_db.domain.value_objects import DistanceMetric, SearchResult

if TYPE_CHECKING:
    pass


class IndexType(Enum):
    """Supported index types."""

    HNSW = "hnsw"
    IVF = "ivf"
    FLAT = "flat"  # Brute-force, exact search


@dataclass
class VectorDatabaseConfig:
    """Configuration for VectorDatabase."""

    dim: int
    index_type: IndexType = IndexType.HNSW
    metric: DistanceMetric = DistanceMetric.L2

    # HNSW params
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50

    # IVF params
    ivf_nlist: int = 100
    ivf_nprobe: int = 10

    # PQ params (for IVF+PQ)
    use_pq: bool = False
    pq_m: int = 8
    pq_ks: int = 256


class VectorDatabase:
    """Main vector database application service.

    Provides a unified interface for vector storage and similarity search,
    supporting multiple index types (HNSW, IVF, flat).

    Attributes:
        dim: Vector dimensionality
        index_type: Type of index being used
        metric: Distance metric for similarity
    """

    def __init__(
        self,
        dim: int,
        index_type: IndexType = IndexType.HNSW,
        metric: DistanceMetric = DistanceMetric.L2,
        config: VectorDatabaseConfig | None = None,
    ) -> None:
        """Initialize the vector database.

        Args:
            dim: Vector dimensionality
            index_type: Type of index to use
            metric: Distance metric
            config: Optional full configuration
        """
        if config:
            self._config = config
        else:
            self._config = VectorDatabaseConfig(
                dim=dim,
                index_type=index_type,
                metric=metric,
            )

        self._dim = self._config.dim
        self._index_type = self._config.index_type
        self._metric = self._config.metric
        self._is_trained = False

        # Initialize the appropriate index
        self._index = self._create_index()

        # For flat index, store vectors directly
        self._flat_vectors: dict[str, NDArray[np.float32]] = {}

    def _create_index(self) -> HNSWIndex | IVFIndex | None:
        """Create the appropriate index based on config."""
        if self._index_type == IndexType.HNSW:
            return HNSWIndex(
                dim=self._dim,
                metric=self._metric,
                params=HNSWParams(
                    M=self._config.hnsw_m,
                    ef_construction=self._config.hnsw_ef_construction,
                    ef_search=self._config.hnsw_ef_search,
                ),
            )
        elif self._index_type == IndexType.IVF:
            return IVFIndex(
                dim=self._dim,
                metric=self._metric,
                params=IVFParams(
                    nlist=self._config.ivf_nlist,
                    nprobe=self._config.ivf_nprobe,
                ),
            )
        else:
            # Flat index - no special data structure
            return None

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    def index_type(self) -> IndexType:
        """Index type being used."""
        return self._index_type

    @property
    def metric(self) -> DistanceMetric:
        """Distance metric."""
        return self._metric

    @property
    def is_trained(self) -> bool:
        """Whether the index is trained (relevant for IVF)."""
        return self._is_trained

    def __len__(self) -> int:
        """Number of vectors in the database."""
        if self._index_type == IndexType.FLAT:
            return len(self._flat_vectors)
        return len(self._index)

    def train(self, vectors: NDArray[np.float32]) -> None:
        """Train the index on sample vectors.

        Required for IVF before insertion. No-op for HNSW and flat.

        Args:
            vectors: Training vectors, shape (n, dim)
        """
        if self._index_type == IndexType.IVF:
            self._index.train(vectors)
        self._is_trained = True

    def insert(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert a vector into the database.

        Args:
            vector_id: Unique identifier for the vector
            vector: The vector data, shape (dim,)

        Raises:
            ValueError: If vector dimension doesn't match
            RuntimeError: If IVF index is not trained
        """
        if vector.shape[0] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {vector.shape[0]}")

        if self._index_type == IndexType.IVF and not self._is_trained:
            raise RuntimeError("IVF index must be trained before insertion")

        if self._index_type == IndexType.FLAT:
            self._flat_vectors[vector_id] = vector.astype(np.float32)
        elif self._index_type == IndexType.HNSW:
            self._index.insert(vector_id, vector)
        else:
            self._index.add(vector_id, vector)

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef_search: int | None = None,
        nprobe: int | None = None,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector, shape (dim,)
            k: Number of neighbors to return
            ef_search: HNSW search beam width (optional override)
            nprobe: IVF clusters to search (optional override)

        Returns:
            List of SearchResult with vector_id and distance
        """
        if query.shape[0] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {query.shape[0]}")

        if self._index_type == IndexType.FLAT:
            return self._flat_search(query, k)
        elif self._index_type == IndexType.HNSW:
            return self._index.search(query, k, ef=ef_search)
        else:
            return self._index.search(query, k, nprobe=nprobe)

    def _flat_search(
        self,
        query: NDArray[np.float32],
        k: int,
    ) -> list[SearchResult]:
        """Brute-force exact search for flat index."""
        from vector_db.domain.services.distance import get_distance_function

        if not self._flat_vectors:
            return []

        dist_fn = get_distance_function(self._metric)

        results = []
        for vid, vec in self._flat_vectors.items():
            dist = dist_fn(query, vec)
            results.append(SearchResult(vector_id=vid, distance=float(dist)))

        results.sort(key=lambda r: r.distance)
        return results[:k]

    def get(self, vector_id: str) -> NDArray[np.float32] | None:
        """Retrieve a vector by ID.

        Args:
            vector_id: The vector's unique identifier

        Returns:
            The vector data, or None if not found
        """
        if self._index_type == IndexType.FLAT:
            return self._flat_vectors.get(vector_id)
        return self._index.get_vector(vector_id)

    def contains(self, vector_id: str) -> bool:
        """Check if a vector exists in the database.

        Args:
            vector_id: The vector's unique identifier

        Returns:
            True if the vector exists
        """
        if self._index_type == IndexType.FLAT:
            return vector_id in self._flat_vectors
        return self._index.contains(vector_id)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector from the database.

        Note: Not all index types support deletion efficiently.

        Args:
            vector_id: The vector to delete

        Returns:
            True if deleted, False if not found
        """
        if self._index_type == IndexType.FLAT:
            if vector_id in self._flat_vectors:
                del self._flat_vectors[vector_id]
                return True
            return False
        # HNSW and IVF don't support efficient deletion
        raise NotImplementedError(
            f"Deletion not supported for {self._index_type.value} index"
        )

    def stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats = {
            "dim": self._dim,
            "index_type": self._index_type.value,
            "metric": str(self._metric),
            "num_vectors": len(self),
            "is_trained": self._is_trained,
        }

        if self._index_type == IndexType.HNSW and self._index:
            index_stats = self._index.get_stats()
            stats.update({
                "hnsw_levels": index_stats.get("num_levels", 0),
                "hnsw_m": self._config.hnsw_m,
                "hnsw_ef_construction": self._config.hnsw_ef_construction,
            })
        elif self._index_type == IndexType.IVF and self._index:
            stats.update({
                "ivf_nlist": self._config.ivf_nlist,
                "ivf_nprobe": self._config.ivf_nprobe,
            })

        return stats
