"""Distance metric enumeration for vector similarity computation."""

from __future__ import annotations

from enum import Enum, auto


class DistanceMetric(Enum):
    """Supported distance metrics for similarity search.

    Reference:
        Different use cases prefer different metrics:
        - L2 (Euclidean): General purpose, good for image features
        - COSINE: Text embeddings, normalized vectors
        - INNER_PRODUCT: When vectors are already normalized, fastest

    Note:
        For normalized vectors: IP = 1 - COSINE_DISTANCE
        FAISS and most vector DBs support all three.
    """

    L2 = auto()  # Euclidean distance: sqrt(sum((a[i] - b[i])^2))
    COSINE = auto()  # Cosine similarity: 1 - (dot(a,b) / (||a|| * ||b||))
    INNER_PRODUCT = auto()  # Inner product (dot product): sum(a[i] * b[i])

    @property
    def is_similarity(self) -> bool:
        """Return True if higher values mean more similar.

        L2: Lower is better (distance)
        COSINE: Higher is better (similarity, 1 = identical)
        INNER_PRODUCT: Higher is better (for normalized vectors)
        """
        return self in (DistanceMetric.COSINE, DistanceMetric.INNER_PRODUCT)

    @property
    def is_distance(self) -> bool:
        """Return True if lower values mean more similar."""
        return self == DistanceMetric.L2

    def __str__(self) -> str:
        return self.name.lower()
