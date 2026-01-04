"""Search parameters value objects for configuring similarity queries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HNSWSearchParams:
    """Search parameters for HNSW index.

    Reference: Malkov & Yashunin (2018), arXiv:1603.09320

    Attributes:
        k: Number of nearest neighbors to return
        ef: Beam width during search (higher = better recall, slower)
            - ef must be >= k
            - Paper recommends ef in range [50, 200] for good recall
    """

    k: int
    ef: int = 50

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.ef < self.k:
            raise ValueError(f"ef ({self.ef}) must be >= k ({self.k})")


@dataclass(frozen=True, slots=True)
class IVFSearchParams:
    """Search parameters for IVF index.

    Reference: FAISS paper, arXiv:1702.08734

    Attributes:
        k: Number of nearest neighbors to return
        nprobe: Number of clusters to search (higher = better recall, slower)
            - nprobe=1: ~60-70% recall
            - nprobe=5% of nlist: ~90% recall
            - nprobe=10% of nlist: ~95% recall
    """

    k: int
    nprobe: int = 10

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.nprobe < 1:
            raise ValueError(f"nprobe must be >= 1, got {self.nprobe}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of a similarity search.

    Attributes:
        vector_id: ID of the matching vector
        distance: Distance/similarity score to the query
        rank: Rank in results (0-indexed)
    """

    vector_id: str  # Using str for simplicity, can be VectorId
    distance: float
    rank: int = 0

    def __lt__(self, other: "SearchResult") -> bool:
        """Enable sorting by distance."""
        return self.distance < other.distance
