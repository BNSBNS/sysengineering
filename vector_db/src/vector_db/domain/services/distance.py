"""Distance functions for vector similarity computation.

This module provides CPU-optimized implementations of common distance metrics
used in approximate nearest neighbor search.

References:
    - FAISS: https://github.com/facebookresearch/faiss
    - ann-benchmarks: http://ann-benchmarks.com/

Performance Notes:
    - All functions use NumPy for vectorized operations
    - For GPU acceleration, use the CUDA adapter in adapters/outbound/cuda_compute.py
    - Batch operations are significantly faster than single-vector operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.value_objects.distance_metric import DistanceMetric

if TYPE_CHECKING:
    pass


# Type aliases for clarity
Vector = NDArray[np.float32]
Vectors = NDArray[np.float32]  # Shape: (n, dim)
Distances = NDArray[np.float32]  # Shape: (n,)


class DistanceFunction(Protocol):
    """Protocol for distance function signatures."""

    def __call__(self, a: Vector, b: Vector) -> float:
        """Compute distance between two vectors."""
        ...


class BatchDistanceFunction(Protocol):
    """Protocol for batch distance function signatures."""

    def __call__(self, query: Vector, candidates: Vectors) -> Distances:
        """Compute distances from query to all candidates."""
        ...


# =============================================================================
# Single-vector distance functions
# =============================================================================


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Formula: sqrt(sum((a[i] - b[i])^2))

    Args:
        a: First vector
        b: Second vector (same dimension as a)

    Returns:
        L2 distance (non-negative, 0 = identical)

    Example:
        >>> a = np.array([1.0, 0.0], dtype=np.float32)
        >>> b = np.array([0.0, 1.0], dtype=np.float32)
        >>> euclidean_distance(a, b)
        1.4142135...
    """
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def squared_euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute squared Euclidean distance (avoids sqrt for efficiency).

    Formula: sum((a[i] - b[i])^2)

    This is faster than euclidean_distance and preserves ordering,
    making it suitable for nearest neighbor comparisons.
    """
    diff = a - b
    return float(np.dot(diff, diff))


def cosine_distance(a: Vector, b: Vector) -> float:
    """Compute cosine distance between two vectors.

    Formula: 1 - (dot(a, b) / (||a|| * ||b||))

    Args:
        a: First vector
        b: Second vector (same dimension as a)

    Returns:
        Cosine distance in range [0, 2]
        - 0 = identical direction
        - 1 = orthogonal
        - 2 = opposite direction

    Note:
        For pre-normalized vectors, use inner_product instead.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0  # Undefined, return max distance

    similarity = dot_product / (norm_a * norm_b)
    # Clamp to [-1, 1] to handle floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two vectors.

    Formula: dot(a, b) / (||a|| * ||b||)

    Returns:
        Cosine similarity in range [-1, 1]
        - 1 = identical direction
        - 0 = orthogonal
        - -1 = opposite direction
    """
    return 1.0 - cosine_distance(a, b)


def inner_product(a: Vector, b: Vector) -> float:
    """Compute inner product (dot product) between two vectors.

    Formula: sum(a[i] * b[i])

    For normalized vectors, inner_product = cosine_similarity.

    Returns:
        Inner product (can be any real number)
        Higher values indicate more similarity for normalized vectors.
    """
    return float(np.dot(a, b))


def negative_inner_product(a: Vector, b: Vector) -> float:
    """Compute negative inner product (for use as distance).

    Since inner product is a similarity (higher = more similar),
    we negate it to use as a distance (lower = more similar).
    """
    return -inner_product(a, b)


# =============================================================================
# Batch distance functions (query vs many candidates)
# =============================================================================


def batch_euclidean_distance(query: Vector, candidates: Vectors) -> Distances:
    """Compute Euclidean distances from query to all candidates.

    Optimized using the identity:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2*dot(a, b)

    Args:
        query: Query vector of shape (dim,)
        candidates: Candidate vectors of shape (n, dim)

    Returns:
        Array of n distances
    """
    # ||q||^2 (scalar)
    query_sq = np.dot(query, query)

    # ||c||^2 for each candidate (n,)
    candidates_sq = np.sum(candidates * candidates, axis=1)

    # -2 * dot(q, c) for each candidate (n,)
    cross_term = -2.0 * candidates @ query

    # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*dot(q, c)
    sq_distances = query_sq + candidates_sq + cross_term

    # Clamp to avoid negative values from floating point errors
    sq_distances = np.maximum(sq_distances, 0.0)

    return np.sqrt(sq_distances).astype(np.float32)


def batch_squared_euclidean_distance(query: Vector, candidates: Vectors) -> Distances:
    """Compute squared Euclidean distances (faster, preserves ordering)."""
    query_sq = np.dot(query, query)
    candidates_sq = np.sum(candidates * candidates, axis=1)
    cross_term = -2.0 * candidates @ query
    sq_distances = query_sq + candidates_sq + cross_term
    return np.maximum(sq_distances, 0.0).astype(np.float32)


def batch_cosine_distance(query: Vector, candidates: Vectors) -> Distances:
    """Compute cosine distances from query to all candidates.

    Args:
        query: Query vector of shape (dim,)
        candidates: Candidate vectors of shape (n, dim)

    Returns:
        Array of n cosine distances in range [0, 2]
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.ones(len(candidates), dtype=np.float32)

    # Normalize query
    query_normalized = query / query_norm

    # Compute norms of all candidates
    candidate_norms = np.linalg.norm(candidates, axis=1)

    # Handle zero-norm candidates
    zero_mask = candidate_norms == 0
    candidate_norms[zero_mask] = 1.0  # Avoid division by zero

    # Normalize candidates
    candidates_normalized = candidates / candidate_norms[:, np.newaxis]

    # Compute dot products (cosine similarities)
    similarities = candidates_normalized @ query_normalized

    # Convert to distances
    distances = 1.0 - similarities

    # Set distance to 1.0 for zero-norm candidates
    distances[zero_mask] = 1.0

    return distances.astype(np.float32)


def batch_inner_product(query: Vector, candidates: Vectors) -> Distances:
    """Compute inner products from query to all candidates.

    Note: Returns negative inner products so that lower = more similar,
    consistent with distance semantics.
    """
    inner_products = candidates @ query
    return (-inner_products).astype(np.float32)


# =============================================================================
# Factory functions
# =============================================================================


def get_distance_function(metric: DistanceMetric) -> DistanceFunction:
    """Get the appropriate single-vector distance function for a metric.

    Args:
        metric: The distance metric to use

    Returns:
        A function that computes distance between two vectors
    """
    dispatch: dict[DistanceMetric, DistanceFunction] = {
        DistanceMetric.L2: euclidean_distance,
        DistanceMetric.COSINE: cosine_distance,
        DistanceMetric.INNER_PRODUCT: negative_inner_product,
    }
    return dispatch[metric]


def get_batch_distance_function(metric: DistanceMetric) -> BatchDistanceFunction:
    """Get the appropriate batch distance function for a metric.

    Args:
        metric: The distance metric to use

    Returns:
        A function that computes distances from query to all candidates
    """
    dispatch: dict[DistanceMetric, BatchDistanceFunction] = {
        DistanceMetric.L2: batch_euclidean_distance,
        DistanceMetric.COSINE: batch_cosine_distance,
        DistanceMetric.INNER_PRODUCT: batch_inner_product,
    }
    return dispatch[metric]


# =============================================================================
# Utility functions
# =============================================================================


def normalize_vectors(vectors: Vectors) -> Vectors:
    """Normalize vectors to unit length.

    Useful for converting L2/IP search to cosine search.

    Args:
        vectors: Vectors of shape (n, dim) or (dim,)

    Returns:
        Normalized vectors with ||v|| = 1
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0:
            return vectors
        return (vectors / norm).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return (vectors / norms).astype(np.float32)


def compute_ground_truth(
    queries: Vectors,
    database: Vectors,
    k: int,
    metric: DistanceMetric = DistanceMetric.L2,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Compute exact k-nearest neighbors (ground truth) using brute force.

    This is used for computing recall@K in benchmarks.

    Args:
        queries: Query vectors of shape (n_queries, dim)
        database: Database vectors of shape (n_database, dim)
        k: Number of neighbors to find
        metric: Distance metric to use

    Returns:
        Tuple of (indices, distances) where:
        - indices: shape (n_queries, k) - indices of k nearest neighbors
        - distances: shape (n_queries, k) - distances to k nearest neighbors
    """
    batch_distance = get_batch_distance_function(metric)
    n_queries = len(queries)

    indices = np.zeros((n_queries, k), dtype=np.int64)
    distances = np.zeros((n_queries, k), dtype=np.float32)

    for i, query in enumerate(queries):
        dists = batch_distance(query, database)

        # Get k smallest distances
        if k < len(dists):
            # argpartition is O(n) vs O(n log n) for full sort
            top_k_unsorted = np.argpartition(dists, k)[:k]
            # Sort just the top k
            top_k_order = np.argsort(dists[top_k_unsorted])
            top_k = top_k_unsorted[top_k_order]
        else:
            top_k = np.argsort(dists)[:k]

        indices[i] = top_k
        distances[i] = dists[top_k]

    return indices, distances


def compute_recall(
    predicted: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """Compute recall@K metric.

    Recall@K = |predicted ∩ ground_truth| / K

    Args:
        predicted: Predicted neighbor indices, shape (n_queries, k)
        ground_truth: True neighbor indices, shape (n_queries, k)
        k: Number of neighbors

    Returns:
        Average recall across all queries, in range [0, 1]
    """
    recalls = []
    for pred, true in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        true_set = set(true[:k])
        recall = len(pred_set & true_set) / k
        recalls.append(recall)
    return float(np.mean(recalls))
