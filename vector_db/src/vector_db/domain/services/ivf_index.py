"""IVF (Inverted File Index) implementation for approximate nearest neighbor search.

This implements the IVF algorithm using K-means clustering to partition the vector space
into Voronoi cells, enabling fast approximate search by only examining nearby clusters.

Reference:
    Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs."
    IEEE Transactions on Big Data, 7(3), 535-547.
    arXiv: https://arxiv.org/abs/1702.08734
    Code: https://github.com/facebookresearch/faiss

Algorithm Overview:
    Training:
        1. Run K-means on a sample of vectors to find nlist centroids
        2. Each centroid defines a Voronoi cell (cluster)

    Indexing:
        - Assign each vector to its nearest centroid

    Searching:
        1. Find the nprobe closest centroids to the query
        2. Exhaustively search vectors in those clusters only
        3. This reduces search from O(n) to O(n * nprobe / nlist)

Complexity:
    - Training: O(n * nlist * iterations) for K-means
    - Insert: O(nlist) to find nearest centroid
    - Search: O(nlist + n * nprobe / nlist)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.services.distance import (
    batch_euclidean_distance,
    get_batch_distance_function,
)
from vector_db.domain.value_objects.distance_metric import DistanceMetric
from vector_db.domain.value_objects.search_params import SearchResult

if TYPE_CHECKING:
    pass


@dataclass
class IVFParams:
    """IVF index parameters.

    Reference: FAISS paper recommendations
    """

    nlist: int = 1000  # Number of clusters (Voronoi cells)
    nprobe: int = 10  # Clusters to search at query time
    training_iterations: int = 20  # K-means iterations
    spherical: bool = False  # Normalize to unit sphere
    verbose: bool = False  # Print training progress

    @classmethod
    def for_dataset_size(cls, n: int, target_recall: float = 0.95) -> "IVFParams":
        """Compute optimal parameters for dataset size.

        Rule of thumb from FAISS:
            nlist ≈ sqrt(n) for balanced recall/speed

        Args:
            n: Number of vectors
            target_recall: Target recall level

        Returns:
            Optimized IVFParams
        """
        nlist = max(1, int(math.sqrt(n)))

        # nprobe ratio based on target recall
        if target_recall >= 0.95:
            nprobe_ratio = 0.10  # 10% of clusters
        elif target_recall >= 0.90:
            nprobe_ratio = 0.05  # 5% of clusters
        else:
            nprobe_ratio = 0.01  # 1% of clusters

        nprobe = max(1, int(nlist * nprobe_ratio))
        return cls(nlist=nlist, nprobe=nprobe)


@dataclass
class InvertedList:
    """A single inverted list (cluster) in the IVF index.

    Stores all vectors assigned to this cluster along with their IDs.
    """

    centroid: NDArray[np.float32]
    vector_ids: list[str] = field(default_factory=list)
    vectors: list[NDArray[np.float32]] = field(default_factory=list)

    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Add a vector to this cluster."""
        self.vector_ids.append(vector_id)
        self.vectors.append(vector)

    def __len__(self) -> int:
        return len(self.vector_ids)

    def get_vectors_array(self) -> NDArray[np.float32] | None:
        """Get all vectors as a 2D array for batch processing."""
        if not self.vectors:
            return None
        return np.array(self.vectors, dtype=np.float32)


class IVFIndex:
    """IVF index for approximate nearest neighbor search.

    The index must be trained before vectors can be added.

    Example:
        >>> index = IVFIndex(dim=128, metric=DistanceMetric.L2)
        >>> # Train on a sample of data
        >>> index.train(training_vectors)
        >>> # Add vectors
        >>> for vid, vec in vectors:
        ...     index.add(vid, vec)
        >>> # Search
        >>> results = index.search(query, k=10)
    """

    def __init__(
        self,
        dim: int,
        metric: DistanceMetric = DistanceMetric.L2,
        params: IVFParams | None = None,
        seed: int | None = None,
    ):
        """Initialize the IVF index.

        Args:
            dim: Dimensionality of vectors
            metric: Distance metric
            params: IVF parameters
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.metric = metric
        self.params = params or IVFParams()
        self._rng = np.random.default_rng(seed)

        # Distance function
        self._batch_distance = get_batch_distance_function(metric)

        # Centroids and inverted lists (populated after training)
        self._centroids: NDArray[np.float32] | None = None
        self._inverted_lists: list[InvertedList] = []
        self._is_trained = False
        self._total_vectors = 0

    def __len__(self) -> int:
        """Return total number of vectors."""
        return self._total_vectors

    @property
    def is_trained(self) -> bool:
        """Return whether the index has been trained."""
        return self._is_trained

    @property
    def centroids(self) -> NDArray[np.float32] | None:
        """Return the cluster centroids."""
        return self._centroids

    def train(self, vectors: NDArray[np.float32]) -> None:
        """Train the index using K-means clustering.

        Args:
            vectors: Training vectors of shape (n, dim)
                    Should have at least nlist vectors for good clustering

        Raises:
            ValueError: If already trained or invalid input
        """
        if self._is_trained:
            raise ValueError("Index is already trained. Create a new index to retrain.")

        if len(vectors) == 0:
            raise ValueError("Cannot train on empty vectors")

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")

        vectors = vectors.astype(np.float32)

        # Normalize if spherical
        if self.params.spherical:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        # Adjust nlist if we have fewer training vectors
        nlist = min(self.params.nlist, len(vectors))

        if self.params.verbose:
            print(f"Training IVF with {nlist} clusters on {len(vectors)} vectors...")

        # Run K-means
        centroids = self._kmeans(vectors, nlist)

        # Initialize inverted lists
        self._centroids = centroids
        self._inverted_lists = [
            InvertedList(centroid=centroids[i]) for i in range(nlist)
        ]
        self._is_trained = True

        if self.params.verbose:
            print("Training complete.")

    def _kmeans(
        self,
        vectors: NDArray[np.float32],
        k: int,
    ) -> NDArray[np.float32]:
        """Run K-means clustering.

        Uses Lloyd's algorithm with random initialization.

        Args:
            vectors: Input vectors of shape (n, dim)
            k: Number of clusters

        Returns:
            Centroids of shape (k, dim)
        """
        n = len(vectors)

        # Initialize centroids using random sampling (k-means++)
        # For simplicity, we use random sampling here
        indices = self._rng.choice(n, size=min(k, n), replace=False)
        centroids = vectors[indices].copy()

        # Pad with zeros if we have fewer vectors than clusters
        if len(centroids) < k:
            padding = np.zeros((k - len(centroids), self.dim), dtype=np.float32)
            centroids = np.vstack([centroids, padding])

        for iteration in range(self.params.training_iterations):
            # Assign each vector to nearest centroid
            assignments = self._assign_to_centroids(vectors, centroids)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int32)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += vectors[i]
                counts[cluster_id] += 1

            # Avoid division by zero for empty clusters
            for c in range(k):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    # Reinitialize empty cluster with random vector
                    new_centroids[c] = vectors[self._rng.integers(n)]

            # Check for convergence
            delta = np.sum(np.abs(new_centroids - centroids))
            centroids = new_centroids

            if self.params.verbose:
                print(f"  Iteration {iteration + 1}: delta = {delta:.6f}")

            if delta < 1e-6:
                break

        return centroids

    def _assign_to_centroids(
        self,
        vectors: NDArray[np.float32],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.int32]:
        """Assign each vector to its nearest centroid.

        Args:
            vectors: Vectors of shape (n, dim)
            centroids: Centroids of shape (k, dim)

        Returns:
            Array of cluster assignments of shape (n,)
        """
        # Compute all pairwise distances efficiently
        # ||v - c||^2 = ||v||^2 + ||c||^2 - 2*v·c

        vectors_sq = np.sum(vectors * vectors, axis=1)  # (n,)
        centroids_sq = np.sum(centroids * centroids, axis=1)  # (k,)
        cross_term = vectors @ centroids.T  # (n, k)

        # distances[i, j] = ||vectors[i] - centroids[j]||^2
        distances = vectors_sq[:, np.newaxis] + centroids_sq - 2 * cross_term

        return np.argmin(distances, axis=1).astype(np.int32)

    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Add a vector to the index.

        Args:
            vector_id: Unique identifier
            vector: The vector to add

        Raises:
            ValueError: If index is not trained
        """
        if not self._is_trained:
            raise ValueError("Index must be trained before adding vectors")

        if len(vector) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(vector)}")

        vector = vector.astype(np.float32)

        if self.params.spherical:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        # Find nearest centroid
        cluster_id = self._find_nearest_centroid(vector)

        # Add to inverted list
        self._inverted_lists[cluster_id].add(vector_id, vector)
        self._total_vectors += 1

    def add_batch(
        self,
        vector_ids: list[str],
        vectors: NDArray[np.float32],
    ) -> None:
        """Add multiple vectors to the index.

        More efficient than adding one at a time.

        Args:
            vector_ids: List of unique identifiers
            vectors: Vectors of shape (n, dim)
        """
        if not self._is_trained:
            raise ValueError("Index must be trained before adding vectors")

        if len(vector_ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")

        vectors = vectors.astype(np.float32)

        if self.params.spherical:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        # Assign all vectors to centroids
        assignments = self._assign_to_centroids(vectors, self._centroids)

        # Add to inverted lists
        for i, (vid, vec) in enumerate(zip(vector_ids, vectors)):
            cluster_id = assignments[i]
            self._inverted_lists[cluster_id].add(vid, vec)

        self._total_vectors += len(vectors)

    def _find_nearest_centroid(self, vector: NDArray[np.float32]) -> int:
        """Find the nearest centroid to a vector."""
        distances = batch_euclidean_distance(vector, self._centroids)
        return int(np.argmin(distances))

    def _find_nprobe_centroids(
        self,
        query: NDArray[np.float32],
        nprobe: int,
    ) -> list[int]:
        """Find the nprobe nearest centroids to a query."""
        distances = self._batch_distance(query, self._centroids)

        if nprobe >= len(self._centroids):
            return list(range(len(self._centroids)))

        # Get nprobe smallest distances
        indices = np.argpartition(distances, nprobe)[:nprobe]
        # Sort by distance
        sorted_indices = indices[np.argsort(distances[indices])]
        return sorted_indices.tolist()

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of neighbors to return
            nprobe: Number of clusters to search (defaults to params.nprobe)

        Returns:
            List of SearchResult sorted by distance
        """
        if not self._is_trained:
            raise ValueError("Index must be trained before searching")

        if len(query) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(query)}")

        query = query.astype(np.float32)

        if self.params.spherical:
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        nprobe = nprobe or self.params.nprobe
        nprobe = min(nprobe, len(self._inverted_lists))

        # Find nprobe nearest clusters
        probe_clusters = self._find_nprobe_centroids(query, nprobe)

        # Collect all candidates from probed clusters
        all_ids: list[str] = []
        all_vectors: list[NDArray[np.float32]] = []

        for cluster_id in probe_clusters:
            inv_list = self._inverted_lists[cluster_id]
            all_ids.extend(inv_list.vector_ids)
            all_vectors.extend(inv_list.vectors)

        if not all_ids:
            return []

        # Compute distances to all candidates
        candidates = np.array(all_vectors, dtype=np.float32)
        distances = self._batch_distance(query, candidates)

        # Get top k
        if k >= len(distances):
            top_k_indices = np.argsort(distances)
        else:
            top_k_indices = np.argpartition(distances, k)[:k]
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        # Build results
        results = []
        for rank, idx in enumerate(top_k_indices[:k]):
            results.append(
                SearchResult(
                    vector_id=all_ids[idx],
                    distance=float(distances[idx]),
                    rank=rank,
                )
            )

        return results

    def batch_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
    ) -> list[list[SearchResult]]:
        """Search for k nearest neighbors for multiple queries.

        Args:
            queries: Query vectors of shape (n_queries, dim)
            k: Number of neighbors per query
            nprobe: Number of clusters to search

        Returns:
            List of search results for each query
        """
        return [self.search(q, k, nprobe) for q in queries]

    def get_stats(self) -> dict:
        """Get statistics about the index."""
        if not self._is_trained:
            return {
                "is_trained": False,
                "num_vectors": 0,
                "dim": self.dim,
            }

        cluster_sizes = [len(inv) for inv in self._inverted_lists]
        non_empty = sum(1 for s in cluster_sizes if s > 0)

        return {
            "is_trained": True,
            "num_vectors": self._total_vectors,
            "dim": self.dim,
            "nlist": len(self._inverted_lists),
            "non_empty_clusters": non_empty,
            "avg_cluster_size": self._total_vectors / non_empty if non_empty else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "params": {
                "nlist": self.params.nlist,
                "nprobe": self.params.nprobe,
                "training_iterations": self.params.training_iterations,
                "spherical": self.params.spherical,
            },
        }
