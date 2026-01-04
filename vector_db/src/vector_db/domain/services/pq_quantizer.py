"""Product Quantization (PQ) implementation for vector compression and fast distance computation.

This implements the Product Quantization algorithm for compressing high-dimensional vectors
and enabling fast approximate distance computation using lookup tables.

Reference:
    Jégou, H., Douze, M., & Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search."
    IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 33(1), 117-128.
    DOI: 10.1109/TPAMI.2010.57

Algorithm Overview:
    Product Quantization exploits the fact that L2 distance decomposes:
        ||x - y||^2 = Σ ||x_m - y_m||^2

    Training:
        1. Split D-dimensional vectors into M subvectors of D/M dimensions each
        2. For each subspace, run K-means to create Ks centroids (codebook)
        3. Store M codebooks, each with Ks centroids

    Encoding:
        1. For each vector, find the nearest centroid in each subspace
        2. Store only the M centroid indices (M bytes for Ks=256)
        3. Compression ratio: D * 4 bytes -> M bytes = 4D/M times

    Distance Computation (ADC - Asymmetric Distance Computation):
        1. Precompute distance table: for each subspace m and centroid k,
           compute distance from query subvector to centroid[m][k]
        2. For each database vector (stored as M codes):
           distance ≈ Σ table[m][code[m]]
        3. This uses M table lookups instead of D multiplications!

Complexity:
    - Training: O(n * Ks * iterations * D/M) per subspace
    - Encoding: O(M * Ks * D/M) = O(D * Ks)
    - Distance (ADC): O(M) per comparison (just table lookups!)
    - Memory: M bytes per vector (vs D * 4 bytes uncompressed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.value_objects.search_params import SearchResult

if TYPE_CHECKING:
    pass


@dataclass
class PQParams:
    """Product Quantization parameters.

    Reference: Jégou et al. (2011)
    """

    M: int = 8  # Number of subquantizers (subspaces)
    Ks: int = 256  # Centroids per subquantizer (256 = 8-bit codes)
    training_iterations: int = 20  # K-means iterations per subspace
    verbose: bool = False

    def __post_init__(self) -> None:
        # Validate Ks is power of 2 for efficient encoding
        if self.Ks & (self.Ks - 1) != 0:
            raise ValueError(f"Ks must be a power of 2, got {self.Ks}")
        if self.Ks > 65536:
            raise ValueError(f"Ks cannot exceed 65536, got {self.Ks}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")

    @property
    def bits_per_code(self) -> int:
        """Number of bits per code (log2(Ks))."""
        return int(np.log2(self.Ks))

    @property
    def bytes_per_vector(self) -> int:
        """Number of bytes per encoded vector."""
        if self.Ks <= 256:
            return self.M  # 1 byte per subquantizer
        else:
            return self.M * 2  # 2 bytes per subquantizer


@dataclass
class PQCodebook:
    """Codebook storing centroids for all subspaces.

    Attributes:
        centroids: Shape (M, Ks, dsub) - centroids for each subspace
        M: Number of subspaces
        Ks: Centroids per subspace
        dsub: Dimension per subspace
    """

    centroids: NDArray[np.float32]  # Shape: (M, Ks, dsub)
    M: int
    Ks: int
    dsub: int

    def encode_vector(self, vector: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Encode a single vector to M codes.

        Args:
            vector: Vector of shape (dim,) where dim = M * dsub

        Returns:
            Array of M codes (uint8 for Ks <= 256)
        """
        codes = np.zeros(self.M, dtype=np.uint8)
        for m in range(self.M):
            subvector = vector[m * self.dsub : (m + 1) * self.dsub]
            # Find nearest centroid in this subspace
            distances = np.sum((self.centroids[m] - subvector) ** 2, axis=1)
            codes[m] = np.argmin(distances)
        return codes

    def encode_batch(self, vectors: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Encode multiple vectors.

        Args:
            vectors: Vectors of shape (n, dim)

        Returns:
            Codes of shape (n, M)
        """
        n = len(vectors)
        codes = np.zeros((n, self.M), dtype=np.uint8)

        for m in range(self.M):
            # Extract subvectors for this subspace
            subvectors = vectors[:, m * self.dsub : (m + 1) * self.dsub]  # (n, dsub)
            centroids_m = self.centroids[m]  # (Ks, dsub)

            # Compute distances efficiently
            # ||s - c||^2 = ||s||^2 + ||c||^2 - 2*s·c
            subvectors_sq = np.sum(subvectors ** 2, axis=1, keepdims=True)  # (n, 1)
            centroids_sq = np.sum(centroids_m ** 2, axis=1)  # (Ks,)
            cross_term = subvectors @ centroids_m.T  # (n, Ks)

            distances = subvectors_sq + centroids_sq - 2 * cross_term  # (n, Ks)
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def decode_vector(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Decode codes back to approximate vector.

        Args:
            codes: Array of M codes

        Returns:
            Reconstructed vector of shape (M * dsub,)
        """
        vector = np.zeros(self.M * self.dsub, dtype=np.float32)
        for m in range(self.M):
            vector[m * self.dsub : (m + 1) * self.dsub] = self.centroids[m, codes[m]]
        return vector

    def compute_distance_tables(self, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Precompute distance lookup tables for a query.

        This is the key to fast ADC (Asymmetric Distance Computation).
        We precompute distances from each query subvector to all centroids.

        Args:
            query: Query vector of shape (dim,)

        Returns:
            Distance table of shape (M, Ks) where table[m][k] = ||q_m - c_m_k||^2
        """
        tables = np.zeros((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            subquery = query[m * self.dsub : (m + 1) * self.dsub]
            # Squared distances to all centroids in this subspace
            tables[m] = np.sum((self.centroids[m] - subquery) ** 2, axis=1)
        return tables


class ProductQuantizer:
    """Product Quantization encoder/decoder.

    Example:
        >>> pq = ProductQuantizer(dim=128, params=PQParams(M=8, Ks=256))
        >>> pq.train(training_vectors)
        >>> codes = pq.encode(vectors)  # (n, 8) uint8
        >>> # 128 * 4 = 512 bytes -> 8 bytes = 64x compression!
    """

    def __init__(
        self,
        dim: int,
        params: PQParams | None = None,
        seed: int | None = None,
    ):
        """Initialize the Product Quantizer.

        Args:
            dim: Vector dimensionality (must be divisible by M)
            params: PQ parameters
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.params = params or PQParams()
        self._rng = np.random.default_rng(seed)

        # Validate dimension
        if dim % self.params.M != 0:
            raise ValueError(
                f"Dimension {dim} must be divisible by M={self.params.M}"
            )

        self.dsub = dim // self.params.M
        self._codebook: PQCodebook | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Return whether the quantizer is trained."""
        return self._is_trained

    @property
    def codebook(self) -> PQCodebook | None:
        """Return the trained codebook."""
        return self._codebook

    @property
    def compression_ratio(self) -> float:
        """Return the compression ratio."""
        original_bytes = self.dim * 4  # float32
        compressed_bytes = self.params.bytes_per_vector
        return original_bytes / compressed_bytes

    def train(self, vectors: NDArray[np.float32]) -> None:
        """Train the product quantizer using K-means on each subspace.

        Args:
            vectors: Training vectors of shape (n, dim)
                    Should have at least Ks vectors per subspace

        Raises:
            ValueError: If already trained or invalid input
        """
        if self._is_trained:
            raise ValueError("Quantizer already trained. Create a new instance to retrain.")

        if len(vectors) == 0:
            raise ValueError("Cannot train on empty vectors")

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")

        vectors = vectors.astype(np.float32)

        M = self.params.M
        Ks = self.params.Ks
        dsub = self.dsub

        # Initialize centroids array
        centroids = np.zeros((M, Ks, dsub), dtype=np.float32)

        if self.params.verbose:
            print(f"Training PQ: M={M}, Ks={Ks}, dsub={dsub}")
            print(f"Compression ratio: {self.compression_ratio:.1f}x")

        # Train K-means for each subspace independently
        for m in range(M):
            if self.params.verbose:
                print(f"  Training subspace {m + 1}/{M}...")

            # Extract subvectors
            subvectors = vectors[:, m * dsub : (m + 1) * dsub]

            # Run K-means
            centroids[m] = self._kmeans(subvectors, Ks)

        self._codebook = PQCodebook(
            centroids=centroids,
            M=M,
            Ks=Ks,
            dsub=dsub,
        )
        self._is_trained = True

        if self.params.verbose:
            print("Training complete.")

    def _kmeans(
        self,
        vectors: NDArray[np.float32],
        k: int,
    ) -> NDArray[np.float32]:
        """Run K-means on a set of vectors.

        Args:
            vectors: Vectors of shape (n, d)
            k: Number of clusters

        Returns:
            Centroids of shape (k, d)
        """
        n, d = vectors.shape
        k = min(k, n)

        # Initialize with random sampling
        indices = self._rng.choice(n, size=k, replace=False)
        centroids = vectors[indices].copy()

        for _ in range(self.params.training_iterations):
            # Assign to nearest centroid
            # distances[i, j] = ||vectors[i] - centroids[j]||^2
            vectors_sq = np.sum(vectors ** 2, axis=1, keepdims=True)
            centroids_sq = np.sum(centroids ** 2, axis=1)
            cross_term = vectors @ centroids.T
            distances = vectors_sq + centroids_sq - 2 * cross_term

            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int32)

            for i, c in enumerate(assignments):
                new_centroids[c] += vectors[i]
                counts[c] += 1

            for c in range(k):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    new_centroids[c] = vectors[self._rng.integers(n)]

            centroids = new_centroids

        return centroids

    def encode(self, vectors: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Encode vectors to PQ codes.

        Args:
            vectors: Vectors of shape (n, dim) or (dim,)

        Returns:
            Codes of shape (n, M) or (M,)
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before encoding")

        single = vectors.ndim == 1
        if single:
            vectors = vectors.reshape(1, -1)

        vectors = vectors.astype(np.float32)
        codes = self._codebook.encode_batch(vectors)

        return codes[0] if single else codes

    def decode(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Decode PQ codes back to approximate vectors.

        Args:
            codes: Codes of shape (n, M) or (M,)

        Returns:
            Reconstructed vectors of shape (n, dim) or (dim,)
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before decoding")

        single = codes.ndim == 1
        if single:
            codes = codes.reshape(1, -1)

        n = len(codes)
        vectors = np.zeros((n, self.dim), dtype=np.float32)

        for i in range(n):
            vectors[i] = self._codebook.decode_vector(codes[i])

        return vectors[0] if single else vectors

    def compute_distance_tables(self, query: NDArray[np.float32]) -> NDArray[np.float32]:
        """Precompute distance tables for a query (ADC).

        Args:
            query: Query vector of shape (dim,)

        Returns:
            Distance tables of shape (M, Ks)
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained first")

        return self._codebook.compute_distance_tables(query.astype(np.float32))

    def asymmetric_distance(
        self,
        query: NDArray[np.float32],
        codes: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Compute approximate distances from query to encoded vectors.

        This is the key speedup of PQ: instead of computing full dot products,
        we use precomputed lookup tables. Cost: M table lookups per comparison
        instead of D multiplications.

        Args:
            query: Query vector of shape (dim,)
            codes: Encoded vectors of shape (n, M)

        Returns:
            Approximate squared distances of shape (n,)
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained first")

        # Precompute distance tables
        tables = self.compute_distance_tables(query)  # (M, Ks)

        n = len(codes)
        M = self.params.M

        # Use lookup tables to compute distances
        # For each vector, sum up table[m][code[m]] for all m
        distances = np.zeros(n, dtype=np.float32)

        # Vectorized implementation using advanced indexing
        for m in range(M):
            distances += tables[m, codes[:, m]]

        return distances


class PQIndex:
    """Complete index using Product Quantization.

    Stores vectors in compressed form and uses ADC for fast search.
    Can be combined with IVF for even better performance (IVF-PQ).

    Example:
        >>> index = PQIndex(dim=128)
        >>> index.train(training_vectors)
        >>> for vid, vec in vectors:
        ...     index.add(vid, vec)
        >>> results = index.search(query, k=10)
    """

    def __init__(
        self,
        dim: int,
        params: PQParams | None = None,
        seed: int | None = None,
    ):
        """Initialize PQ index.

        Args:
            dim: Vector dimensionality
            params: PQ parameters
            seed: Random seed
        """
        self.dim = dim
        self.params = params or PQParams()
        self._pq = ProductQuantizer(dim, params, seed)

        # Storage
        self._vector_ids: list[str] = []
        self._codes: list[NDArray[np.uint8]] = []
        self._original_vectors: list[NDArray[np.float32]] = []  # For re-ranking

    def __len__(self) -> int:
        return len(self._vector_ids)

    @property
    def is_trained(self) -> bool:
        return self._pq.is_trained

    def train(self, vectors: NDArray[np.float32]) -> None:
        """Train the quantizer."""
        self._pq.train(vectors)

    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Add a vector to the index."""
        if not self._pq.is_trained:
            raise ValueError("Index must be trained before adding")

        vector = vector.astype(np.float32)
        code = self._pq.encode(vector)

        self._vector_ids.append(vector_id)
        self._codes.append(code)
        self._original_vectors.append(vector)

    def add_batch(
        self,
        vector_ids: list[str],
        vectors: NDArray[np.float32],
    ) -> None:
        """Add multiple vectors."""
        if not self._pq.is_trained:
            raise ValueError("Index must be trained before adding")

        vectors = vectors.astype(np.float32)
        codes = self._pq.encode(vectors)

        for i, vid in enumerate(vector_ids):
            self._vector_ids.append(vid)
            self._codes.append(codes[i])
            self._original_vectors.append(vectors[i])

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        rerank: bool = True,
        rerank_factor: int = 10,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of results
            rerank: Whether to re-rank using original vectors
            rerank_factor: Fetch k * rerank_factor candidates for re-ranking

        Returns:
            List of SearchResult sorted by distance
        """
        if not self._pq.is_trained:
            raise ValueError("Index must be trained before searching")

        if len(self._codes) == 0:
            return []

        query = query.astype(np.float32)

        # Stack all codes
        all_codes = np.array(self._codes, dtype=np.uint8)

        # Compute approximate distances using ADC
        approx_distances = self._pq.asymmetric_distance(query, all_codes)

        # Get candidates
        n_candidates = min(k * rerank_factor if rerank else k, len(approx_distances))

        if n_candidates >= len(approx_distances):
            candidate_indices = np.argsort(approx_distances)
        else:
            candidate_indices = np.argpartition(approx_distances, n_candidates)[:n_candidates]
            candidate_indices = candidate_indices[np.argsort(approx_distances[candidate_indices])]

        if rerank and n_candidates > k:
            # Re-rank using exact distances on original vectors
            candidates = [(idx, approx_distances[idx]) for idx in candidate_indices]
            reranked = []

            for idx, _ in candidates:
                orig_vector = self._original_vectors[idx]
                exact_dist = float(np.sum((query - orig_vector) ** 2))
                reranked.append((idx, exact_dist))

            reranked.sort(key=lambda x: x[1])
            candidate_indices = [idx for idx, _ in reranked[:k]]
            final_distances = [dist for _, dist in reranked[:k]]
        else:
            final_distances = approx_distances[candidate_indices[:k]].tolist()
            candidate_indices = candidate_indices[:k]

        # Build results
        results = []
        for rank, (idx, dist) in enumerate(zip(candidate_indices, final_distances)):
            results.append(
                SearchResult(
                    vector_id=self._vector_ids[idx],
                    distance=float(np.sqrt(dist)),  # Return L2 distance, not squared
                    rank=rank,
                )
            )

        return results

    def get_stats(self) -> dict:
        """Get statistics about the index."""
        return {
            "is_trained": self._pq.is_trained,
            "num_vectors": len(self._vector_ids),
            "dim": self.dim,
            "compression_ratio": self._pq.compression_ratio,
            "params": {
                "M": self.params.M,
                "Ks": self.params.Ks,
                "dsub": self._pq.dsub,
                "bytes_per_vector": self.params.bytes_per_vector,
            },
        }
