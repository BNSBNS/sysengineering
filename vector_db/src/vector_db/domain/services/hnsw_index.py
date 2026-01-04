"""HNSW (Hierarchical Navigable Small World) index implementation.

This is a faithful implementation of the HNSW algorithm following the original paper:

Reference:
    Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and Robust Approximate
    Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs."
    IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 42(4), 824-836.
    DOI: 10.1109/TPAMI.2018.2889473
    arXiv: https://arxiv.org/abs/1603.09320

Algorithm Overview:
    HNSW builds a multi-layer graph where:
    - Layer 0 contains ALL vectors with short-range connections
    - Higher layers contain exponentially fewer nodes with longer-range connections
    - Search starts at the top layer and greedily descends
    - Uses beam search at the bottom layer for final candidate collection

Complexity:
    - Search: O(log n) expected
    - Insert: O(log n) expected
    - Memory: O(n * M * L) where M = connections per node, L = average layers
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vector_db.domain.services.distance import (
    BatchDistanceFunction,
    get_batch_distance_function,
    get_distance_function,
)
from vector_db.domain.value_objects.distance_metric import DistanceMetric
from vector_db.domain.value_objects.search_params import SearchResult

if TYPE_CHECKING:
    from vector_db.domain.services.distance import DistanceFunction


@dataclass
class HNSWParams:
    """HNSW index parameters.

    Reference: Table 1 in Malkov & Yashunin (2018)
    """

    M: int = 16  # Max connections per layer (paper recommends 12-48)
    M_max_0: int = 32  # Max connections at layer 0 (typically 2*M)
    ef_construction: int = 200  # Beam width during construction
    ef_search: int = 50  # Default beam width during search
    mL: float = field(default=0.0)  # Level multiplier, computed if not set

    def __post_init__(self) -> None:
        if self.mL == 0.0:
            # Default: 1/ln(M) as recommended in the paper
            self.mL = 1.0 / math.log(self.M)

        # Validation
        if not 4 <= self.M <= 64:
            raise ValueError(f"M should be in [4, 64], got {self.M}")
        if self.ef_construction < self.M:
            raise ValueError(f"ef_construction should be >= M")
        if self.mL <= 0:
            raise ValueError("mL must be positive")


@dataclass
class HNSWNode:
    """A node in the HNSW graph.

    Attributes:
        vector_id: Unique identifier for this vector
        vector: The embedding vector data
        level: Maximum level this node appears in (0 to level inclusive)
        neighbors: Dict mapping layer -> list of neighbor vector_ids
    """

    vector_id: str
    vector: NDArray[np.float32]
    level: int
    neighbors: dict[int, list[str]] = field(default_factory=dict)

    def get_neighbors(self, layer: int) -> list[str]:
        """Get neighbors at a specific layer."""
        return self.neighbors.get(layer, [])

    def set_neighbors(self, layer: int, neighbor_ids: list[str]) -> None:
        """Set neighbors at a specific layer."""
        self.neighbors[layer] = list(neighbor_ids)

    def add_neighbor(self, layer: int, neighbor_id: str) -> None:
        """Add a neighbor at a specific layer."""
        if layer not in self.neighbors:
            self.neighbors[layer] = []
        if neighbor_id not in self.neighbors[layer]:
            self.neighbors[layer].append(neighbor_id)


class HNSWIndex:
    """HNSW index for approximate nearest neighbor search.

    This implementation follows Algorithms 1-5 from Malkov & Yashunin (2018).

    Example:
        >>> index = HNSWIndex(dim=128, metric=DistanceMetric.L2)
        >>> index.insert("vec1", np.random.randn(128).astype(np.float32))
        >>> results = index.search(query_vector, k=10)
    """

    def __init__(
        self,
        dim: int,
        metric: DistanceMetric = DistanceMetric.L2,
        params: HNSWParams | None = None,
        seed: int | None = None,
    ):
        """Initialize the HNSW index.

        Args:
            dim: Dimensionality of vectors
            metric: Distance metric (L2, COSINE, or INNER_PRODUCT)
            params: HNSW parameters (uses defaults if not provided)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.metric = metric
        self.params = params or HNSWParams()

        # Distance functions
        self._distance: DistanceFunction = get_distance_function(metric)
        self._batch_distance: BatchDistanceFunction = get_batch_distance_function(metric)

        # Graph structure
        self._nodes: dict[str, HNSWNode] = {}
        self._entry_point: str | None = None
        self._max_level: int = 0

        # Random state
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._nodes)

    @property
    def entry_point(self) -> str | None:
        """Return the entry point vector ID."""
        return self._entry_point

    @property
    def max_level(self) -> int:
        """Return the maximum level in the graph."""
        return self._max_level

    def _random_level(self) -> int:
        """Generate a random level for a new node.

        Uses exponential distribution as described in the paper:
            l = floor(-ln(uniform(0,1)) * mL)

        This ensures ~63% of nodes are only in layer 0,
        ~23% reach layer 1, ~8% reach layer 2, etc.
        """
        # -ln(uniform(0,1)) follows exponential distribution
        return int(-math.log(self._rng.random()) * self.params.mL)

    def insert(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert a vector into the index.

        Implements Algorithm 1 (INSERT) from the paper.

        Args:
            vector_id: Unique identifier for the vector
            vector: The embedding vector to insert

        Raises:
            ValueError: If vector_id already exists or vector has wrong dimension
        """
        if vector_id in self._nodes:
            raise ValueError(f"Vector {vector_id} already exists")
        if len(vector) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(vector)}")

        # Ensure float32
        vector = vector.astype(np.float32)

        # Generate random level for this node
        level = self._random_level()

        # Create the node
        node = HNSWNode(vector_id=vector_id, vector=vector, level=level)

        # First node becomes entry point
        if self._entry_point is None:
            self._nodes[vector_id] = node
            self._entry_point = vector_id
            self._max_level = level
            # Initialize empty neighbor lists for all layers
            for lc in range(level + 1):
                node.set_neighbors(lc, [])
            return

        # Get current entry point
        entry_point = self._entry_point
        current_max_level = self._max_level

        # Phase 1: Traverse from top layer to level+1, finding the best entry point
        # At each layer, do greedy search with ef=1
        for lc in range(current_max_level, level, -1):
            # Search this layer with ef=1 to find closest node
            candidates = self._search_layer(vector, [entry_point], ef=1, layer=lc)
            if candidates:
                entry_point = candidates[0][0]  # Closest node becomes new entry

        # Store the node BEFORE adding connections (so it can be looked up)
        self._nodes[vector_id] = node

        # Phase 2: Insert into layers from min(level, max_level) down to 0
        for lc in range(min(level, current_max_level), -1, -1):
            # Search this layer to find candidates
            candidates = self._search_layer(
                vector, [entry_point], ef=self.params.ef_construction, layer=lc
            )

            # Select neighbors using the heuristic
            M = self.params.M_max_0 if lc == 0 else self.params.M
            neighbors = self._select_neighbors_heuristic(vector, candidates, M, lc)

            # Add bidirectional connections
            node.set_neighbors(lc, neighbors)

            for neighbor_id in neighbors:
                neighbor = self._nodes[neighbor_id]
                neighbor.add_neighbor(lc, vector_id)

                # Shrink neighbor's connections if exceeding max
                M_max = self.params.M_max_0 if lc == 0 else self.params.M
                if len(neighbor.get_neighbors(lc)) > M_max:
                    # Need to shrink - select best M_max neighbors
                    neighbor_candidates = [
                        (nid, self._distance(neighbor.vector, self._nodes[nid].vector))
                        for nid in neighbor.get_neighbors(lc)
                    ]
                    new_neighbors = self._select_neighbors_heuristic(
                        neighbor.vector, neighbor_candidates, M_max, lc
                    )
                    neighbor.set_neighbors(lc, new_neighbors)

            # Update entry point for next layer
            if candidates:
                entry_point = candidates[0][0]

        # Update entry point if new node has higher level
        if level > current_max_level:
            self._entry_point = vector_id
            self._max_level = level

    def _search_layer(
        self,
        query: NDArray[np.float32],
        entry_points: list[str],
        ef: int,
        layer: int,
    ) -> list[tuple[str, float]]:
        """Search a single layer using beam search.

        Implements Algorithm 2 (SEARCH-LAYER) from the paper.

        Args:
            query: Query vector
            entry_points: Starting node IDs
            ef: Beam width (number of candidates to track)
            layer: Layer to search

        Returns:
            List of (vector_id, distance) tuples, sorted by distance (ascending)
        """
        visited: set[str] = set(entry_points)

        # Candidates: min-heap of (-distance, id) - we use negative for max extraction
        # We want to expand from the closest candidates first
        candidates: list[tuple[float, str]] = []

        # Results: max-heap of (distance, id) - we keep the ef closest
        results: list[tuple[float, str]] = []

        # Initialize with entry points
        for ep_id in entry_points:
            ep_node = self._nodes[ep_id]
            dist = float(self._distance(query, ep_node.vector))
            heapq.heappush(candidates, (dist, ep_id))
            heapq.heappush(results, (-dist, ep_id))  # Negative for max-heap behavior

        while candidates:
            # Get closest candidate
            c_dist, c_id = heapq.heappop(candidates)

            # Get furthest result
            # results is a max-heap with negative distances
            f_dist = -results[0][0] if results else float("inf")

            # Stop if closest candidate is further than furthest result
            if c_dist > f_dist:
                break

            # Explore neighbors of current candidate
            c_node = self._nodes[c_id]
            for neighbor_id in c_node.get_neighbors(layer):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor = self._nodes[neighbor_id]
                dist = float(self._distance(query, neighbor.vector))

                f_dist = -results[0][0] if results else float("inf")

                # Add to results if closer than furthest or results not full
                if dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(results, (-dist, neighbor_id))

                    # Keep only ef best results
                    if len(results) > ef:
                        heapq.heappop(results)  # Remove furthest

        # Convert results to sorted list (ascending by distance)
        result_list = [(-d, vid) for d, vid in results]
        result_list.sort()
        return [(vid, d) for d, vid in result_list]

    def _select_neighbors_heuristic(
        self,
        query: NDArray[np.float32],
        candidates: list[tuple[str, float]],
        M: int,
        layer: int,
        extend_candidates: bool = True,
        keep_pruned: bool = True,
    ) -> list[str]:
        """Select M neighbors using the heuristic algorithm.

        Implements Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) from the paper.

        The heuristic prefers diverse neighbors that cover different directions,
        rather than just the M closest.

        Args:
            query: Query vector
            candidates: List of (vector_id, distance) candidates
            M: Maximum number of neighbors to select
            layer: Current layer (for extending candidates)
            extend_candidates: Whether to extend candidate set with neighbors
            keep_pruned: Whether to keep pruned candidates as fallback

        Returns:
            List of selected neighbor vector_ids
        """
        if not candidates:
            return []

        # Sort candidates by distance
        working_queue = sorted(candidates, key=lambda x: x[1])

        # Optionally extend candidates with their neighbors
        if extend_candidates:
            extended = set(c[0] for c in working_queue)
            for cid, _ in list(working_queue):
                cnode = self._nodes.get(cid)
                if cnode:
                    for neighbor_id in cnode.get_neighbors(layer):
                        if neighbor_id not in extended:
                            extended.add(neighbor_id)
                            neighbor = self._nodes[neighbor_id]
                            dist = float(self._distance(query, neighbor.vector))
                            working_queue.append((neighbor_id, dist))
            working_queue.sort(key=lambda x: x[1])

        result: list[str] = []
        discarded: list[tuple[str, float]] = []

        for cid, cdist in working_queue:
            if len(result) >= M:
                break

            # Check if this candidate is closer to query than to all selected neighbors
            # This ensures diversity - we prefer candidates in different "directions"
            is_good = True
            c_vector = self._nodes[cid].vector

            for selected_id in result:
                selected_vector = self._nodes[selected_id].vector
                dist_to_selected = float(self._distance(c_vector, selected_vector))

                # If candidate is closer to an existing neighbor than to query,
                # it's redundant (in the same "direction")
                if dist_to_selected < cdist:
                    is_good = False
                    break

            if is_good:
                result.append(cid)
            else:
                discarded.append((cid, cdist))

        # If we don't have enough neighbors, add from discarded
        if keep_pruned and len(result) < M:
            for cid, _ in discarded:
                if len(result) >= M:
                    break
                if cid not in result:
                    result.append(cid)

        return result

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int | None = None,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Implements Algorithm 5 (K-NN-SEARCH) from the paper.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Beam width (defaults to params.ef_search)
                Higher ef = better recall, slower search

        Returns:
            List of SearchResult objects, sorted by distance (ascending)
        """
        if not self._nodes:
            return []

        if len(query) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(query)}")

        ef = ef or self.params.ef_search
        if ef < k:
            ef = k  # ef must be at least k

        query = query.astype(np.float32)
        entry_point = self._entry_point

        # Phase 1: Greedy descent from top layer to layer 1
        for lc in range(self._max_level, 0, -1):
            candidates = self._search_layer(query, [entry_point], ef=1, layer=lc)
            if candidates:
                entry_point = candidates[0][0]

        # Phase 2: Beam search at layer 0
        candidates = self._search_layer(query, [entry_point], ef=ef, layer=0)

        # Return top k results
        results = []
        for rank, (vector_id, distance) in enumerate(candidates[:k]):
            results.append(SearchResult(vector_id=vector_id, distance=distance, rank=rank))

        return results

    def batch_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        ef: int | None = None,
    ) -> list[list[SearchResult]]:
        """Search for k nearest neighbors for multiple queries.

        Args:
            queries: Query vectors of shape (n_queries, dim)
            k: Number of neighbors per query
            ef: Beam width

        Returns:
            List of search results for each query
        """
        return [self.search(q, k, ef) for q in queries]

    def get_stats(self) -> dict:
        """Get statistics about the index."""
        if not self._nodes:
            return {
                "num_vectors": 0,
                "dim": self.dim,
                "max_level": 0,
                "avg_connections": 0,
            }

        total_connections = 0
        level_counts = {}

        for node in self._nodes.values():
            for layer, neighbors in node.neighbors.items():
                total_connections += len(neighbors)
                level_counts[layer] = level_counts.get(layer, 0) + 1

        return {
            "num_vectors": len(self._nodes),
            "dim": self.dim,
            "max_level": self._max_level,
            "avg_connections": total_connections / len(self._nodes) if self._nodes else 0,
            "level_distribution": level_counts,
            "entry_point": self._entry_point,
            "params": {
                "M": self.params.M,
                "M_max_0": self.params.M_max_0,
                "ef_construction": self.params.ef_construction,
                "ef_search": self.params.ef_search,
                "mL": self.params.mL,
            },
        }

    def contains(self, vector_id: str) -> bool:
        """Check if a vector ID exists in the index."""
        return vector_id in self._nodes

    def get_vector(self, vector_id: str) -> NDArray[np.float32] | None:
        """Get a vector by its ID."""
        node = self._nodes.get(vector_id)
        return node.vector if node else None
