"""Unit tests for HNSW index."""

import numpy as np
import pytest

from vector_db.domain.services.distance import compute_ground_truth, compute_recall
from vector_db.domain.services.hnsw_index import HNSWIndex, HNSWParams
from vector_db.domain.value_objects.distance_metric import DistanceMetric


class TestHNSWParams:
    """Tests for HNSW parameters."""

    def test_default_params(self):
        """Default parameters are valid."""
        params = HNSWParams()
        assert params.M == 16
        assert params.M_max_0 == 32
        assert params.ef_construction == 200
        assert params.ef_search == 50
        assert params.mL == pytest.approx(1.0 / np.log(16))

    def test_ml_computed_from_m(self):
        """mL is computed from M if not provided."""
        params = HNSWParams(M=32)
        assert params.mL == pytest.approx(1.0 / np.log(32))

    def test_invalid_m(self):
        """M outside valid range raises error."""
        with pytest.raises(ValueError):
            HNSWParams(M=2)  # Too small
        with pytest.raises(ValueError):
            HNSWParams(M=100)  # Too large

    def test_ef_construction_too_small(self):
        """ef_construction < M raises error."""
        with pytest.raises(ValueError):
            HNSWParams(M=16, ef_construction=8)


class TestHNSWIndex:
    """Tests for HNSW index."""

    @pytest.fixture
    def small_index(self):
        """Create a small HNSW index."""
        return HNSWIndex(
            dim=32,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=8, ef_construction=50, ef_search=20),
            seed=42,
        )

    @pytest.fixture
    def populated_index(self, small_index):
        """Create an index with some vectors."""
        np.random.seed(42)
        for i in range(100):
            vec = np.random.randn(32).astype(np.float32)
            small_index.insert(f"vec_{i}", vec)
        return small_index

    def test_empty_index(self, small_index):
        """Empty index has zero length."""
        assert len(small_index) == 0
        assert small_index.entry_point is None
        assert small_index.max_level == 0

    def test_insert_first_vector(self, small_index):
        """First vector becomes entry point."""
        vec = np.random.randn(32).astype(np.float32)
        small_index.insert("first", vec)

        assert len(small_index) == 1
        assert small_index.entry_point == "first"
        assert small_index.contains("first")

    def test_insert_multiple(self, small_index):
        """Can insert multiple vectors."""
        np.random.seed(42)
        for i in range(10):
            vec = np.random.randn(32).astype(np.float32)
            small_index.insert(f"vec_{i}", vec)

        assert len(small_index) == 10

    def test_insert_duplicate_raises(self, small_index):
        """Inserting duplicate ID raises error."""
        vec = np.random.randn(32).astype(np.float32)
        small_index.insert("duplicate", vec)

        with pytest.raises(ValueError, match="already exists"):
            small_index.insert("duplicate", vec)

    def test_insert_wrong_dimension(self, small_index):
        """Wrong dimension raises error."""
        vec = np.random.randn(64).astype(np.float32)  # Wrong dimension

        with pytest.raises(ValueError, match="Expected dim"):
            small_index.insert("wrong_dim", vec)

    def test_search_empty_index(self, small_index):
        """Searching empty index returns empty results."""
        query = np.random.randn(32).astype(np.float32)
        results = small_index.search(query, k=10)
        assert results == []

    def test_search_returns_k_results(self, populated_index):
        """Search returns k results."""
        query = np.random.randn(32).astype(np.float32)
        results = populated_index.search(query, k=10)

        assert len(results) == 10

    def test_search_results_sorted_by_distance(self, populated_index):
        """Results are sorted by distance (ascending)."""
        query = np.random.randn(32).astype(np.float32)
        results = populated_index.search(query, k=10)

        for i in range(1, len(results)):
            assert results[i].distance >= results[i - 1].distance

    def test_search_finds_exact_match(self, small_index):
        """Search finds an exact match."""
        vec = np.array([1.0] * 32, dtype=np.float32)
        small_index.insert("exact", vec)

        # Add some other vectors
        np.random.seed(42)
        for i in range(50):
            small_index.insert(f"other_{i}", np.random.randn(32).astype(np.float32))

        results = small_index.search(vec, k=1)

        assert len(results) == 1
        assert results[0].vector_id == "exact"
        assert results[0].distance == pytest.approx(0.0)

    def test_search_wrong_dimension(self, populated_index):
        """Wrong query dimension raises error."""
        query = np.random.randn(64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected dim"):
            populated_index.search(query, k=10)

    def test_higher_ef_improves_recall(self, populated_index):
        """Higher ef_search should improve recall."""
        np.random.seed(123)
        query = np.random.randn(32).astype(np.float32)

        # Get results with different ef values
        results_low = populated_index.search(query, k=10, ef=10)
        results_high = populated_index.search(query, k=10, ef=100)

        # Both should return results
        assert len(results_low) == 10
        assert len(results_high) == 10

        # Higher ef should find closer neighbors on average
        # (this is probabilistic, but with seed should be consistent)
        avg_dist_low = sum(r.distance for r in results_low) / 10
        avg_dist_high = sum(r.distance for r in results_high) / 10

        assert avg_dist_high <= avg_dist_low + 0.01  # Allow small tolerance

    def test_get_vector(self, populated_index):
        """Can retrieve vector by ID."""
        vec = populated_index.get_vector("vec_0")
        assert vec is not None
        assert vec.shape == (32,)

    def test_get_nonexistent_vector(self, populated_index):
        """Getting nonexistent vector returns None."""
        vec = populated_index.get_vector("nonexistent")
        assert vec is None

    def test_contains(self, populated_index):
        """Contains correctly reports membership."""
        assert populated_index.contains("vec_0")
        assert not populated_index.contains("nonexistent")

    def test_stats(self, populated_index):
        """Stats returns reasonable values."""
        stats = populated_index.get_stats()

        assert stats["num_vectors"] == 100
        assert stats["dim"] == 32
        assert stats["max_level"] >= 0
        assert stats["avg_connections"] > 0


class TestHNSWRecall:
    """Tests for HNSW recall quality."""

    @pytest.fixture
    def recall_test_data(self):
        """Create data for recall testing."""
        np.random.seed(42)
        n_vectors = 1000
        n_queries = 50
        dim = 64

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        return vectors, queries

    def test_recall_at_10(self, recall_test_data):
        """Recall@10 should be reasonably high."""
        vectors, queries = recall_test_data

        # Build index
        index = HNSWIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=200, ef_search=100),
            seed=42,
        )

        for i, vec in enumerate(vectors):
            index.insert(f"vec_{i}", vec)

        # Compute ground truth
        gt_indices, _ = compute_ground_truth(queries, vectors, k=10)

        # Search with HNSW
        predicted = np.zeros((len(queries), 10), dtype=np.int64)
        for i, query in enumerate(queries):
            results = index.search(query, k=10)
            for j, r in enumerate(results):
                predicted[i, j] = int(r.vector_id.split("_")[1])

        # Compute recall
        recall = compute_recall(predicted, gt_indices, k=10)

        # HNSW should achieve > 90% recall with these parameters
        assert recall > 0.90, f"Recall {recall:.2f} is too low"


class TestHNSWCosineMetric:
    """Tests for HNSW with cosine distance."""

    def test_cosine_search(self):
        """HNSW works with cosine distance."""
        np.random.seed(42)
        index = HNSWIndex(
            dim=32,
            metric=DistanceMetric.COSINE,
            params=HNSWParams(M=8, ef_construction=50, ef_search=20),
        )

        # Insert normalized vectors
        for i in range(50):
            vec = np.random.randn(32).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            index.insert(f"vec_{i}", vec)

        # Search with normalized query
        query = np.random.randn(32).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = index.search(query, k=5)
        assert len(results) == 5

        # Distances should be in [0, 2] for cosine
        for r in results:
            assert 0 <= r.distance <= 2


class TestHNSWBatchSearch:
    """Tests for batch search."""

    def test_batch_search(self):
        """Batch search returns results for all queries."""
        np.random.seed(42)
        index = HNSWIndex(dim=32, params=HNSWParams(M=8, ef_construction=50))

        # Insert vectors
        for i in range(100):
            vec = np.random.randn(32).astype(np.float32)
            index.insert(f"vec_{i}", vec)

        # Batch search
        queries = np.random.randn(10, 32).astype(np.float32)
        all_results = index.batch_search(queries, k=5)

        assert len(all_results) == 10
        for results in all_results:
            assert len(results) == 5
