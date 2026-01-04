"""Unit tests for distance functions."""

import numpy as np
import pytest

from vector_db.domain.services.distance import (
    batch_cosine_distance,
    batch_euclidean_distance,
    batch_inner_product,
    compute_ground_truth,
    compute_recall,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    get_batch_distance_function,
    get_distance_function,
    inner_product,
    normalize_vectors,
    squared_euclidean_distance,
)
from vector_db.domain.value_objects.distance_metric import DistanceMetric


class TestEuclideanDistance:
    """Tests for Euclidean (L2) distance."""

    def test_identical_vectors(self):
        """Distance between identical vectors is 0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_orthogonal_unit_vectors(self):
        """Distance between orthogonal unit vectors is sqrt(2)."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert euclidean_distance(a, b) == pytest.approx(np.sqrt(2))

    def test_symmetry(self):
        """Distance is symmetric: d(a, b) == d(b, a)."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_known_distance(self):
        """Test with known distance."""
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        assert euclidean_distance(a, b) == pytest.approx(5.0)


class TestSquaredEuclideanDistance:
    """Tests for squared Euclidean distance."""

    def test_preserves_ordering(self):
        """Squared distance preserves ordering."""
        q = np.array([0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([2.0, 0.0], dtype=np.float32)

        sq_a = squared_euclidean_distance(q, a)
        sq_b = squared_euclidean_distance(q, b)

        assert sq_a < sq_b

    def test_equals_squared_l2(self):
        """Squared distance equals L2 distance squared."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        l2 = euclidean_distance(a, b)
        sq = squared_euclidean_distance(a, b)

        assert sq == pytest.approx(l2 * l2)


class TestCosineDistance:
    """Tests for cosine distance."""

    def test_identical_vectors(self):
        """Cosine distance between identical vectors is 0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Cosine distance between orthogonal vectors is 1."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Cosine distance between opposite vectors is 2."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_distance(a, b) == pytest.approx(2.0)

    def test_scale_invariance(self):
        """Cosine distance is scale-invariant."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        d1 = cosine_distance(a, b)
        d2 = cosine_distance(a * 10, b * 5)

        assert d1 == pytest.approx(d2, rel=1e-4)

    def test_zero_vector(self):
        """Cosine distance with zero vector returns 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        assert cosine_distance(a, zero) == pytest.approx(1.0)


class TestCosineSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors(self):
        """Cosine similarity between identical vectors is 1."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Cosine similarity between orthogonal vectors is 0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_relationship_with_distance(self):
        """similarity = 1 - distance."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        sim = cosine_similarity(a, b)
        dist = cosine_distance(a, b)

        assert sim + dist == pytest.approx(1.0)


class TestInnerProduct:
    """Tests for inner product."""

    def test_orthogonal_vectors(self):
        """Inner product of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert inner_product(a, b) == pytest.approx(0.0)

    def test_known_value(self):
        """Test with known inner product."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert inner_product(a, b) == pytest.approx(32.0)

    def test_symmetry(self):
        """Inner product is symmetric."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert inner_product(a, b) == pytest.approx(inner_product(b, a))


class TestBatchDistanceFunctions:
    """Tests for batch distance functions."""

    @pytest.fixture
    def query_and_candidates(self):
        """Create a query and candidate vectors."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [2.0, 0.0, 0.0],  # Same direction, further
            [-1.0, 0.0, 0.0],  # Opposite
        ], dtype=np.float32)
        return query, candidates

    def test_batch_euclidean(self, query_and_candidates):
        """Test batch Euclidean distance."""
        query, candidates = query_and_candidates
        distances = batch_euclidean_distance(query, candidates)

        assert distances[0] == pytest.approx(0.0)  # Identical
        assert distances[1] == pytest.approx(np.sqrt(2))  # Orthogonal
        assert distances[2] == pytest.approx(1.0)  # Distance 1
        assert distances[3] == pytest.approx(2.0)  # Distance 2

    def test_batch_cosine(self, query_and_candidates):
        """Test batch cosine distance."""
        query, candidates = query_and_candidates
        distances = batch_cosine_distance(query, candidates)

        assert distances[0] == pytest.approx(0.0)  # Identical
        assert distances[1] == pytest.approx(1.0)  # Orthogonal
        assert distances[2] == pytest.approx(0.0)  # Same direction
        assert distances[3] == pytest.approx(2.0)  # Opposite

    def test_batch_inner_product(self, query_and_candidates):
        """Test batch inner product (negative for distance semantics)."""
        query, candidates = query_and_candidates
        distances = batch_inner_product(query, candidates)

        # Note: returns negative inner product
        assert distances[0] == pytest.approx(-1.0)  # IP = 1
        assert distances[1] == pytest.approx(0.0)  # IP = 0
        assert distances[2] == pytest.approx(-2.0)  # IP = 2
        assert distances[3] == pytest.approx(1.0)  # IP = -1

    def test_consistency_with_single(self):
        """Batch functions should be consistent with single-vector functions."""
        query = np.random.randn(128).astype(np.float32)
        candidates = np.random.randn(100, 128).astype(np.float32)

        batch_dists = batch_euclidean_distance(query, candidates)

        for i in range(len(candidates)):
            single_dist = euclidean_distance(query, candidates[i])
            assert batch_dists[i] == pytest.approx(single_dist, rel=1e-5)


class TestNormalizeVectors:
    """Tests for vector normalization."""

    def test_unit_norm(self):
        """Normalized vectors have unit norm."""
        vectors = np.random.randn(10, 128).astype(np.float32)
        normalized = normalize_vectors(vectors)

        norms = np.linalg.norm(normalized, axis=1)
        for norm in norms:
            assert norm == pytest.approx(1.0, rel=1e-5)

    def test_single_vector(self):
        """Normalization works for single vector."""
        v = np.array([3.0, 4.0], dtype=np.float32)
        normalized = normalize_vectors(v)

        assert np.linalg.norm(normalized) == pytest.approx(1.0)
        assert normalized[0] == pytest.approx(0.6)
        assert normalized[1] == pytest.approx(0.8)

    def test_zero_vector_unchanged(self):
        """Zero vector should be returned unchanged."""
        zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = normalize_vectors(zero)
        np.testing.assert_array_equal(result, zero)


class TestGetDistanceFunction:
    """Tests for distance function factory."""

    def test_l2(self):
        """Get L2 distance function."""
        func = get_distance_function(DistanceMetric.L2)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert func(a, b) == pytest.approx(np.sqrt(2))

    def test_cosine(self):
        """Get cosine distance function."""
        func = get_distance_function(DistanceMetric.COSINE)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert func(a, b) == pytest.approx(1.0)

    def test_inner_product(self):
        """Get inner product distance function."""
        func = get_distance_function(DistanceMetric.INNER_PRODUCT)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        # Returns negative inner product
        assert func(a, b) == pytest.approx(-1.0)


class TestComputeGroundTruth:
    """Tests for ground truth computation."""

    def test_small_dataset(self):
        """Test ground truth on small dataset."""
        # Create simple dataset where neighbors are obvious
        database = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        queries = np.array([
            [0.1, 0.1],  # Closest to [0, 0]
        ], dtype=np.float32)

        indices, distances = compute_ground_truth(queries, database, k=2)

        assert indices.shape == (1, 2)
        assert indices[0, 0] == 0  # Closest is [0, 0]

    def test_returns_sorted(self):
        """Ground truth returns neighbors sorted by distance."""
        np.random.seed(42)
        database = np.random.randn(100, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)

        indices, distances = compute_ground_truth(queries, database, k=10)

        # Check distances are sorted
        for i in range(len(queries)):
            for j in range(1, 10):
                assert distances[i, j] >= distances[i, j - 1]


class TestComputeRecall:
    """Tests for recall computation."""

    def test_perfect_recall(self):
        """Recall is 1.0 when predictions match ground truth."""
        predicted = np.array([[0, 1, 2, 3, 4]])
        ground_truth = np.array([[0, 1, 2, 3, 4]])

        recall = compute_recall(predicted, ground_truth, k=5)
        assert recall == pytest.approx(1.0)

    def test_zero_recall(self):
        """Recall is 0.0 when no predictions match."""
        predicted = np.array([[0, 1, 2, 3, 4]])
        ground_truth = np.array([[5, 6, 7, 8, 9]])

        recall = compute_recall(predicted, ground_truth, k=5)
        assert recall == pytest.approx(0.0)

    def test_partial_recall(self):
        """Recall is partial when some predictions match."""
        predicted = np.array([[0, 1, 2, 3, 4]])
        ground_truth = np.array([[0, 1, 5, 6, 7]])

        recall = compute_recall(predicted, ground_truth, k=5)
        assert recall == pytest.approx(2.0 / 5.0)  # 2 out of 5 match

    def test_multiple_queries(self):
        """Recall is averaged over multiple queries."""
        predicted = np.array([
            [0, 1, 2],  # 2/3 match
            [0, 1, 2],  # 1/3 match
        ])
        ground_truth = np.array([
            [0, 1, 5],  # 2/3 match
            [0, 5, 6],  # 1/3 match
        ])

        recall = compute_recall(predicted, ground_truth, k=3)
        expected = (2/3 + 1/3) / 2
        assert recall == pytest.approx(expected)
