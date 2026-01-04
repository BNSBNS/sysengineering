"""
Recall and QPS benchmarks following ann-benchmarks methodology.

Reference:
    Aumuller, M., Bernhardsson, E., & Faithfull, A. (2020).
    "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms."
    Information Systems, 87.
    Website: http://ann-benchmarks.com/
"""

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pytest

from vector_db.domain.services.distance import compute_ground_truth, compute_recall
from vector_db.domain.services.hnsw_index import HNSWIndex, HNSWParams
from vector_db.domain.services.ivf_index import IVFIndex, IVFParams
from vector_db.domain.services.pq_quantizer import PQIndex, PQParams
from vector_db.domain.value_objects.distance_metric import DistanceMetric


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    algorithm: str
    params: str
    recall_at_k: float
    qps: float
    qps_batch: float
    build_time_seconds: float
    index_size_bytes: int
    latency_p50_ms: float
    latency_p99_ms: float
    num_vectors: int
    dim: int
    k: int


class IndexProtocol(Protocol):
    """Protocol for indexable structures."""

    def __len__(self) -> int: ...


def generate_random_dataset(
    n_vectors: int, n_queries: int, dim: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random vectors for benchmarking."""
    np.random.seed(seed)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    return vectors, queries


class TestRecallBenchmarks:
    """Recall benchmarks for all index types."""

    @pytest.fixture
    def benchmark_data_small(self):
        """Small dataset for quick benchmarks."""
        return generate_random_dataset(n_vectors=1000, n_queries=50, dim=64)

    @pytest.fixture
    def benchmark_data_medium(self):
        """Medium dataset for thorough benchmarks."""
        return generate_random_dataset(n_vectors=10000, n_queries=100, dim=128)

    @pytest.fixture
    def ground_truth_small(self, benchmark_data_small):
        """Ground truth for small dataset."""
        vectors, queries = benchmark_data_small
        indices, _ = compute_ground_truth(queries, vectors, k=10)
        return indices

    @pytest.fixture
    def ground_truth_medium(self, benchmark_data_medium):
        """Ground truth for medium dataset."""
        vectors, queries = benchmark_data_medium
        indices, _ = compute_ground_truth(queries, vectors, k=10)
        return indices

    def test_hnsw_recall_at_10_small(self, benchmark_data_small, ground_truth_small):
        """HNSW recall@10 on small dataset."""
        vectors, queries = benchmark_data_small
        k = 10

        index = HNSWIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=200, ef_search=100),
            seed=42,
        )

        # Build index
        for i, vec in enumerate(vectors):
            index.insert(f"v{i}", vec)

        # Search
        predicted = np.zeros((len(queries), k), dtype=np.int64)
        for i, query in enumerate(queries):
            results = index.search(query, k=k)
            for j, r in enumerate(results):
                predicted[i, j] = int(r.vector_id[1:])  # Remove 'v' prefix

        recall = compute_recall(predicted, ground_truth_small, k=k)
        assert recall > 0.90, f"HNSW recall@10 = {recall:.3f}, expected > 0.90"

    def test_ivf_recall_at_10_small(self, benchmark_data_small, ground_truth_small):
        """IVF recall@10 on small dataset."""
        vectors, queries = benchmark_data_small
        k = 10

        index = IVFIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=IVFParams(nlist=32, nprobe=8),
        )

        # Train and build
        index.train(vectors)
        for i, vec in enumerate(vectors):
            index.add(f"v{i}", vec)

        # Search
        predicted = np.zeros((len(queries), k), dtype=np.int64)
        for i, query in enumerate(queries):
            results = index.search(query, k=k)
            for j, r in enumerate(results):
                predicted[i, j] = int(r.vector_id[1:])

        recall = compute_recall(predicted, ground_truth_small, k=k)
        # IVF recall depends on nprobe/nlist ratio - 25% probing yields ~70% recall
        assert recall > 0.65, f"IVF recall@10 = {recall:.3f}, expected > 0.65"

    def test_pq_recall_at_10_small(self, benchmark_data_small, ground_truth_small):
        """PQ recall@10 on small dataset (lossy compression)."""
        vectors, queries = benchmark_data_small
        k = 10

        index = PQIndex(
            dim=64,
            params=PQParams(M=8, Ks=256),
        )

        # Train
        index.train(vectors)

        # Add vectors
        vector_ids = [f"v{i}" for i in range(len(vectors))]
        index.add_batch(vector_ids, vectors)

        # Search
        predicted = np.zeros((len(queries), k), dtype=np.int64)
        for i, query in enumerate(queries):
            results = index.search(query, k=k)
            for j, r in enumerate(results):
                predicted[i, j] = int(r.vector_id[1:])

        recall = compute_recall(predicted, ground_truth_small, k=k)
        # PQ has lower recall due to compression
        assert recall > 0.50, f"PQ recall@10 = {recall:.3f}, expected > 0.50"


class TestQPSBenchmarks:
    """QPS (queries per second) benchmarks."""

    @pytest.fixture
    def benchmark_data(self):
        """Dataset for QPS benchmarks."""
        return generate_random_dataset(n_vectors=5000, n_queries=100, dim=64)

    def test_hnsw_qps(self, benchmark_data, benchmark):
        """Measure HNSW QPS."""
        vectors, queries = benchmark_data

        index = HNSWIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=100, ef_search=50),
            seed=42,
        )

        # Build index
        for i, vec in enumerate(vectors):
            index.insert(f"v{i}", vec)

        # Benchmark search
        def run_queries():
            for query in queries:
                index.search(query, k=10)

        benchmark(run_queries)

    def test_ivf_qps(self, benchmark_data, benchmark):
        """Measure IVF QPS."""
        vectors, queries = benchmark_data

        index = IVFIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=IVFParams(nlist=64, nprobe=4),
        )

        # Train and build
        index.train(vectors)
        for i, vec in enumerate(vectors):
            index.add(f"v{i}", vec)

        # Benchmark search
        def run_queries():
            for query in queries:
                index.search(query, k=10)

        benchmark(run_queries)

    def test_pq_qps(self, benchmark_data, benchmark):
        """Measure PQ QPS."""
        vectors, queries = benchmark_data

        index = PQIndex(
            dim=64,
            params=PQParams(M=8, Ks=256),
        )

        # Train and build
        index.train(vectors)
        vector_ids = [f"v{i}" for i in range(len(vectors))]
        index.add_batch(vector_ids, vectors)

        # Benchmark search
        def run_queries():
            for query in queries:
                index.search(query, k=10)

        benchmark(run_queries)


class TestBuildTimeBenchmarks:
    """Index build time benchmarks."""

    @pytest.fixture
    def build_data(self):
        """Dataset for build benchmarks."""
        return generate_random_dataset(n_vectors=2000, n_queries=10, dim=64)

    def test_hnsw_build_time(self, build_data, benchmark):
        """Measure HNSW build time."""
        vectors, _ = build_data

        def build_index():
            index = HNSWIndex(
                dim=64,
                metric=DistanceMetric.L2,
                params=HNSWParams(M=12, ef_construction=100),
                seed=42,
            )
            for i, vec in enumerate(vectors):
                index.insert(f"v{i}", vec)
            return index

        benchmark(build_index)

    def test_ivf_build_time(self, build_data, benchmark):
        """Measure IVF build time (train + add)."""
        vectors, _ = build_data

        def build_index():
            index = IVFIndex(
                dim=64,
                metric=DistanceMetric.L2,
                params=IVFParams(nlist=32, nprobe=4),
            )
            index.train(vectors)
            for i, vec in enumerate(vectors):
                index.add(f"v{i}", vec)
            return index

        benchmark(build_index)


class TestParameterSweep:
    """Parameter sweep to generate recall-QPS pareto curves."""

    @pytest.fixture
    def sweep_data(self):
        """Dataset for parameter sweep."""
        vectors, queries = generate_random_dataset(
            n_vectors=5000, n_queries=50, dim=64
        )
        gt_indices, _ = compute_ground_truth(queries, vectors, k=10)
        return vectors, queries, gt_indices

    def test_hnsw_ef_sweep(self, sweep_data):
        """Sweep ef_search to find recall-QPS tradeoff."""
        vectors, queries, gt_indices = sweep_data
        k = 10

        # Build index once
        index = HNSWIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=200, ef_search=10),
            seed=42,
        )
        for i, vec in enumerate(vectors):
            index.insert(f"v{i}", vec)

        results = []
        ef_values = [10, 20, 50, 100, 200]

        for ef in ef_values:
            # Measure QPS
            start = time.perf_counter()
            predicted = np.zeros((len(queries), k), dtype=np.int64)
            for i, query in enumerate(queries):
                search_results = index.search(query, k=k, ef=ef)
                for j, r in enumerate(search_results):
                    predicted[i, j] = int(r.vector_id[1:])
            elapsed = time.perf_counter() - start

            qps = len(queries) / elapsed
            recall = compute_recall(predicted, gt_indices, k=k)
            results.append((ef, recall, qps))

        # Verify recall increases with ef
        recalls = [r[1] for r in results]
        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1] - 0.05, (
                f"Recall should generally increase with ef: {results}"
            )

        # Print results for analysis
        print("\nHNSW ef_search sweep:")
        print("ef\tRecall@10\tQPS")
        for ef, recall, qps in results:
            print(f"{ef}\t{recall:.4f}\t\t{qps:.1f}")

    def test_ivf_nprobe_sweep(self, sweep_data):
        """Sweep nprobe to find recall-QPS tradeoff."""
        vectors, queries, gt_indices = sweep_data
        k = 10

        # Build index once
        index = IVFIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=IVFParams(nlist=64, nprobe=1),
        )
        index.train(vectors)
        for i, vec in enumerate(vectors):
            index.add(f"v{i}", vec)

        results = []
        nprobe_values = [1, 2, 4, 8, 16, 32]

        for nprobe in nprobe_values:
            # Measure QPS
            start = time.perf_counter()
            predicted = np.zeros((len(queries), k), dtype=np.int64)
            for i, query in enumerate(queries):
                search_results = index.search(query, k=k, nprobe=nprobe)
                for j, r in enumerate(search_results):
                    predicted[i, j] = int(r.vector_id[1:])
            elapsed = time.perf_counter() - start

            qps = len(queries) / elapsed
            recall = compute_recall(predicted, gt_indices, k=k)
            results.append((nprobe, recall, qps))

        # Verify recall increases with nprobe
        recalls = [r[1] for r in results]
        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1] - 0.05, (
                f"Recall should generally increase with nprobe: {results}"
            )

        # Print results for analysis
        print("\nIVF nprobe sweep:")
        print("nprobe\tRecall@10\tQPS")
        for nprobe, recall, qps in results:
            print(f"{nprobe}\t{recall:.4f}\t\t{qps:.1f}")


class TestLatencyBenchmarks:
    """Latency percentile benchmarks."""

    @pytest.fixture
    def latency_data(self):
        """Dataset for latency benchmarks."""
        return generate_random_dataset(n_vectors=5000, n_queries=200, dim=64)

    def test_hnsw_latency_percentiles(self, latency_data):
        """Measure HNSW p50 and p99 latency."""
        vectors, queries = latency_data

        index = HNSWIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=100, ef_search=50),
            seed=42,
        )

        # Build index
        for i, vec in enumerate(vectors):
            index.insert(f"v{i}", vec)

        # Measure latencies
        latencies_ms = []
        for query in queries:
            start = time.perf_counter()
            index.search(query, k=10)
            elapsed = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed)

        p50 = np.percentile(latencies_ms, 50)
        p99 = np.percentile(latencies_ms, 99)

        print(f"\nHNSW latency (n={len(vectors)}, d=64):")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # p99 should be reasonable (< 100ms for this size)
        assert p99 < 100, f"p99 latency {p99:.2f}ms too high"

    def test_ivf_latency_percentiles(self, latency_data):
        """Measure IVF p50 and p99 latency."""
        vectors, queries = latency_data

        index = IVFIndex(
            dim=64,
            metric=DistanceMetric.L2,
            params=IVFParams(nlist=64, nprobe=8),
        )

        # Train and build
        index.train(vectors)
        for i, vec in enumerate(vectors):
            index.add(f"v{i}", vec)

        # Measure latencies
        latencies_ms = []
        for query in queries:
            start = time.perf_counter()
            index.search(query, k=10)
            elapsed = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed)

        p50 = np.percentile(latencies_ms, 50)
        p99 = np.percentile(latencies_ms, 99)

        print(f"\nIVF latency (n={len(vectors)}, d=64, nlist=64, nprobe=8):")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # p99 should be reasonable
        assert p99 < 100, f"p99 latency {p99:.2f}ms too high"


class TestCompressionRatio:
    """Memory and compression benchmarks."""

    def test_pq_compression_ratio(self):
        """Verify PQ compression achieves expected ratio."""
        dim = 128
        M = 8  # 8 subquantizers with 256 centroids each = 8 bytes per vector

        index = PQIndex(
            dim=dim,
            params=PQParams(M=M, Ks=256),
        )

        # Original: 128 floats * 4 bytes = 512 bytes per vector
        # Compressed: 8 bytes per vector
        expected_ratio = (dim * 4) / M  # 64x

        stats = index.get_stats()
        assert stats["compression_ratio"] == expected_ratio, (
            f"Expected {expected_ratio}x compression, got {stats['compression_ratio']}x"
        )

    def test_memory_usage_comparison(self):
        """Compare memory usage across index types."""
        n_vectors = 1000
        dim = 128

        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        # Raw vectors: n * dim * 4 bytes
        raw_size = n_vectors * dim * 4

        # HNSW: vectors + graph structure
        hnsw = HNSWIndex(
            dim=dim,
            metric=DistanceMetric.L2,
            params=HNSWParams(M=16, ef_construction=50),
            seed=42,
        )
        for i, vec in enumerate(vectors):
            hnsw.insert(f"v{i}", vec)

        hnsw_stats = hnsw.get_stats()

        # IVF: vectors + centroids
        ivf = IVFIndex(
            dim=dim,
            metric=DistanceMetric.L2,
            params=IVFParams(nlist=32, nprobe=4),
        )
        ivf.train(vectors)
        for i, vec in enumerate(vectors):
            ivf.add(f"v{i}", vec)

        ivf_stats = ivf.get_stats()

        # PQ: compressed codes + codebooks
        pq = PQIndex(
            dim=dim,
            params=PQParams(M=8, Ks=256),
        )
        pq.train(vectors)
        vector_ids = [f"v{i}" for i in range(len(vectors))]
        pq.add_batch(vector_ids, vectors)

        pq_stats = pq.get_stats()

        # Memory estimates
        # HNSW: vectors (n*d*4) + graph (estimated as n * M * 2 * 8 bytes for IDs)
        hnsw_vectors_bytes = n_vectors * dim * 4
        hnsw_graph_bytes = int(n_vectors * hnsw_stats["avg_connections"] * 8)

        # IVF: vectors (n*d*4) + centroids (nlist*d*4)
        ivf_vectors_bytes = n_vectors * dim * 4
        ivf_centroids_bytes = ivf_stats["nlist"] * dim * 4

        # PQ: codes (n*M bytes) + codebooks (M*Ks*dsub*4 bytes)
        pq_codes_bytes = n_vectors * pq_stats["params"]["M"]
        pq_codebooks_bytes = (
            pq_stats["params"]["M"]
            * pq_stats["params"]["Ks"]
            * pq_stats["params"]["dsub"]
            * 4
        )

        print(f"\nMemory comparison (n={n_vectors}, d={dim}):")
        print(f"  Raw vectors:  {raw_size:,} bytes ({raw_size / 1024:.1f} KB)")
        print(
            f"  HNSW:         {hnsw_vectors_bytes:,} vectors + {hnsw_graph_bytes:,} graph "
            f"({hnsw_stats['avg_connections']:.1f} avg connections)"
        )
        print(
            f"  IVF:          {ivf_vectors_bytes:,} vectors + {ivf_centroids_bytes:,} centroids"
        )
        print(
            f"  PQ:           {pq_codes_bytes:,} codes + {pq_codebooks_bytes:,} codebooks "
            f"= {pq_stats['compression_ratio']:.0f}x compression"
        )

        # Verify PQ achieves significant compression
        assert pq_codes_bytes < raw_size / 10, "PQ should compress by at least 10x"
