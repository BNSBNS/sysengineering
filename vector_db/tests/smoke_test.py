"""Smoke test to verify all imports and basic functionality."""

import numpy as np

# Test all imports work - Domain layer
from vector_db.domain.value_objects import VectorId, DistanceMetric, SearchResult
from vector_db.domain.entities import Vector, VectorWithDistance
from vector_db.domain.services import (
    HNSWIndex, HNSWParams,
    IVFIndex, IVFParams,
    ProductQuantizer, PQIndex, PQParams,
    euclidean_distance, cosine_distance, inner_product,
    compute_recall, compute_ground_truth
)
from vector_db.infrastructure.config import Config, get_config

# Test Application layer imports
from vector_db.application import VectorDatabase, IndexType

# Test Ports layer imports
from vector_db.ports.inbound import VectorDatabasePort
from vector_db.ports.outbound import VectorStoragePort

# Test Adapters layer imports
from vector_db.adapters.inbound import create_app
from vector_db.adapters.outbound import FileVectorStorage, InMemoryVectorStorage


def test_hnsw():
    """Test HNSW index."""
    print('=== Testing HNSW Index ===')
    np.random.seed(42)
    index = HNSWIndex(dim=32, params=HNSWParams(M=8, ef_construction=50))
    for i in range(100):
        index.insert(f'v{i}', np.random.randn(32).astype(np.float32))
    results = index.search(np.random.randn(32).astype(np.float32), k=5)
    print(f'HNSW: {len(index)} vectors, search returned {len(results)} results')
    assert len(index) == 100
    assert len(results) == 5
    print('HNSW: PASSED')


def test_ivf():
    """Test IVF index."""
    print('\n=== Testing IVF Index ===')
    np.random.seed(42)
    ivf = IVFIndex(dim=32, params=IVFParams(nlist=10, nprobe=3))
    train_data = np.random.randn(500, 32).astype(np.float32)
    ivf.train(train_data)
    for i in range(100):
        ivf.add(f'v{i}', np.random.randn(32).astype(np.float32))
    results = ivf.search(np.random.randn(32).astype(np.float32), k=5)
    print(f'IVF: {len(ivf)} vectors, search returned {len(results)} results')
    assert len(ivf) == 100
    assert len(results) == 5
    print('IVF: PASSED')


def test_pq():
    """Test Product Quantization."""
    print('\n=== Testing Product Quantization ===')
    np.random.seed(42)
    pq = ProductQuantizer(dim=32, params=PQParams(M=4, Ks=256))
    pq.train(np.random.randn(500, 32).astype(np.float32))
    codes = pq.encode(np.random.randn(10, 32).astype(np.float32))
    print(f'PQ: compression ratio = {pq.compression_ratio:.1f}x, codes shape = {codes.shape}')
    assert pq.compression_ratio == 32.0  # 32*4 bytes -> 4 bytes
    assert codes.shape == (10, 4)
    print('PQ: PASSED')


def test_distance_functions():
    """Test distance functions."""
    print('\n=== Testing Distance Functions ===')
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    l2 = euclidean_distance(a, b)
    cos = cosine_distance(a, b)
    ip = inner_product(a, b)

    print(f'L2 distance: {l2:.4f} (expected: 1.4142)')
    print(f'Cosine distance: {cos:.4f} (expected: 1.0)')
    print(f'Inner product: {ip:.4f} (expected: 0.0)')

    assert abs(l2 - np.sqrt(2)) < 0.001
    assert abs(cos - 1.0) < 0.001
    assert abs(ip - 0.0) < 0.001
    print('Distance functions: PASSED')


def test_value_objects():
    """Test value objects."""
    print('\n=== Testing Value Objects ===')
    vid = VectorId.generate()
    print(f'Generated VectorId: {vid}')

    vec = Vector(id=VectorId("test"), data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    print(f'Vector dim: {vec.dim}, norm: {vec.norm:.4f}')

    assert vec.dim == 3
    assert abs(vec.norm - np.sqrt(14)) < 0.001
    print('Value objects: PASSED')


def test_vector_database():
    """Test VectorDatabase application service."""
    print('\n=== Testing VectorDatabase Application ===')
    np.random.seed(42)

    # Test HNSW-based database
    db = VectorDatabase(dim=32, index_type=IndexType.HNSW)
    for i in range(50):
        db.insert(f'v{i}', np.random.randn(32).astype(np.float32))

    results = db.search(np.random.randn(32).astype(np.float32), k=5)
    print(f'VectorDatabase (HNSW): {len(db)} vectors, search returned {len(results)} results')
    assert len(db) == 50
    assert len(results) == 5

    # Test get and contains
    assert db.contains('v0')
    vec = db.get('v0')
    assert vec is not None

    # Test stats
    stats = db.stats()
    assert stats['dim'] == 32
    assert stats['index_type'] == 'hnsw'

    print('VectorDatabase: PASSED')


def test_storage_adapters():
    """Test storage adapters."""
    print('\n=== Testing Storage Adapters ===')
    import tempfile

    # Test InMemoryVectorStorage
    mem_storage = InMemoryVectorStorage()
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mem_storage.save_vector('test', vec)
    loaded = mem_storage.load_vector('test')
    assert loaded is not None
    assert np.allclose(vec, loaded)
    print('InMemoryVectorStorage: PASSED')

    # Test FileVectorStorage
    with tempfile.TemporaryDirectory() as tmpdir:
        file_storage = FileVectorStorage(tmpdir)
        file_storage.save_vector('test', vec)
        loaded = file_storage.load_vector('test')
        assert loaded is not None
        assert np.allclose(vec, loaded)
        assert 'test' in file_storage.list_vectors()
        file_storage.delete_vector('test')
        assert file_storage.load_vector('test') is None
    print('FileVectorStorage: PASSED')


def test_rest_api():
    """Test REST API adapter."""
    print('\n=== Testing REST API Adapter ===')
    from fastapi.testclient import TestClient

    db = VectorDatabase(dim=4, index_type=IndexType.HNSW)
    app = create_app(db)
    client = TestClient(app)

    # Test health endpoint
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

    # Test insert
    response = client.post('/vectors', json={'id': 'v1', 'vector': [1.0, 2.0, 3.0, 4.0]})
    assert response.status_code == 200
    assert response.json()['success'] is True

    # Test search
    response = client.post('/search', json={'vector': [1.0, 2.0, 3.0, 4.0], 'k': 1})
    assert response.status_code == 200
    assert len(response.json()['results']) == 1
    assert response.json()['results'][0]['id'] == 'v1'

    # Test get
    response = client.get('/vectors/v1')
    assert response.status_code == 200
    assert response.json()['id'] == 'v1'

    # Test stats
    response = client.get('/stats')
    assert response.status_code == 200
    assert response.json()['num_vectors'] == 1

    print('REST API: PASSED')


if __name__ == '__main__':
    test_value_objects()
    test_distance_functions()
    test_hnsw()
    test_ivf()
    test_pq()
    test_vector_database()
    test_storage_adapters()
    test_rest_api()
    print('\n' + '='*50)
    print('=== ALL SMOKE TESTS PASSED! ===')
    print('='*50)
