# Project H: Vector Database

A production-grade vector similarity search database implementing HNSW, IVF, and Product Quantization algorithms for approximate nearest neighbor (ANN) search.

## Goal

Build a vector database demonstrating state-of-the-art approximate nearest neighbor search algorithms with configurable recall/speed tradeoffs.

## Key Requirements

* Multiple index types: HNSW (graph-based), IVF (clustering-based), Flat (brute-force)
* Distance metrics: L2 (Euclidean), Cosine, Inner Product
* Product Quantization for memory-efficient storage
* REST API for vector operations

## Evaluation Metrics

* Recall@10 > 90% for HNSW (default parameters)
* Sub-linear search complexity O(log n) for HNSW
* 64x compression ratio with Product Quantization

## Mandatory Tests

* Unit: Distance functions, HNSW operations, IVF clustering
* Benchmarks: Recall@K, QPS, latency percentiles
* Integration: Full vector lifecycle via REST API

## High-Level Overview

This vector database implements core similarity search algorithms from first principles, providing a learning platform for understanding how production systems like Pinecone, Milvus, and FAISS work internally.

### What Problem Does This Solve?

Understanding vector search internals is essential for:
- **Building AI applications** that need semantic search (RAG, recommendations)
- **Optimizing recall vs latency** tradeoffs for production workloads
- **Choosing index types** based on dataset size and memory constraints

### Key Design Decisions

- **HNSW**: Best general-purpose algorithm with O(log n) search (Malkov & Yashunin 2018)
- **IVF**: Scalable clustering-based approach for large datasets (FAISS, Johnson et al. 2019)
- **Product Quantization**: 64x compression via subspace quantization (Jegou et al. 2011)
- **Hexagonal Architecture**: Clean separation between domain logic and I/O

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REST API (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│                     VectorDatabase Service                       │
├─────────────────────────────────────────────────────────────────┤
│     HNSW Index     │     IVF Index      │    Flat Index         │
│   (Graph-based)    │  (Clustering)      │   (Brute-force)       │
├─────────────────────────────────────────────────────────────────┤
│                   Distance Functions                             │
│         L2 (Euclidean)  │  Cosine  │  Inner Product             │
├─────────────────────────────────────────────────────────────────┤
│                   Storage Adapters                               │
│              In-Memory  │  File-based (.npy)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Distance Functions (`domain/services/distance.py`)

CPU-optimized vector distance computations.

- **Purpose**: Compute similarity between vectors
- **Key Functions**: `euclidean_distance`, `cosine_distance`, `inner_product`
- **How it works**:
  - Single-vector: Direct NumPy computation
  - Batch: Optimized using `||a-b||² = ||a||² + ||b||² - 2·dot(a,b)`
- **Relationships**: Used by all index types

### HNSW Index (`domain/services/hnsw_index.py`)

Hierarchical Navigable Small World graph index.

- **Purpose**: O(log n) approximate nearest neighbor search
- **Key Classes**: `HNSWIndex`, `HNSWNode`, `HNSWParams`
- **How it works**:
  - Multi-layer graph with exponentially fewer nodes at higher layers
  - Insert: Greedy descent to find neighbors, bidirectional connections
  - Search: Greedy descent from top layer, beam search at layer 0
- **Reference**: Malkov & Yashunin (2018) arXiv:1603.09320

### IVF Index (`domain/services/ivf_index.py`)

Inverted File index with K-means clustering.

- **Purpose**: Scalable search by partitioning vector space
- **Key Classes**: `IVFIndex`, `IVFParams`, `InvertedList`
- **How it works**:
  - Training: K-means to find `nlist` centroids
  - Add: Assign each vector to nearest centroid
  - Search: Find `nprobe` closest centroids, search only those lists
- **Reference**: FAISS paper (Johnson et al. 2019)

### Product Quantization (`domain/services/pq_quantizer.py`)

Vector compression via subspace quantization.

- **Purpose**: 64x memory reduction with acceptable accuracy loss
- **Key Classes**: `ProductQuantizer`, `PQCodebook`, `PQIndex`
- **How it works**:
  - Split D-dimensional vector into M subvectors
  - K-means per subspace to create codebooks
  - Encode: Store M centroid indices (M bytes) instead of D floats (4D bytes)
  - Search: Asymmetric Distance Computation (ADC) using precomputed tables
- **Reference**: Jegou et al. (2011) IEEE TPAMI

### Infrastructure (`infrastructure/`)

Cross-cutting concerns: configuration, logging, metrics.

- **Key Files**:
  - `config.py` - Pydantic-based configuration with env var support
- **How it works**: Uses `VECTOR_DB_` prefix for environment variables

### Port Interfaces (`ports/`)

Abstract interfaces following Hexagonal Architecture.

- **Key Files**:
  - `ports/inbound/vector_database_port.py` - VectorDatabasePort protocol
  - `ports/outbound/vector_storage_port.py` - VectorStoragePort protocol
- **How it works**: Uses Python's Protocol for structural typing

### Adapters (`adapters/`)

Concrete implementations of port interfaces.

- **Inbound**: `rest_api.py` - FastAPI REST endpoints
- **Outbound**:
  - `memory_storage.py` - In-memory storage for testing
  - `file_storage.py` - NumPy .npy file persistence

## Quick Start

```bash
# Using Conda
conda env create -f conda.yaml
conda activate vector_db_env
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run smoke tests
python -m pytest tests/smoke_test.py -v
```

## REST API

```bash
# Health check
GET /health

# Insert vector
POST /vectors
{"id": "vec1", "vector": [0.1, 0.2, ...]}

# Search
POST /search
{"vector": [0.1, 0.2, ...], "k": 10}

# Get vector
GET /vectors/{id}

# Delete vector
DELETE /vectors/{id}

# Statistics
GET /stats
```

## Implementation Status

### Implemented

- [x] **Distance Functions** - L2, Cosine, Inner Product (single + batch)
- [x] **HNSW Index** - Full Algorithm 1-5 from paper
- [x] **IVF Index** - K-means training, nprobe search
- [x] **Product Quantization** - Codebook training, ADC search
- [x] **VectorDatabase Service** - Unified interface for all index types
- [x] **REST API** - Full CRUD + search endpoints
- [x] **Storage Adapters** - In-memory and file-based
- [x] **Unit Tests** - Distance functions, HNSW, benchmarks
- [x] **Smoke Tests** - Full integration coverage

### Planned

- [ ] **gRPC Server** - High-performance API
- [ ] **GPU Acceleration** - CUDA distance computation
- [ ] **Index Persistence** - Save/load trained indices
- [ ] **IVF-PQ** - Combined index for large-scale search

## Key Parameters

### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per node (12-48 recommended) |
| `M_max_0` | 32 | Max connections at layer 0 (typically 2*M) |
| `ef_construction` | 200 | Beam width during index building |
| `ef_search` | 50 | Beam width during search (higher = better recall) |

### IVF Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nlist` | 100 | Number of clusters (~√n for balanced tradeoff) |
| `nprobe` | 10 | Clusters to search (5-10% of nlist for 90%+ recall) |

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
vector_db/
├── src/vector_db/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Vector, VectorWithDistance
│   │   ├── services/            # HNSW, IVF, PQ, Distance
│   │   └── value_objects/       # VectorId, DistanceMetric, SearchResult
│   ├── ports/                   # Interface definitions
│   │   ├── inbound/             # VectorDatabasePort
│   │   └── outbound/            # VectorStoragePort
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # REST API
│   │   └── outbound/            # File/Memory storage
│   ├── application/             # Use cases / orchestration
│   │   └── vector_database.py   # VectorDatabase service
│   └── infrastructure/          # Config, logging
├── tests/
│   ├── unit/                    # Component tests
│   ├── benchmarks/              # Performance tests
│   └── smoke_test.py            # Integration tests
└── docs/
    └── design.md                # Detailed design document
```

## Documentation

- [Design Document](docs/design.md) - Architecture, algorithms, and academic references

## References

1. **HNSW**: Malkov, Y. & Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" IEEE TPAMI (2018)
2. **IVF/FAISS**: Johnson, J. et al. "Billion-scale similarity search with GPUs" IEEE Big Data (2019)
3. **Product Quantization**: Jegou, H. et al. "Product Quantization for Nearest Neighbor Search" IEEE TPAMI (2011)
4. **ANN Benchmarks**: http://ann-benchmarks.com/

## License

MIT
