# Vector Database Design Document

## High-Level Component Overview

This document describes a vector similarity search database implementing state-of-the-art approximate nearest neighbor (ANN) algorithms used by production systems like Pinecone, Milvus, Weaviate, and FAISS.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Vector Database                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  REST API    │───▶│  Vector      │───▶│   Index      │                   │
│  │  (FastAPI)   │    │  Database    │    │   Router     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                             │                   │                            │
│                             ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      Index Implementations                        │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │   HNSW     │  │    IVF     │  │   Flat     │  │  PQ Index  │  │       │
│  │  │ (Graph)    │  │ (Cluster)  │  │ (Brute)    │  │(Compressed)│  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Distance Functions                             │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐  │       │
│  │  │ Euclidean  │  │  Cosine    │  │    Inner Product           │  │       │
│  │  │   (L2)     │  │ Distance   │  │   (Dot Product)            │  │       │
│  │  └────────────┘  └────────────┘  └────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **HNSW** | O(log n) search | Best recall/speed tradeoff, industry standard [1] | Malkov & Yashunin 2018 |
| **IVF** | Scalable search | Partitions space, reduces search scope [2] | FAISS 2019 |
| **Product Quantization** | Memory efficiency | 64x compression with acceptable accuracy loss [3] | Jegou et al. 2011 |
| **Flat Index** | Exact search | Baseline for recall measurement, small datasets | Standard brute-force |
| **Distance Functions** | Similarity measurement | L2 for images, Cosine for text, IP for normalized | Standard metrics |

### How Data Flows Through the System

1. **Insert Path**: REST API → VectorDatabase → Index.insert() → Update graph/clusters
2. **Search Path**: REST API → VectorDatabase → Index.search() → Distance computation → Top-K results
3. **Training Path** (IVF/PQ): Training vectors → K-means clustering → Codebook/Centroids

### References

1. **HNSW**: Malkov, Y. & Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" IEEE TPAMI (2018)
2. **IVF/FAISS**: Johnson, J. et al. "Billion-scale similarity search with GPUs" IEEE Big Data (2019)
3. **Product Quantization**: Jegou, H. et al. "Product Quantization for Nearest Neighbor Search" IEEE TPAMI (2011)

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a vector similarity search database that demonstrates fundamental ANN algorithms:
- Graph-based search (HNSW)
- Clustering-based search (IVF)
- Compression techniques (Product Quantization)
- Multiple distance metrics

**Why build this?** Understanding vector search internals is essential for:
- Building RAG (Retrieval-Augmented Generation) applications
- Implementing semantic search and recommendations
- Optimizing recall vs latency tradeoffs for production

### Goals

- Implement HNSW with >90% recall@10
- Support multiple distance metrics (L2, Cosine, Inner Product)
- Provide IVF for scalable search with configurable nprobe
- Implement Product Quantization for 64x compression
- REST API for integration

### Non-Goals

- Distributed search (single-node only)
- GPU acceleration (CPU-only for learning)
- Hybrid search (vector + metadata filtering)
- Streaming updates during search

---

## 2. Core Algorithms

### HNSW (Hierarchical Navigable Small World)

#### Algorithm Overview

HNSW builds a multi-layer proximity graph where:
- **Layer 0**: Contains ALL vectors with short-range connections
- **Higher layers**: Contain exponentially fewer vectors with long-range connections
- **Search**: Greedy descent from top layer, beam search at layer 0

```
Layer 2:    [A]─────────────────────[D]         (Long-range, ~1/e² nodes)
             │                        │
Layer 1:    [A]────[B]───────[C]────[D]         (Medium-range, ~1/e nodes)
             │      │         │      │
Layer 0:    [A]─[E]─[B]─[F]─[G]─[C]─[H]─[D]     (Short-range, ALL nodes)
```

#### Key Parameters

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| Max connections | M | 16 | Higher = better recall, more memory |
| Layer 0 connections | M_max_0 | 32 | Typically 2*M |
| Construction beam | ef_construction | 200 | Higher = better index quality, slower build |
| Search beam | ef_search | 50 | Higher = better recall, slower search |
| Level multiplier | mL | 1/ln(M) | Controls layer distribution |

#### Algorithm 1: INSERT (from paper)

```
INSERT(hnsw, q, M, M_max, ef_construction, mL):
    W ← ∅  // candidates
    ep ← entry_point
    L ← max_level

    // Generate random level for new node
    l ← floor(-ln(uniform(0,1)) * mL)

    // Phase 1: Greedy descent to layer l+1
    for lc ← L down to l+1:
        W ← SEARCH-LAYER(q, ep, ef=1, lc)
        ep ← nearest element from W

    // Phase 2: Insert at layers l down to 0
    for lc ← min(L, l) down to 0:
        W ← SEARCH-LAYER(q, ep, ef_construction, lc)
        neighbors ← SELECT-NEIGHBORS(q, W, M, lc)
        add bidirectional connections from q to neighbors

        // Shrink connections if needed
        for e ∈ neighbors:
            if |connections(e)| > M_max:
                shrink to M_max using SELECT-NEIGHBORS

        ep ← nearest from W

    // Update entry point if needed
    if l > L:
        entry_point ← q
```

#### Algorithm 2: SEARCH-LAYER (Beam Search)

```
SEARCH-LAYER(q, ep, ef, layer):
    visited ← {ep}
    candidates ← min-heap with (dist(q, ep), ep)
    results ← max-heap with (dist(q, ep), ep)

    while candidates not empty:
        c ← extract-min(candidates)
        f ← peek-max(results)

        if dist(c) > dist(f):
            break  // Can't improve

        for e ∈ neighbors(c, layer):
            if e not in visited:
                visited.add(e)
                f ← peek-max(results)

                if dist(q, e) < dist(f) or |results| < ef:
                    candidates.push((dist(q, e), e))
                    results.push((dist(q, e), e))

                    if |results| > ef:
                        pop-max(results)

    return results sorted by distance
```

#### Complexity Analysis

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Insert | O(log n) | O(n) |
| Search | O(log n) | O(n) |
| Memory | O(n * M * L) | - |

---

### IVF (Inverted File Index)

#### Algorithm Overview

IVF partitions the vector space into Voronoi cells using K-means:
1. **Training**: Run K-means to find `nlist` centroids
2. **Add**: Assign each vector to nearest centroid
3. **Search**: Find `nprobe` closest centroids, search only those lists

```
       c0          c1          c2          c3      ← Centroids
       ↓           ↓           ↓           ↓
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ v1, v5  │  │ v2, v8  │  │ v3, v6  │  │ v4, v7  │  ← Inverted Lists
│ v9, v12 │  │ v10     │  │ v11     │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

#### Key Parameters

| Parameter | Default | Rule of Thumb |
|-----------|---------|---------------|
| nlist | sqrt(n) | Balance between list size and centroid comparison |
| nprobe | nlist * 0.05 | 5% for ~90% recall, 10% for ~95% recall |

#### Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Training | O(n * nlist * iterations) |
| Add | O(nlist) |
| Search | O(nlist + n * nprobe / nlist) |

---

### Product Quantization

#### Algorithm Overview

PQ compresses vectors by quantizing subspaces independently:

1. **Split**: Divide D-dimensional vector into M subvectors of D/M dimensions
2. **Train**: Run K-means on each subspace to get Ks centroids (codebook)
3. **Encode**: Replace each subvector with nearest centroid index (1 byte for Ks=256)
4. **Search**: Use Asymmetric Distance Computation (ADC)

```
Original: [────────────────────────────────────] 128 dims × 4 bytes = 512 bytes
           ↓ Split into M=8 subvectors
          [──][──][──][──][──][──][──][──]  8 × 16 dims
           ↓ Quantize each to centroid index
          [42][187][23][99][156][12][88][201]  8 bytes

Compression ratio: 512 / 8 = 64x
```

#### Asymmetric Distance Computation (ADC)

Instead of computing full distance, precompute lookup tables:

```
# Precompute for query q
for m in 0..M:
    for k in 0..Ks:
        table[m][k] = ||q_subm - centroid_mk||²

# Compute distance to database vector (stored as codes)
distance = sum(table[m][codes[m]] for m in 0..M)
```

This reduces per-vector comparison from O(D) multiplications to O(M) table lookups.

---

## 3. Distance Metrics

### L2 (Euclidean) Distance

```
d(a, b) = sqrt(sum((a[i] - b[i])²))
```

**Use case**: General purpose, image embeddings
**Range**: [0, ∞), lower = more similar

**Optimization**: Use squared distance for comparisons (avoids sqrt)
```
d²(a, b) = ||a||² + ||b||² - 2·dot(a, b)
```

### Cosine Distance

```
d(a, b) = 1 - (dot(a, b) / (||a|| * ||b||))
```

**Use case**: Text embeddings (angle matters, not magnitude)
**Range**: [0, 2], lower = more similar

**Optimization**: Pre-normalize vectors, then use inner product

### Inner Product

```
d(a, b) = -dot(a, b)  // Negated for distance semantics
```

**Use case**: Pre-normalized vectors (fastest)
**Range**: (-∞, ∞), lower = more similar (when negated)

---

## 4. API Design

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /vectors | Insert vector |
| POST | /search | Search k-NN |
| GET | /vectors/{id} | Get vector by ID |
| DELETE | /vectors/{id} | Delete vector |
| GET | /stats | Index statistics |
| GET | /health | Health check |

### Request/Response Examples

**Insert**:
```json
POST /vectors
{"id": "doc_123", "vector": [0.1, 0.2, ...]}
→ {"success": true}
```

**Search**:
```json
POST /search
{"vector": [0.1, 0.2, ...], "k": 10}
→ {"results": [{"id": "doc_456", "distance": 0.123, "rank": 0}, ...]}
```

---

## 5. Benchmarking Methodology

### Recall@K

```
Recall@K = |predicted ∩ ground_truth| / K
```

Ground truth computed using brute-force search.

### Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Recall@10 | >90% (HNSW) | Compare to brute-force |
| QPS | Maximize | Queries per second |
| Build time | Minimize | Time to index N vectors |
| Memory | Minimize | Bytes per vector |
| p50/p99 latency | Minimize | Search latency distribution |

### Benchmark Protocol

1. Generate random vectors (reproducible seed)
2. Build index with training vectors
3. Compute ground truth with brute-force
4. Measure recall at various parameter settings
5. Plot recall vs QPS curve

---

## 6. Implementation Notes

### NumPy Optimizations

- Use `float32` throughout (4 bytes vs 8 bytes for float64)
- Batch distance computations with matrix operations
- Use `argpartition` for top-K (O(n) vs O(n log n) for full sort)

### Memory Layout

- Vectors stored as contiguous NumPy arrays
- HNSW neighbors stored as dict[layer → list[id]]
- IVF lists stored as list[NDArray]

### Thread Safety

- HNSW: Thread-safe reads, single-writer for inserts
- IVF: Thread-safe after training

---

## 7. Future Enhancements

### IVF-PQ

Combine IVF partitioning with PQ compression:
- IVF reduces search scope
- PQ reduces memory and speeds up distance computation

### GPU Acceleration

- cuBLAS for batch distance computation
- CUDA kernels for HNSW traversal

### Hybrid Search

- Combine vector similarity with metadata filtering
- Pre-filtering vs post-filtering tradeoffs

---

## References

1. Malkov, Y. & Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" IEEE TPAMI (2018) - https://arxiv.org/abs/1603.09320
2. Johnson, J. et al. "Billion-scale similarity search with GPUs" IEEE Big Data (2019)
3. Jegou, H. et al. "Product Quantization for Nearest Neighbor Search" IEEE TPAMI (2011)
4. ANN Benchmarks - http://ann-benchmarks.com/
5. FAISS Wiki - https://github.com/facebookresearch/faiss/wiki
