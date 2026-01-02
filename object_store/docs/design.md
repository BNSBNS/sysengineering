# Object Store Design Document

## High-Level Component Overview

This document describes an S3-compatible object store implementing the principles from Amazon S3's design, the content-addressed storage approach of IPFS/Git, and Reed-Solomon erasure coding for fault tolerance.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Object Store                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ S3-compat    │───▶│  Processing  │───▶│  Durability  │                   │
│  │    API       │    │   (Dedup)    │    │    (EC)      │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Storage Layer                                  │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │  Content-  │  │  Merkle    │  │  Erasure   │  │  Metadata  │  │       │
│  │  │  Addressed │  │   Tree     │  │  Shards    │  │    DB      │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **S3 API** | Compatibility | Industry standard, existing tooling works [1] | AWS S3 API |
| **Content Addressing** | Deduplication | Same content → same hash → stored once [2] | Git, IPFS |
| **Chunking** | Large object handling | Stream processing, partial updates [3] | rsync algorithm |
| **Erasure Coding** | Fault tolerance | 1.5x storage for N+2 fault tolerance vs 3x replication [4] | Reed-Solomon |
| **Merkle Trees** | Integrity verification | Efficient verification, tamper detection [5] | Merkle 1979 |
| **Multipart Upload** | Large files | Resume uploads, parallel parts [6] | S3 multipart API |

### Why Content-Addressing?

```
Path-based storage:              Content-addressed storage:
┌─────────────────────┐          ┌─────────────────────┐
│ /bucket/file1.txt   │──┐       │ Hash(content)       │
│ /bucket/file2.txt   │──┼─Same  │   = abc123...       │
│ /backup/file1.txt   │──┘ data  │                     │
└─────────────────────┘          └─────────────────────┘
        │                                 │
        ▼                                 ▼
  3 copies stored                   1 copy stored
  (300% overhead)                   (pointers to hash)
  No integrity check                Built-in verification
```

### References

1. **S3 API**: AWS "Amazon S3 REST API Reference" docs.aws.amazon.com
2. **Content Addressing**: Benet, J. "IPFS - Content Addressed, Versioned, P2P File System" (2014)
3. **Chunking**: Muthitacharoen, A. et al. "A Low-bandwidth Network File System" SOSP (2001)
4. **Erasure Coding**: Plank, J. "Erasure Codes for Storage Systems: A Brief Primer" ;login: (2013)
5. **Merkle Trees**: Merkle, R. "A Certified Digital Signature" CRYPTO (1989)
6. **Multipart Upload**: AWS "Uploading and copying objects using multipart upload"

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build an S3-compatible object store with content-addressed storage, erasure coding for durability, and Merkle tree integrity verification.

**Why build this?** Understanding object storage is essential for:
- Designing cost-effective storage tiers (hot/warm/cold)
- Building backup and archival systems
- Understanding cloud storage internals

### Goals

- S3-compatible REST API (core operations)
- Content-addressed storage with deduplication
- Erasure coding (4+2 default) for fault tolerance
- Merkle tree integrity verification
- Multipart uploads for large objects

### Non-Goals

- Full S3 API compatibility (ACLs, versioning, lifecycle)
- Cross-region replication
- Server-side encryption
- Event notifications

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    S3-Compatible REST API                        ││
│  │     PUT/GET/DELETE Object, Multipart, Bucket Operations         ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                        Processing Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Chunking   │  │ Dedup       │  │    Integrity                │  │
│  │  Engine     │  │ Manager     │  │    Verifier                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      Durability Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Erasure    │  │  Merkle     │  │    Placement                │  │
│  │  Coding     │  │  Tree       │  │    Engine                   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Storage Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Content-Addressed Chunk Store                       ││
│  │                  (SHA-256 addressed)                             ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Upload Flow**: API → Chunk → Dedup check → Encode → Store shards → Update metadata
2. **Download Flow**: API → Metadata lookup → Read shards → Decode → Verify → Return
3. **Integrity Flow**: Merkle root → Verify path → Reconstruct if needed

---

## 3. Core Components & APIs

### Object Service

**S3-compatible operations**:
```
PUT /bucket/key          → Create or overwrite object
GET /bucket/key          → Retrieve object
DELETE /bucket/key       → Delete object
HEAD /bucket/key         → Get metadata only
GET /bucket?list-type=2  → List objects
PUT /bucket              → Create bucket
DELETE /bucket           → Delete bucket (must be empty)
```

```python
class ObjectService(Protocol):
    async def put_object(self, bucket: str, key: str, data: AsyncIterator[bytes]) -> ObjectMetadata: ...
    async def get_object(self, bucket: str, key: str) -> AsyncIterator[bytes]: ...
    async def delete_object(self, bucket: str, key: str) -> None: ...
    async def head_object(self, bucket: str, key: str) -> ObjectMetadata: ...
```

### Chunk Store

**Why content-addressing?**
```python
def store_chunk(data: bytes) -> ChunkRef:
    """
    Content-addressed storage:
    - Same data always produces same hash
    - Automatic deduplication
    - Built-in integrity verification
    """
    hash = sha256(data)

    # Check if already stored (dedup)
    if exists(hash):
        increment_refcount(hash)
        return ChunkRef(hash)

    # New chunk: encode and store
    shards = erasure_encode(data)
    for i, shard in enumerate(shards):
        store_shard(f"{hash}/shard_{i}", shard)

    return ChunkRef(hash)
```

```python
class ChunkStore(Protocol):
    async def store(self, data: bytes) -> ChunkRef: ...
    async def retrieve(self, ref: ChunkRef) -> bytes: ...
    async def exists(self, ref: ChunkRef) -> bool: ...
```

### Erasure Coding

**How Reed-Solomon works**:
```
Original data: 4 data chunks (D1, D2, D3, D4)

Encoding (4+2 RS code):
┌────┬────┬────┬────┬────┬────┐
│ D1 │ D2 │ D3 │ D4 │ P1 │ P2 │
└────┴────┴────┴────┴────┴────┘
  │    │    │    │    │    │
  Data chunks    Parity chunks

Can recover from ANY 2 failures:
- Lost D1, D2? Reconstruct from D3, D4, P1, P2 ✓
- Lost D1, P1? Reconstruct from D2, D3, D4, P2 ✓
- Lost P1, P2? No problem, data is complete ✓

Storage overhead: 6/4 = 1.5x (vs 3x for triple replication)
```

```python
class ErasureCoder(Protocol):
    def encode(self, data: bytes) -> list[Shard]: ...
    def decode(self, shards: list[Shard | None]) -> bytes: ...
    def reconstruct(self, shards: list[Shard | None], missing: list[int]) -> list[Shard]: ...
```

---

## 4. Data Models & State Machines

### Object Upload State Machine

```
    ┌─────────────┐    chunking     ┌─────────────┐
    │  RECEIVING  │ ──────────────▶ │  CHUNKING   │
    └─────────────┘                 └─────────────┘
                                          │
                                    dedup + encode
                                          │
                                          ▼
    ┌─────────────┐    verify       ┌─────────────┐
    │  COMPLETE   │ ◀────────────── │  STORING    │
    └─────────────┘                 └─────────────┘
          │
          │ error at any stage
          ▼
    ┌─────────────┐
    │   FAILED    │──▶ Cleanup partial data
    └─────────────┘
```

### Object Structure

```
Object (metadata DB):
├── bucket: str
├── key: str
├── size: int (total bytes)
├── etag: str (MD5 of content for S3 compat)
├── content_type: str
├── created_at: datetime
├── merkle_root: bytes (SHA-256)
└── chunks: list[ChunkRef]
    └── ChunkRef:
        ├── hash: SHA-256 (content address)
        ├── offset: int (position in object)
        ├── size: int (chunk size)
        └── shards: list[ShardLocation]
            └── ShardLocation:
                ├── node: str (storage node)
                ├── path: str (file path)
                └── index: int (0-3 data, 4-5 parity)
```

### Merkle Tree Structure

```
Object Merkle Tree (64MB object, 1MB chunks):

                     Root (published)
                    /                \
               Hash0-3              Hash4-7
              /      \             /      \
          H0-1      H2-3       H4-5      H6-7
         /    \    /    \     /    \    /    \
        H0    H1  H2    H3   H4    H5  H6    H7
        │     │   │     │    │     │   │     │
       C0    C1  C2    C3   C4    C5  C6    C7  (chunks)

Verification: To verify chunk C3:
1. Receive C3, compute H3 = SHA256(C3)
2. Receive proof: [H2, H0-1, Hash4-7]
3. Compute: H2-3 = SHA256(H2 || H3)
4. Compute: Hash0-3 = SHA256(H0-1 || H2-3)
5. Compute: Root' = SHA256(Hash0-3 || Hash4-7)
6. Verify: Root' == Root ✓
```

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| HTTP handling | asyncio (aiohttp) | High concurrency, connection pooling |
| Chunk I/O | aiofiles | Non-blocking disk operations |
| Erasure coding | Thread pool | CPU-bound (matrix math) |
| Merkle computation | Thread pool | CPU-bound (SHA-256) |

### Upload Pipeline

```python
async def put_object(self, bucket: str, key: str, data: AsyncIterator[bytes]) -> ObjectMetadata:
    """
    Streaming upload with concurrent encoding.

    Pipeline:
    [Receive] → [Chunk] → [Hash] → [Dedup] → [Encode] → [Store]
              ↓         ↓        ↓         ↓          ↓
           1MB      SHA-256   Check DB   4+2 RS    6 shards
    """
    chunks = []
    merkle_leaves = []

    async for chunk_data in chunk_stream(data, chunk_size=1024*1024):
        # Compute hash (content address)
        chunk_hash = await run_in_executor(sha256, chunk_data)
        merkle_leaves.append(chunk_hash)

        # Deduplication check
        if not await self.chunk_store.exists(chunk_hash):
            # Encode and store (concurrent with next chunk receive)
            shards = await run_in_executor(self.erasure.encode, chunk_data)
            await self.store_shards(chunk_hash, shards)

        chunks.append(ChunkRef(hash=chunk_hash, size=len(chunk_data)))

    # Build Merkle tree
    merkle_root = await run_in_executor(build_merkle_tree, merkle_leaves)

    # Store metadata
    return await self.metadata.put(bucket, key, chunks, merkle_root)
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Data Loss? |
|---------|-----------|----------|------------|
| Chunk corruption | SHA-256 mismatch | Reconstruct from EC shards | No |
| Missing shards (≤2) | Read failure | EC reconstruction | No |
| Missing shards (>2) | EC decode failure | Restore from backup | Yes |
| Metadata corruption | Checksum mismatch | Rebuild from chunks | Possible |
| Upload timeout | Session expiry | Cleanup incomplete parts | No (not committed) |

### Corruption Detection and Repair

```python
async def verify_and_repair(self, chunk_ref: ChunkRef) -> bool:
    """
    Background integrity check with automatic repair.
    """
    shards = await self.read_all_shards(chunk_ref)

    # Check each shard
    healthy_shards = []
    missing_indices = []
    for i, shard in enumerate(shards):
        if shard is None:
            missing_indices.append(i)
        elif sha256(shard) != chunk_ref.shard_hashes[i]:
            missing_indices.append(i)  # Treat corrupt as missing
        else:
            healthy_shards.append((i, shard))

    if len(missing_indices) == 0:
        return True  # All healthy

    if len(missing_indices) <= 2:
        # Can repair with erasure coding
        reconstructed = self.erasure.reconstruct(
            shards, missing_indices
        )
        for i, shard in zip(missing_indices, reconstructed):
            await self.store_shard(chunk_ref, i, shard)
        return True

    # Too many failures
    raise UnrecoverableError(f"Chunk {chunk_ref.hash} lost >2 shards")
```

---

## 7. Security Threat Model

| Threat | Asset | Mitigation |
|--------|-------|------------|
| Data tampering | Object content | Merkle tree verification, SHA-256 per chunk |
| Unauthorized access | Bucket contents | Bucket policies, request signatures |
| Replay attacks | API requests | Timestamp validation, nonce |
| DoS via large uploads | Storage capacity | Size limits, rate limiting, quotas |
| Data exfiltration | Object content | Access logging, anomaly detection |

### Request Signing (S3 Signature V4)

```python
def verify_signature(request: Request) -> bool:
    """
    AWS Signature V4 verification.

    Signature = HMAC-SHA256(
        HMAC-SHA256(
            HMAC-SHA256(
                HMAC-SHA256("AWS4" + secret_key, date),
                region
            ),
            service
        ),
        "aws4_request"
    )
    """
    # Extract components
    auth_header = request.headers["Authorization"]
    # AWS4-HMAC-SHA256 Credential=.../20240101/us-east-1/s3/aws4_request, ...

    # Compute expected signature
    string_to_sign = build_string_to_sign(request)
    signing_key = derive_signing_key(secret_key, date, region, "s3")
    expected_sig = hmac_sha256(signing_key, string_to_sign)

    return constant_time_compare(expected_sig, provided_sig)
```

---

## 8. Performance Targets

| Metric | Target | How Achieved |
|--------|--------|--------------|
| PUT latency (1MB) | < 100ms p99 | Streaming, async I/O |
| GET latency (1MB) | < 50ms p99 | Parallel shard reads |
| Throughput | > 1 GB/s aggregate | Concurrent connections |
| Dedup ratio | > 2x for similar data | Content addressing |
| EC overhead | 1.5x storage | 4+2 Reed-Solomon |

### Latency Breakdown

```
PUT 1MB object:
├── Network receive:     20ms (50 MB/s)
├── SHA-256 hash:        2ms
├── Dedup check:         1ms (in-memory bloom filter)
├── Erasure encode:      5ms (optimized RS library)
├── Store 6 shards:     30ms (parallel, local SSD)
├── Metadata update:     5ms (SQLite)
└── Response:            1ms
                        ────
                        ~65ms typical

GET 1MB object:
├── Metadata lookup:     2ms
├── Read 4 shards:      20ms (parallel, only data shards)
├── Erasure decode:      0ms (no decode if all present)
├── Network send:       20ms
└── Verify (optional):   2ms
                        ────
                        ~45ms typical
```

---

## 9. Operational Concerns

### Bucket Management

```bash
# Create bucket with storage class
object-store bucket create --name my-bucket --storage-class STANDARD

# List buckets with usage
object-store bucket list --show-size

# Get bucket statistics
object-store bucket stats --name my-bucket
# Output: Objects: 10,000, Size: 50GB, Dedup ratio: 2.3x
```

### Data Integrity

```bash
# Verify all objects in bucket (background job)
object-store verify --bucket my-bucket

# Repair corrupted objects
object-store repair --bucket my-bucket --key corrupted-object

# Garbage collection (remove unreferenced chunks)
object-store gc --dry-run
object-store gc --execute
```

### Capacity Planning

```bash
# Show storage breakdown
object-store storage stats
# Output:
# Raw data:      100 GB
# After dedup:    60 GB (1.67x ratio)
# With EC:        90 GB (1.5x overhead)
# Actual used:    90 GB

# Project future usage
object-store storage forecast --days 30
```

---

## 10. Alternatives Considered

### Durability Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Erasure Coding (4+2)** | Space efficient (1.5x) | CPU overhead | **Selected** |
| Triple replication | Simple, fast reads | 3x storage | For hot data tier |
| RAID-like striping | Fast | Single node failure | Rejected |

**Source**: Google, Facebook, Azure all use erasure coding for cold storage.

### Content Addressing

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **SHA-256** | Secure, automatic dedup | Hash computation | **Selected** |
| Path-based | Simple | No dedup, no verification | Rejected |
| UUID-based | Fast | No dedup | Rejected |

**Source**: IPFS and Git use content addressing. Amazon S3 uses ETags (MD5).

### Chunking Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Fixed 1MB chunks** | Simple, predictable | Suboptimal dedup | **Selected** |
| Content-defined (Rabin) | Better dedup | Complex, variable | Future enhancement |
| No chunking | Simplest | No partial updates | Rejected |

**Source**: rsync uses Rabin fingerprinting. Dropbox uses fixed chunks.

---

## Further Reading

1. **Amazon S3 Design**: DeCandia, G. et al. "Dynamo: Amazon's Highly Available Key-value Store" SOSP (2007)
2. **Erasure Coding**: Plank, J. "Erasure Codes for Storage Systems" ;login: USENIX (2013)
3. **Content Addressing**: Benet, J. "IPFS Whitepaper" (2014)
4. **Merkle Trees**: Crosby, S. & Wallach, D. "Efficient Data Structures for Tamper-Evident Logging" USENIX Security (2009)
5. **S3 API Reference**: docs.aws.amazon.com/AmazonS3/latest/API/
6. **Designing Data-Intensive Applications**: Kleppmann, M. Chapter 3 "Storage and Retrieval"
