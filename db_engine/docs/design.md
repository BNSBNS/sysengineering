# Database Engine Design Document

## High-Level Component Overview

This document describes a single-node ACID-compliant database engine implementing the fundamental algorithms that power production databases like PostgreSQL, MySQL, and SQLite.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Database Engine                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  SQL Parser  │───▶│  Query       │───▶│  Execution   │                   │
│  │  (sqlglot)   │    │  Planner     │    │   Engine     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Transaction Manager (MVCC)                     │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │  Snapshot  │  │   Lock     │  │  Deadlock  │  │ Version    │  │       │
│  │  │  Manager   │  │  Manager   │  │  Detector  │  │  Store     │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Storage Engine                                 │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │  Buffer    │  │  B+Tree    │  │    WAL     │  │   Disk     │  │       │
│  │  │   Pool     │  │   Index    │  │  (ARIES)   │  │  Manager   │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **ARIES WAL** | Crash recovery | Industry standard, supports REDO/UNDO. Used by DB2, SQL Server, PostgreSQL [1] | IBM Research 1992 |
| **MVCC** | Concurrency | Readers don't block writers. Used by PostgreSQL, Oracle, MySQL InnoDB [2] | Reed's dissertation 1978 |
| **B+Tree Index** | Efficient lookup | O(log n) search, range queries, cache-friendly. Used by virtually all RDBMS [3] | Bayer & McCreight 1972 |
| **Buffer Pool (LRU)** | Memory management | Exploits temporal locality, simple eviction policy [4] | Standard OS concept |
| **Slotted Pages** | Variable-length records | PostgreSQL, SQLite approach. Efficient space utilization [5] | Database internals |
| **2PL Locking** | Isolation guarantee | Serializability. Combined with MVCC for optimal read performance [6] | Gray & Reuter 1993 |

### How Data Flows Through the System

1. **Query Path**: Client → SQL Parser → Planner → Optimizer → Executor → Storage
2. **Write Path**: Executor → WAL (fsync) → Buffer Pool → Background Flush to Disk
3. **Recovery Path**: WAL Replay → Buffer Pool → Verify Checksums

### References

1. **ARIES**: Mohan, C. et al. "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging" TODS (1992)
2. **MVCC**: Reed, D. "Naming and Synchronization in a Decentralized Computer System" MIT PhD Thesis (1978)
3. **B+Tree**: Bayer, R. & McCreight, E. "Organization and Maintenance of Large Ordered Indices" Acta Informatica (1972)
4. **Buffer Management**: Effelsberg, W. & Haerder, T. "Principles of Database Buffer Management" TODS (1984)
5. **Page Formats**: PostgreSQL Documentation "Database Physical Storage"
6. **Transaction Processing**: Gray, J. & Reuter, A. "Transaction Processing: Concepts and Techniques" Morgan Kaufmann (1993)

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a single-node, production-grade database engine that demonstrates fundamental database concepts:
- ACID transactions with MVCC
- Crash recovery with WAL
- Efficient indexing with B+Tree
- Both OLTP and OLAP query execution

**Why build this?** Understanding database internals is essential for:
- Performance tuning production databases (buffer pool sizing, index selection)
- Debugging transaction anomalies (phantom reads, write skew)
- Designing data-intensive applications (choosing isolation levels)

### Goals

- Implement ARIES-style WAL with guaranteed crash recovery
- Support snapshot isolation via MVCC
- Provide B+Tree indexing with concurrent access
- Support both iterator-based (OLTP) and vectorized (OLAP) execution
- Achieve deterministic recovery across repeated runs

### Non-Goals

- Distributed transactions (single-node only)
- Full SQL compatibility (subset only)
- Production-grade security (focus on correctness)
- Network protocol compatibility (custom gRPC API)
- Query optimization beyond basic cost-based planning

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  SQL Parser │  │ gRPC Server │  │    Prometheus Metrics       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                          Query Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │   Planner   │  │  Optimizer  │  │    Execution Engine         │  │
│  │  (Logical)  │  │(Cost-based) │  │  (Iterator + Vectorized)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Transaction Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │    MVCC     │  │Lock Manager │  │    Deadlock Detector        │  │
│  │  (Snapshot) │  │   (2PL)     │  │   (Wait-for Graph)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Storage Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Buffer Pool │  │Index Manager│  │        WAL Manager          │  │
│  │   (LRU)     │  │  (B+Tree)   │  │        (ARIES)              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                          Disk Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Disk Manager (mmap / O_DIRECT)                     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Query Execution**: Parser → AST → Logical Plan → Physical Plan → Iterator Tree
2. **Transaction Flow**: BEGIN → Operations → WAL Entries → COMMIT (WAL flush) → ACK
3. **Recovery Flow**: Read WAL → Analysis → Redo → Undo → Ready

---

## 3. Core Components & APIs

### Buffer Pool

**Why a buffer pool?** Disk I/O is 100,000x slower than memory access. The buffer pool caches frequently accessed pages, achieving hit ratios >99% in production workloads.

**How it works**:
- Fixed-size page frames in memory
- LRU eviction policy tracks access recency
- Pin/unpin semantics prevent eviction during use
- Dirty page tracking enables lazy writes

```python
class BufferPool(Protocol):
    def fetch_page(self, page_id: PageId) -> Page: ...
    def new_page(self) -> Page: ...
    def unpin_page(self, page_id: PageId, is_dirty: bool) -> bool: ...
    def flush_page(self, page_id: PageId) -> bool: ...
    def flush_all_pages(self) -> None: ...
```

### WAL Manager

**Why Write-Ahead Logging?** The WAL guarantee: **A transaction is durable if and only if its COMMIT record is on stable storage.** This enables:
- Fast commits (sequential WAL write vs. random data page writes)
- Crash recovery (replay WAL to restore state)

**How ARIES works** (Algorithm for Recovery and Isolation Exploiting Semantics):
- **Analysis**: Scan WAL to find active transactions and dirty pages
- **Redo**: Replay all operations from oldest dirty page LSN
- **Undo**: Roll back uncommitted transactions

```python
class WALManager(Protocol):
    def append_log(self, record: LogRecord) -> LSN: ...
    def flush(self, lsn: LSN) -> None: ...
    def recover(self) -> None: ...
    def checkpoint(self) -> None: ...
```

### Transaction Manager

**Why MVCC?** Multi-Version Concurrency Control allows:
- Readers never block writers
- Writers never block readers
- Snapshot isolation without locks on read

**How snapshots work**: Each transaction sees a consistent view of data as of its start time. Writes create new versions; old versions are garbage collected after all readers finish.

```python
class TransactionManager(Protocol):
    def begin(self) -> Transaction: ...
    def commit(self, txn: Transaction) -> None: ...
    def abort(self, txn: Transaction) -> None: ...
    def get_snapshot(self, txn: Transaction) -> Snapshot: ...
```

### Index Manager

**Why B+Trees?** Optimal for disk-based systems:
- High fanout (100-1000) minimizes tree height
- All data in leaves enables efficient range scans
- Sequential leaf traversal exploits disk prefetching

```python
class IndexManager(Protocol):
    def create_index(self, table: str, column: str) -> Index: ...
    def insert(self, index: Index, key: Key, rid: RID) -> None: ...
    def delete(self, index: Index, key: Key) -> None: ...
    def search(self, index: Index, key: Key) -> Optional[RID]: ...
    def range_scan(self, index: Index, low: Key, high: Key) -> Iterator[RID]: ...
```

---

## 4. Data Models & State Machines

### Page Layout (Slotted Page)

**Why slotted pages?** Supports variable-length records while allowing record movement during compaction without updating external references.

```
┌─────────────────────────────────────────────────────────────┐
│ Page Header (24 bytes)                                       │
│ ┌─────────┬─────────┬──────────┬─────────┬─────────────────┐│
│ │ Page ID │   LSN   │ Checksum │ Slot Cnt│  Free Space Ptr ││
│ │ (4B)    │  (8B)   │   (4B)   │  (2B)   │     (2B)        ││
│ └─────────┴─────────┴──────────┴─────────┴─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│ Slot Array (grows downward)                                  │
│ ┌────────┬────────┬────────┬────────┐                       │
│ │ Slot 0 │ Slot 1 │ Slot 2 │  ...   │                       │
│ │(offset,│(offset,│(offset,│        │                       │
│ │ length)│ length)│ length)│        │                       │
│ └────────┴────────┴────────┴────────┘                       │
├─────────────────────────────────────────────────────────────┤
│                    Free Space                                │
├─────────────────────────────────────────────────────────────┤
│ Records (grow upward from bottom)                            │
│ ┌────────────────┬────────────────┬────────────────────────┐│
│ │    Record 2    │    Record 1    │       Record 0         ││
│ └────────────────┴────────────────┴────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Transaction State Machine

```
         ┌───────────────────────────────────────┐
         │                                       │
         ▼                                       │
    ┌─────────┐     begin()     ┌─────────────┐ │
    │  IDLE   │ ───────────────▶│   ACTIVE    │ │
    └─────────┘                 └─────────────┘ │
                                      │         │
                    ┌─────────────────┼─────────┤
                    │                 │         │
              commit()            abort()       │
                    │                 │         │
                    ▼                 ▼         │
            ┌─────────────┐   ┌─────────────┐   │
            │ COMMITTING  │   │  ABORTING   │   │
            └─────────────┘   └─────────────┘   │
                    │                 │         │
                    │                 │         │
                    ▼                 ▼         │
            ┌─────────────┐   ┌─────────────┐   │
            │  COMMITTED  │   │   ABORTED   │───┘
            └─────────────┘   └─────────────┘
```

### WAL Record Types

**Why these record types?** Each maps to an ARIES operation:

| Record | Purpose | Recovery Action |
|--------|---------|-----------------|
| BEGIN | Start transaction | Add to active set |
| COMMIT | End successfully | Remove from active, no undo needed |
| ABORT | End unsuccessfully | Trigger undo |
| UPDATE | Data modification | Redo if committed, undo if not |
| CLR | Compensation | Never undo (marks undo progress) |
| CHECKPOINT | Recovery speedup | Analysis starts here |

```
LogRecord:
  - BEGIN(txn_id)
  - COMMIT(txn_id)
  - ABORT(txn_id)
  - UPDATE(txn_id, page_id, offset, before_image, after_image)
  - INSERT(txn_id, page_id, offset, data)
  - DELETE(txn_id, page_id, offset, data)
  - CHECKPOINT(active_txns, dirty_pages)
  - CLR(txn_id, undo_next_lsn)  # Compensation Log Record
```

---

## 5. Concurrency Model

### Threading Strategy

| Component | Threading Model | Rationale |
|-----------|-----------------|-----------|
| Client connections | asyncio | High concurrency, I/O-bound (network wait) |
| Query execution | Thread pool | CPU-bound query processing |
| Buffer pool | Lock-based | Fine-grained page locking enables parallelism |
| WAL writes | Single writer | Sequential I/O performance (no seek overhead) |
| Checkpointing | Background thread | Non-blocking normal operations |
| Deadlock detection | Periodic thread | Low overhead (100ms interval typical) |

### Lock Hierarchy

**Why lock hierarchy?** Prevents deadlocks by enforcing consistent lock ordering. Intention locks allow coarse-grained compatibility checks before acquiring fine-grained locks.

```
Database Lock (intention locks only)
    └── Table Lock (IX/IS/X/S)
        └── Page Lock (X/S)
            └── Row Lock (X/S)
```

### MVCC Visibility Rules

**How snapshot isolation works**: A record is visible to a transaction T if:
1. It was created by a committed transaction before T started
2. It was not deleted (or deleted by a transaction that started after T)

```python
def is_visible(record: Record, snapshot: Snapshot) -> bool:
    # Created after snapshot? Not visible
    if record.created_by > snapshot.txn_id:
        return False

    # Created by concurrent transaction? Not visible
    if record.created_by in snapshot.active_txns:
        return False

    # Not deleted? Visible
    if record.deleted_by is None:
        return True

    # Deleted after snapshot? Still visible
    if record.deleted_by > snapshot.txn_id:
        return True

    # Deleted by concurrent transaction? Still visible
    if record.deleted_by in snapshot.active_txns:
        return True

    # Deleted before snapshot by committed txn
    return False
```

---

## 6. Failure Modes & Recovery

### Failure Scenarios

| Failure | Detection | Recovery | Source |
|---------|-----------|----------|--------|
| Process crash | Startup check | ARIES recovery | WAL replay |
| Page corruption | Checksum mismatch | Restore from backup/WAL | CRC32 validation |
| Disk failure | I/O error | External backup | Beyond scope |
| Deadlock | Wait-for graph cycle | Victim abort | Graph cycle detection |
| OOM | Memory allocation failure | Transaction abort | Graceful degradation |

### ARIES Recovery Algorithm

**Why three phases?** Separates concerns:
- Analysis: Figure out what happened
- Redo: Restore to crash-time state (history repeating)
- Undo: Remove effects of incomplete transactions

1. **Analysis Phase**: Scan WAL from last checkpoint
   - Rebuild active transaction table (ATT)
   - Rebuild dirty page table (DPT)
   - Determine redo start point (min recLSN in DPT)

2. **Redo Phase**: Replay from redo point
   - Apply all logged operations
   - Restore database to crash state
   - **Key**: Redo even for aborted transactions (then undo)

3. **Undo Phase**: Rollback losers
   - Undo uncommitted transactions in reverse LSN order
   - Write CLR records (ensures idempotence on repeated crash)

### Recovery Guarantees

- **Atomicity**: All-or-nothing via undo phase
- **Durability**: COMMIT record on disk = transaction persists
- **Determinism**: Same WAL → same database state (testable!)

---

## 7. Security Threat Model

### STRIDE Analysis

| Threat | Asset | Mitigation |
|--------|-------|------------|
| **S**poofing | Client identity | Authentication (out of scope) |
| **T**ampering | Data pages | Checksums on all pages (CRC32) |
| **R**epudiation | Operations | Audit logging (WAL provides this) |
| **I**nformation Disclosure | Data | Access control (basic) |
| **D**enial of Service | Resources | Resource limits (connections, memory) |
| **E**levation of Privilege | System access | Sandboxing (container isolation) |

### Input Validation

- **SQL injection**: Parameterized queries via sqlglot AST
- **Buffer overflow**: Bounds checking on all page access
- **Integer overflow**: Checked arithmetic for page offsets

---

## 8. Performance Targets

### OLTP Workload (TPC-C style)

| Metric | Target | How Measured |
|--------|--------|--------------|
| p50 latency | < 5ms | Simple point queries |
| p95 latency | < 20ms | Mixed read/write |
| p99 latency | < 50ms | Under load |
| Throughput | > 1000 txn/s | Concurrent clients |

### OLAP Workload (Analytical)

| Metric | Target | How Measured |
|--------|--------|--------------|
| Scan throughput | > 100 MB/s | Sequential table scan |
| Aggregation | > 10M rows/s | COUNT/SUM/AVG |
| Vectorized ops | > 1B ops/s | SIMD operations (NumPy/Numba) |

### Resource Limits

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Buffer pool | 1GB default | 80% of available memory typical |
| WAL segment | 64MB | Balance between checkpoint frequency and space |
| Max connections | 100 | Prevent resource exhaustion |
| Query timeout | 300s | Kill runaway queries |

---

## 9. Operational Concerns

### Deployment

```bash
# Docker deployment
docker-compose up -d

# Kubernetes deployment
kubectl apply -f k8s/
```

### Configuration

```yaml
# config.yaml
storage:
  data_dir: /data
  wal_dir: /wal
  page_size: 4096
  buffer_pool_size: 1073741824  # 1GB

wal:
  segment_size: 67108864  # 64MB
  sync_mode: fsync  # fsync | fdatasync | none

server:
  host: 0.0.0.0
  port: 5432
  max_connections: 100
```

### Monitoring

Key metrics exposed via Prometheus:
- `db_transactions_total{status="committed|aborted"}` - Transaction outcomes
- `db_query_latency_seconds` - Query latency percentiles
- `db_buffer_pool_hit_ratio` - Cache effectiveness (target: >0.99)
- `db_wal_bytes_written_total` - Write volume
- `db_checkpoint_duration_seconds` - Checkpoint overhead

### Backup & Restore

```bash
# Backup (includes WAL for point-in-time recovery)
db-engine backup --output /backup/$(date +%Y%m%d).tar.gz

# Restore
db-engine restore --input /backup/20240101.tar.gz
```

---

## 10. Alternatives Considered & Tradeoffs

### Storage Engine

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Slotted pages** | Simple, proven, PostgreSQL model | Internal fragmentation | **Selected** |
| Log-structured | Write-optimized | Read amplification, compaction | For write-heavy workloads |
| Column store | OLAP-optimized | OLTP overhead | Separate OLAP engine |

**Source**: PostgreSQL uses slotted pages; RocksDB uses log-structured merge.

### Concurrency Control

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **MVCC + 2PL** | Read-write isolation | Version storage overhead | **Selected** |
| Pure 2PL | Simple | Readers block writers | High-contention workloads |
| OCC | No blocking | High abort rate under contention | Read-heavy workloads |

**Source**: PostgreSQL, MySQL InnoDB use MVCC. Google Spanner uses OCC.

### WAL Implementation

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **ARIES** | Proven, flexible, partial rollback | Complex implementation | **Selected** |
| Shadow paging | Simple recovery | No incremental writes | SQLite uses this |
| Command logging | Space efficient | Slow replay (execute commands) | VoltDB approach |

**Source**: ARIES paper (1992) is the foundation for DB2, SQL Server, PostgreSQL WAL.

### Index Structure

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **B+Tree** | Range queries, proven | Memory overhead for pointers | **Selected** |
| Hash index | O(1) lookup | No range queries | Secondary option |
| LSM tree | Write-optimized | Read amplification | Not for OLTP |

**Source**: Every major RDBMS uses B+Trees. LevelDB/RocksDB use LSM.

---

## Further Reading

1. **Database Internals**: Petrov, A. "Database Internals" O'Reilly (2019)
2. **ARIES Paper**: Mohan, C. et al. "ARIES" ACM TODS (1992)
3. **PostgreSQL Internals**: interdb.jp/pg/ (free online book)
4. **CMU Database Course**: 15-445/645 Intro to Database Systems (YouTube)
5. **Transaction Processing**: Gray, J. & Reuter, A. "Transaction Processing" (1993)
6. **B+Tree Tutorial**: cstack.github.io/db_tutorial/
