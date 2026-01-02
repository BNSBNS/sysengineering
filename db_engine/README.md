# Project A: OLTP to OLAP Database Engine

A production-grade, single-node database engine demonstrating WAL (Write-Ahead Logging), MVCC (Multi-Version Concurrency Control), crash recovery, B+Tree indexing, and vectorized OLAP execution.

## Goal

Build a single-node, production-grade database engine demonstrating WAL, MVCC, recovery, indexing, and vectorized OLAP execution.

## Key Requirements

* Fixed-size pages with checksums
* ARIES-style WAL and recovery
* Snapshot isolation via MVCC
* B+Tree indexing
* Iterator + vectorized execution engines

## Evaluation Metrics

* Crash recovery correctness: 100% pass rate
* OLTP p95 latency < defined hardware baseline
* Deterministic recovery across repeated runs

## Mandatory Tests

* Unit: pages, WAL, index operations
* Integration: transactional workloads
* Chaos: crash at every WAL boundary
* Benchmarks: OLTP + OLAP workloads

## High-Level Overview

This database engine implements core database concepts from first principles, providing a learning platform for understanding how production databases like PostgreSQL, MySQL, and SQLite work internally.

### What Problem Does This Solve?

Understanding database internals is essential for:
- **Performance tuning** production databases (buffer pool sizing, index selection)
- **Debugging** transaction anomalies (phantom reads, write skew)
- **Designing** data-intensive applications (choosing isolation levels)

### Key Design Decisions

- **ARIES WAL**: Industry-standard recovery algorithm (used by DB2, PostgreSQL)
- **MVCC**: Readers never block writers (PostgreSQL, Oracle approach)
- **B+Tree**: O(log n) lookups, efficient range scans
- **Slotted Pages**: Variable-length records with slot stability

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SQL Interface                             │
├─────────────────────────────────────────────────────────────────┤
│  Parser (sqlglot)  │  Planner/Optimizer  │  Execution Engine    │
├─────────────────────────────────────────────────────────────────┤
│                    Transaction Manager (MVCC)                    │
├─────────────────────────────────────────────────────────────────┤
│     Buffer Pool     │     Index Manager    │    Lock Manager     │
├─────────────────────────────────────────────────────────────────┤
│                    Write-Ahead Log (WAL)                         │
├─────────────────────────────────────────────────────────────────┤
│                    Disk Manager (O_DIRECT)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Value Objects (`domain/value_objects/`)

Core type-safe identifiers and enumerations.

- **Purpose**: Provide type safety and prevent misuse of raw integers
- **Key Files**:
  - `identifiers.py` - PageId, TransactionId, LSN, RecordId
  - `transaction_types.py` - TransactionState, IsolationLevel, LockMode
- **How it works**: Uses Python's `NewType` for zero-cost type hints and frozen dataclasses for immutability
- **Relationships**: Used by all other components

### Slotted Page (`domain/entities/page.py`)

Variable-length record storage using the slotted page format.

- **Purpose**: Store variable-length records while allowing compaction without breaking external references
- **Key Files**: `page.py` - PageHeader, Slot, SlottedPage
- **How it works**:
  - 24-byte header (page_id, LSN, checksum, slot_count, free_space_ptr)
  - Slot array grows downward from header
  - Records grow upward from bottom
  - CRC32 checksums detect corruption
- **Relationships**: Used by Buffer Pool and WAL for persistence

### MVCC Records (`domain/entities/record.py`)

Records with multi-version concurrency control metadata.

- **Purpose**: Enable snapshot isolation without blocking
- **Key Files**: `record.py` - RecordHeader, Record, Snapshot
- **How it works**:
  - Each record has `created_by` (xmin) and `deleted_by` (xmax) transaction IDs
  - Visibility rules determine if a record is visible to a given snapshot
  - Old versions garbage collected after all readers finish
- **Relationships**: Stored in SlottedPages, visibility checked by Transaction Manager

### WAL Records (`domain/entities/wal_record.py`)

Write-Ahead Log record types for crash recovery.

- **Purpose**: Enable crash recovery via ARIES algorithm
- **Key Files**: `wal_record.py` - LogRecord (base), BeginRecord, CommitRecord, UpdateRecord, CLRRecord, CheckpointRecord
- **How it works**:
  - Each operation writes a log record before modifying data
  - Records contain before/after images for undo/redo
  - CLR records track undo progress for repeated crash safety
  - Checkpoint records speed up recovery
- **Relationships**: Written by all data-modifying operations, read by Recovery Service

### Infrastructure (`infrastructure/`)

Cross-cutting concerns: configuration, logging, metrics, tracing.

- **Purpose**: Provide operational observability and configurability
- **Key Files**:
  - `config.py` - Pydantic-based configuration
  - `logging.py` - Structured logging via structlog
  - `metrics.py` - Prometheus metrics
  - `tracing.py` - OpenTelemetry distributed tracing
  - `container.py` - Dependency injection
- **How it works**: Uses environment variables for configuration, exports metrics on `/metrics` endpoint
- **Relationships**: Used by all components

### Port Interfaces (`ports/`)

Abstract interfaces (protocols) following Hexagonal Architecture.

- **Purpose**: Define contracts between layers, enabling testability and flexibility
- **Key Files**:
  - `ports/inbound/buffer_pool.py` - BufferPool protocol with fetch/unpin/flush
  - `ports/inbound/wal_manager.py` - WALManager protocol with append/flush/recover
  - `ports/inbound/transaction_manager.py` - TransactionManager protocol with begin/commit/abort
  - `ports/inbound/index_manager.py` - IndexManager protocol with B+Tree operations
  - `ports/outbound/disk_manager.py` - DiskManager protocol for page I/O
  - `ports/outbound/wal_writer.py` - WALWriter protocol for WAL segment persistence
- **How it works**: Uses Python's Protocol for structural typing (duck typing)
- **Relationships**: Inbound ports used by application layer, outbound ports implemented by adapters

### Storage Adapters (`adapters/outbound/`)

Concrete implementations of outbound port interfaces.

- **Purpose**: Provide file-based persistence for pages and WAL segments
- **Key Files**:
  - `file_disk_manager.py` - File-based page I/O with header page tracking allocations
  - `lru_buffer_pool.py` - LRU eviction buffer pool with pin counting and dirty tracking
  - `file_wal_writer.py` - Segmented WAL files with CRC32 checksums and recovery support
- **How it works**:
  - **FileDiskManager**: Single data file with page 0 as header containing page count and free list
  - **LRUBufferPool**: Fixed-size frame array with OrderedDict for LRU tracking, per-frame latches
  - **FileWALWriter**: Segment files (wal_XXXXXXXX.log) with 32-byte headers, record wrappers with length + CRC
- **Relationships**: Implements DiskManager, BufferPool protocols; used by Transaction Manager and Recovery Service

## Quick Start

```bash
# Using Conda
conda env create -f conda.yaml
conda activate db_engine
pip install -e ".[dev]"

# Run tests
make test
```

## Implementation Status

### Implemented

- [x] **Value Objects** - Type-safe identifiers (PageId, TransactionId, LSN, RecordId)
- [x] **Transaction Types** - State machine, isolation levels, lock modes
- [x] **Slotted Page** - Variable-length record storage with checksums
- [x] **MVCC Records** - Record entity with visibility metadata
- [x] **WAL Record Types** - All ARIES log record types
- [x] **Infrastructure** - Config, logging, metrics, tracing, DI container
- [x] **Port Interfaces** - BufferPool, WALManager, TransactionManager, IndexManager, DiskManager, WALWriter protocols
- [x] **Disk Manager Adapter** - File-based page I/O with allocation/deallocation
- [x] **Buffer Pool Adapter** - LRU eviction, pin/unpin, dirty tracking, flush
- [x] **WAL Writer Adapter** - Segment management, CRC checksums, recovery support
- [x] **Recovery Service** - ARIES (Analysis, Redo, Undo)
- [x] **Transaction Manager** - MVCC snapshots, 2PL locking
- [x] **B+Tree Index** - Search, insert, delete, range scan
- [x] **SQL Parser** - Integration with sqlglot (LIMIT, OFFSET fixed)
- [x] **Query Executor** - Volcano-style iterators with projection, filtering, sorting, limiting
- [x] **Unit Tests** - **292 tests passing** (all domain, adapters, and executor tests)

### Planned

- [ ] **gRPC Server** - External API
- [ ] **Integration Tests** - Multi-component workflows
- [ ] **Chaos Tests** - Crash recovery validation
- [ ] **Property Tests** - Invariant verification

## Evaluation Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Crash recovery correctness | 100% pass rate | Planned |
| OLTP p95 latency | < hardware baseline | Planned |
| Deterministic recovery | Identical state across runs | Planned |

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
db_engine/
├── src/db_engine/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Page, Record, WAL records
│   │   ├── events/              # Domain events
│   │   ├── services/            # Recovery, visibility, locking
│   │   └── value_objects/       # PageId, LSN, TransactionId
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # BufferPool, TransactionManager
│   │   └── outbound/            # DiskManager, WALWriter
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # SQL parser, gRPC handler
│   │   └── outbound/            # File I/O, buffer pool
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # ExecuteSQL, BeginTxn
│   │   └── queries/             # GetSchema, ExplainPlan
│   └── infrastructure/          # Config, logging, metrics, tracing
├── tests/
│   ├── unit/                    # Isolated component tests
│   ├── integration/             # Component interaction tests
│   ├── benchmarks/              # Performance tests
│   ├── chaos/                   # Crash recovery tests
│   ├── property/                # Property-based tests
│   └── security/                # Security tests
└── docs/
    └── design.md                # Detailed design document
```

## Documentation

- [Design Document](docs/design.md) - Architecture, algorithms, and academic references

## References

1. **ARIES**: Mohan, C. et al. "ARIES: A Transaction Recovery Method" (1992)
2. **MVCC**: Reed, D. "Naming and Synchronization" MIT PhD Thesis (1978)
3. **B+Tree**: Bayer, R. & McCreight, E. "Organization and Maintenance of Large Ordered Indices" (1972)

## License

MIT
