# Project G: Filesystem / Object Store

A versioned, checksummed object store with content-addressable storage, deduplication, erasure coding, and S3-compatible API.

## Goal

Build a versioned, checksummed object store with replication.

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Checksum validation | 100% corruption detection |
| Version consistency | No orphaned versions |
| Erasure coding | Tolerate 2 shard failures |

## Quick Start

```bash
conda env create -f conda.yaml
conda activate object_store
pip install -e ".[dev]"
make test
```

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
object_store/
├── src/object_store/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Object, Chunk, Bucket entities
│   │   ├── events/              # Domain events (ObjectCreated, ChunkStored)
│   │   ├── services/            # Domain services (chunking, erasure coding)
│   │   └── value_objects/       # ChunkRef, MerkleProof, ETag
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # ObjectService, BucketService protocols
│   │   └── outbound/            # ChunkStore, MetadataDB protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # S3-compatible REST API
│   │   └── outbound/            # File storage, SQLite metadata
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # PutObject, DeleteObject commands
│   │   └── queries/             # GetObject, ListBucket queries
│   └── infrastructure/          # Cross-cutting concerns
│       ├── config.py            # Pydantic settings
│       ├── logging.py           # Structured logging (structlog)
│       ├── metrics.py           # Prometheus metrics
│       ├── tracing.py           # OpenTelemetry tracing
│       └── container.py         # Dependency injection
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── benchmarks/              # Performance benchmarks
│   ├── chaos/                   # Failure injection tests
│   ├── property/                # Property-based tests
│   └── security/                # Security tests
├── docs/
│   └── design.md                # Design document
├── .github/workflows/ci.yml     # CI pipeline
├── pyproject.toml               # Python packaging
├── Makefile                     # Build automation
├── Dockerfile                   # Container image
├── docker-compose.yml           # Local development
└── conda.yaml                   # Conda environment
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           API Layer                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    S3-Compatible REST API                    ││
│  │     PUT/GET/DELETE Object, Multipart, Bucket Operations     ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                        Processing Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Chunking   │  │ Dedup       │  │    Integrity            │  │
│  │  Engine     │  │ Manager     │  │    Verifier             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Durability Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Erasure    │  │  Merkle     │  │    Placement            │  │
│  │  Coding     │  │  Tree       │  │    Engine               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Storage Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Content-Addressed Chunk Store                   ││
│  │                  (SHA-256 addressed)                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
