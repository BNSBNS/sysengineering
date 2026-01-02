# Project E: Distributed Streaming System

A replicated, partitioned log system with Raft consensus, durability guarantees, and consumer group semantics (Kafka-like).

## Goal

Build a replicated, partitioned log with durability and consumer semantics.

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Data loss after ack | 0% (never) |
| Leader election time | < 10 seconds |
| Consumer offsets | No duplicates/losses |

## Quick Start

```bash
conda env create -f conda.yaml
conda activate streaming_system
pip install -e ".[dev]"
make test
```

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
streaming_system/
├── src/streaming_system/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Partition, Record, ConsumerGroup entities
│   │   ├── events/              # Domain events (RecordAppended, LeaderElected)
│   │   ├── services/            # Domain services (Raft, replication)
│   │   └── value_objects/       # Offset, TopicPartition, ConsumerID
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # Producer, Consumer, Admin protocols
│   │   └── outbound/            # LogStorage, RaftNetwork protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # Kafka protocol handlers
│   │   └── outbound/            # Segment files, network I/O
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # Produce, Consume, CreateTopic commands
│   │   └── queries/             # DescribeTopic, ListGroups queries
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
│                         Client Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Producers  │  │  Consumers  │  │    Admin API            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         Broker Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Partition   │  │   Raft      │  │    Consumer             │  │
│  │    Log      │  │  Consensus  │  │    Coordinator          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Replication Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    ISR      │  │  Controller │  │    Offset Manager       │  │
│  │  Tracking   │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Storage Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Segmented Log (append-only, indexed)            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
