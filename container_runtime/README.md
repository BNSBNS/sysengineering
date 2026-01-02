# Project B: Container Runtime & Scheduler

A production-grade container runtime implementing process isolation and deterministic scheduling with CPU, memory, and GPU awareness.

## Goal

Implement process isolation and deterministic scheduling with CPU, memory, and GPU awareness.

## Key Requirements

* Linux namespaces & cgroups
* OCI image handling
* Deterministic job placement
* GPU discovery and isolation

## Evaluation Metrics

* Resource isolation correctness
* Scheduler determinism under load
* GPU allocation accuracy

## Mandatory Tests

* Isolation tests
* Resource exhaustion tests
* Scheduler replay tests

## Overview

- **Namespaces**: Linux namespace isolation (PID, NET, MNT, UTS, IPC, USER)
- **Cgroups v2**: Resource limits and accounting
- **OCI Images**: Pull, unpack, and manage container images
- **Scheduler**: Deterministic job placement with bin-packing
- **GPU Support**: NVIDIA GPU discovery and allocation

## Quick Start

```bash
conda env create -f conda.yaml
conda activate container_runtime
pip install -e ".[dev]"

# Run tests
make test
```

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Resource isolation | 100% correctness |
| Scheduler determinism | Same input -> same placement |
| GPU allocation | No oversubscription |

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
container_runtime/
├── src/container_runtime/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Container, Job, Resource entities
│   │   ├── events/              # Domain events (ContainerStarted, JobScheduled)
│   │   ├── services/            # Domain services (scheduling algorithms)
│   │   └── value_objects/       # ResourceLimits, Namespace, Capability
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # ContainerManager, Scheduler protocols
│   │   └── outbound/            # CgroupsManager, ImageRegistry protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # gRPC/REST API handlers
│   │   └── outbound/            # Linux syscalls, OCI registry client
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # CreateContainer, ScheduleJob commands
│   │   └── queries/             # ListContainers, GetJobStatus queries
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
│                        API Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  gRPC API   │  │  REST API   │  │    Prometheus /metrics  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Application Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Create    │  │   Schedule  │  │    Monitor              │  │
│  │  Container  │  │    Job      │  │   Resources             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Domain Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Container  │  │  Bin-Pack   │  │    Resource             │  │
│  │  Lifecycle  │  │  Scheduler  │  │    Manager              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Adapters Layer                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │          namespaces · cgroups v2 · seccomp · OCI            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
