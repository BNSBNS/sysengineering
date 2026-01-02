# Project D: GPU-Aware ML Platform

A data-center-level GPU scheduling platform with NUMA-aware placement, fault handling, and MPS/MIG GPU sharing support.

## Goal

Demonstrate data-center-level GPU scheduling, accounting, and fault handling.

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| GPU oversubscription | 0% (never) |
| NUMA-aware gains | Measurable improvement |
| Job recovery | 100% correctness |

## Quick Start

```bash
conda env create -f conda.yaml
conda activate gpu_platform
pip install -e ".[dev]"
make test
```

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
gpu_platform/
├── src/gpu_platform/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # GPU, Job, Allocation entities
│   │   ├── events/              # Domain events (JobScheduled, GPUFailed)
│   │   ├── services/            # Domain services (placement algorithms)
│   │   └── value_objects/       # NUMA topology, GPU capabilities
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # GPUScheduler, JobManager protocols
│   │   └── outbound/            # GPUDiscovery, HealthMonitor protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # gRPC/REST API handlers
│   │   └── outbound/            # NVML bindings, health checks
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # SubmitJob, AllocateGPU commands
│   │   └── queries/             # ListGPUs, GetJobStatus queries
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
│  │   Submit    │  │   Allocate  │  │    Monitor              │  │
│  │    Job      │  │    GPU      │  │    Health               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Domain Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ NUMA-Aware  │  │    Gang     │  │    Fault                │  │
│  │  Placement  │  │  Scheduler  │  │    Handler              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Adapters Layer                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              NVML · hwloc · cgroups                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
