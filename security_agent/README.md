# Project F: Runtime Security Agent

An eBPF-based runtime security monitor with detection rules, event correlation, and automated response capabilities (EDR-lite).

## Goal

Implement an eBPF-based runtime security monitor with rules and response.

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| CPU overhead | < 3% |
| False positive rate | < 5% |
| Detection latency | < 100ms |

## Quick Start

```bash
# Requires Linux with eBPF support (kernel 5.8+)
conda env create -f conda.yaml
conda activate security_agent
pip install -e ".[dev]"
make test
```

## Requirements

- Linux kernel 5.8+ with eBPF support
- BCC (BPF Compiler Collection) installed
- Root/CAP_BPF privileges

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
security_agent/
├── src/security_agent/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Event, Detection, Rule entities
│   │   ├── events/              # Domain events (ThreatDetected, ResponseExecuted)
│   │   ├── services/            # Domain services (detection engine, baseline)
│   │   └── value_objects/       # Syscall, ProcessInfo, NetworkConnection
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # DetectionEngine, ResponseEngine protocols
│   │   └── outbound/            # ProbeManager, SIEMConnector protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # gRPC/REST API handlers
│   │   └── outbound/            # eBPF probes, SIEM integration
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # LoadRule, ExecuteResponse commands
│   │   └── queries/             # GetEvents, GetBaseline queries
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
│                          User Space                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Detection  │  │  Response   │  │    Alert Manager        │  │
│  │   Engine    │  │   Engine    │  │    (SIEM)               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Event Processing                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Rules    │  │  Baseline   │  │    ML Anomaly           │  │
│  │   (Sigma)   │  │  Learning   │  │    Detection            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                          eBPF Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Syscall    │  │  Network    │  │    File                 │  │
│  │  Probes     │  │  Probes     │  │    Probes               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         Kernel Space                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                       Linux Kernel                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
