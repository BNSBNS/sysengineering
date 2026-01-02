# Project C: Secure Data Platform (Zero Trust)

A secure control plane implementing Zero Trust architecture with identity management, policy enforcement, and tamper-evident audit logs.

## Goal

Build a secure control plane with identity, policy enforcement, and immutable audit logs.

## Key Requirements

* mTLS everywhere
* Policy-based authorization
* Tamper-evident audit logs

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Cert rotation RTO | < 30 seconds |
| Policy eval latency | < 1ms p99 |
| Audit tamper detection | 100% correctness |

## Quick Start

```bash
conda env create -f conda.yaml
conda activate secure_platform
pip install -e ".[dev]"
make test
```

## Project Structure

This project follows **Hexagonal Architecture** (Ports & Adapters):

```
secure_platform/
├── src/secure_platform/
│   ├── domain/                  # Core business logic
│   │   ├── entities/            # Certificate, Policy, AuditEntry entities
│   │   ├── events/              # Domain events (CertIssued, PolicyEvaluated)
│   │   ├── services/            # Domain services (PKI, policy engine)
│   │   └── value_objects/       # SPIFFE ID, Permission, MerkleProof
│   ├── ports/                   # Interface definitions (ABCs)
│   │   ├── inbound/             # PKIManager, PolicyEngine protocols
│   │   └── outbound/            # CertStore, AuditLogger protocols
│   ├── adapters/                # Implementations
│   │   ├── inbound/             # gRPC/REST API handlers
│   │   └── outbound/            # X.509 crypto, Casbin, Merkle tree
│   ├── application/             # Use cases / orchestration
│   │   ├── commands/            # IssueCert, EvaluatePolicy commands
│   │   └── queries/             # GetCert, VerifyAudit queries
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
└── conda.yaml                   # Conda environment
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (mTLS)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  gRPC API   │  │  REST API   │  │    Prometheus /metrics  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Identity Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │     PKI     │  │   SPIFFE    │  │    Certificate          │  │
│  │   (X.509)   │  │     IDs     │  │    Rotation             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Authorization Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    RBAC     │  │    ABAC     │  │    Policy Engine        │  │
│  │   (Roles)   │  │ (Attributes)│  │    (Casbin)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Audit Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Merkle Tree Audit Log (Append-Only)             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Design Document](docs/design.md) - Architecture, components, and sources

## License

MIT
