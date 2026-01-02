# Secure Data Platform (Zero Trust) Design Document

## High-Level Component Overview

This document describes a Zero Trust security control plane implementing the principles established by NIST SP 800-207, Google's BeyondCorp, and the SPIFFE/SPIRE identity framework.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Secure Data Platform                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Identity   │───▶│ Authorization│───▶│    Audit     │                   │
│  │    (PKI)     │    │   (Policy)   │    │   (Merkle)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Cryptographic Layer                            │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │   X.509    │  │   mTLS     │  │   HMAC     │  │  SHA-256   │  │       │
│  │  │   Certs    │  │  Channels  │  │  Signing   │  │  Hashing   │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **PKI (X.509)** | Identity verification | Industry standard for machine identity. Every TLS connection uses X.509 [1] | RFC 5280 |
| **mTLS** | Mutual authentication | Both parties verify identity. Zero Trust mandates "never trust, always verify" [2] | NIST SP 800-207 |
| **SPIFFE IDs** | Workload identity | Standard format for service identity. Used by Istio, Envoy, HashiCorp [3] | CNCF SPIFFE spec |
| **RBAC/ABAC** | Access control | Role-based for simplicity, attribute-based for fine-grained control [4] | NIST ABAC Guide |
| **Merkle Trees** | Tamper evidence | Cryptographic proof of log integrity. Used by Certificate Transparency [5] | Merkle 1979 |
| **Casbin** | Policy engine | Flexible PERM model supports RBAC, ABAC, ACL in one framework [6] | Open source |

### How Zero Trust Works

**Traditional Model** (perimeter-based):
```
[Internet] → [Firewall] → [Trusted Internal Network]
                              ↓
                         All traffic trusted
```

**Zero Trust Model** (identity-based):
```
[Service A] ←─mTLS─→ [Policy Engine] ←─mTLS─→ [Service B]
                          ↓
                    Every request verified:
                    1. Identity (who?)
                    2. Authorization (allowed?)
                    3. Audit (logged?)
```

### References

1. **X.509**: Cooper, D. et al. "Internet X.509 Public Key Infrastructure Certificate and CRL Profile" RFC 5280 (2008)
2. **Zero Trust**: Rose, S. et al. "Zero Trust Architecture" NIST SP 800-207 (2020)
3. **SPIFFE**: CNCF. "Secure Production Identity Framework for Everyone" spiffe.io
4. **ABAC**: Hu, V. et al. "Guide to Attribute Based Access Control" NIST SP 800-162 (2014)
5. **Merkle Trees**: Merkle, R. "A Certified Digital Signature" CRYPTO (1989)
6. **Casbin**: casbin.org - Open source authorization library

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a secure control plane implementing Zero Trust architecture with identity management, policy-based authorization, and tamper-evident audit logs.

**Why build this?** Understanding security architecture is essential for:
- Designing secure microservices (service mesh, API gateways)
- Implementing compliance requirements (SOC2, PCI-DSS audit trails)
- Debugging authentication/authorization failures in production

### Goals

- Implement PKI for certificate issuance and rotation (RTO < 30s)
- mTLS for all service-to-service communication
- RBAC and ABAC policy evaluation (< 1ms p99)
- Merkle tree-based tamper-evident audit logs

### Non-Goals

- External identity provider integration (OIDC, SAML)
- Hardware security module (HSM) support
- Multi-region deployment
- Key escrow or recovery

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer (mTLS)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ gRPC Server │  │ REST API    │  │    Prometheus Metrics       │  │
│  │   (mTLS)    │  │  (mTLS)     │  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Identity Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │     PKI     │  │   SPIFFE    │  │    Certificate Rotation     │  │
│  │  (X.509)    │  │     IDs     │  │       (Auto)                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Authorization Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │    RBAC     │  │    ABAC     │  │    Policy Engine            │  │
│  │   (Roles)   │  │ (Attributes)│  │    (Casbin)                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Audit Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Merkle Tree Audit Log (Append-Only)                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Authentication Flow**: Service A → Present Cert → PKI Verify → Extract SPIFFE ID
2. **Authorization Flow**: SPIFFE ID + Action + Resource → Policy Engine → Allow/Deny
3. **Audit Flow**: All decisions → Merkle Tree → Tamper-evident log

---

## 3. Core Components & APIs

### PKI Manager

**Why PKI?** Public Key Infrastructure provides:
- **Authentication**: Certificates prove identity cryptographically
- **Confidentiality**: TLS encryption using certificate public keys
- **Non-repudiation**: Signed certificates can't be forged

**How certificates work**:
```
CA (Certificate Authority)
    │
    │ signs
    ▼
Service Certificate
    ├── Subject: spiffe://cluster/service-a
    ├── Public Key: [RSA/EC public key]
    ├── Valid: 2024-01-01 to 2024-01-08 (short-lived)
    └── Signature: [CA's signature]
```

```python
class PKIManager(Protocol):
    def issue_certificate(self, csr: CSR) -> Certificate: ...
    def revoke_certificate(self, serial: str) -> None: ...
    def rotate_certificate(self, cert: Certificate) -> Certificate: ...
    def verify_certificate(self, cert: Certificate) -> bool: ...
```

### Policy Engine

**Why RBAC + ABAC?** Combined approach provides:
- RBAC: Simple role assignments (admin, reader, writer)
- ABAC: Fine-grained rules (user.department == resource.owner)

**How policy evaluation works**:
```
Request: {subject: "service-a", action: "read", resource: "database-x"}
    │
    ▼
Policy Engine (Casbin)
    │
    ├── Check RBAC: service-a has role "reader"? YES
    │
    ├── Check ABAC: service-a.environment == "production"? YES
    │
    └── Result: ALLOW (logged to audit)
```

```python
class PolicyEngine(Protocol):
    def evaluate(self, subject: Principal, action: str, resource: Resource) -> Decision: ...
    def add_policy(self, policy: Policy) -> None: ...
    def remove_policy(self, policy_id: str) -> None: ...
```

### Audit Logger

**Why Merkle trees?** Provides:
- **Tamper evidence**: Any modification changes the root hash
- **Efficient verification**: O(log n) proof for any entry
- **Append-only**: Historical entries can't be modified

**How Merkle proofs work**:
```
To prove Event2 is in the log:
                    Root
                   /    \
              Hash01    Hash23  ← Include this
             /    \    /    \
          H0    H1   H2    H3
                     |
                  Event2  ← Prove this

Proof = [H3, Hash01]  (sibling hashes on path to root)
Verifier: hash(hash(H2, H3), Hash01) == Root? ✓
```

```python
class AuditLogger(Protocol):
    def log(self, event: AuditEvent) -> None: ...
    def verify_integrity(self) -> bool: ...
    def get_proof(self, event_id: str) -> MerkleProof: ...
```

---

## 4. Data Models & State Machines

### Certificate Lifecycle

**Why short-lived certificates?** Reduces blast radius of compromise. Industry trend: Netflix uses 4-hour certs, Google uses hours-to-days.

```
    ┌─────────────┐   issue()    ┌─────────────┐
    │   PENDING   │ ───────────▶ │    VALID    │
    └─────────────┘              └─────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
               rotate()            expire()           revoke()
                    │                  │                  │
                    ▼                  ▼                  ▼
            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
            │  ROTATING   │    │   EXPIRED   │    │   REVOKED   │
            └─────────────┘    └─────────────┘    └─────────────┘
                    │
                    ▼
            ┌─────────────┐
            │    VALID    │ (new cert)
            └─────────────┘
```

### Merkle Tree Structure

```
                    Root Hash (published)
                    /                   \
                Hash01                  Hash23
               /      \                /      \
           Hash0      Hash1        Hash2      Hash3
             |          |            |          |
          Event0    Event1       Event2     Event3
           (t=0)     (t=1)        (t=2)      (t=3)

Append Event4:
- Compute Hash4 = SHA256(Event4)
- Compute Hash45 = SHA256(Hash4 || Hash5)  # Hash5 from next event
- Update root: Root' = SHA256(Root || Hash45)
```

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| gRPC Server | asyncio | High concurrency TLS handshakes |
| Cert Rotation | Background thread | Non-blocking, periodic task |
| Policy Evaluation | Thread-safe cache | Cached policies, < 1ms latency |
| Audit Logging | Lock-free append | High throughput, never blocks |

### Certificate Rotation Flow

```python
async def rotate_certificates():
    """Background task: rotate certificates before expiry."""
    while True:
        for cert in get_expiring_certificates(threshold=timedelta(hours=1)):
            # 1. Generate new key pair (async CPU-bound)
            new_key = await generate_key_pair()

            # 2. Create CSR with same identity
            csr = create_csr(cert.subject, new_key)

            # 3. Request new cert from CA
            new_cert = await pki.issue_certificate(csr)

            # 4. Atomic swap (both certs valid briefly)
            await cert_store.swap(cert.serial, new_cert)

            # 5. Audit log
            audit.log(CertRotated(old=cert.serial, new=new_cert.serial))

        await asyncio.sleep(60)  # Check every minute
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Impact |
|---------|-----------|----------|--------|
| Cert expiration | Monitoring (cert_expiry_seconds) | Auto-rotation | Service disruption if missed |
| CA key compromise | Out-of-band detection | Manual re-key all certs | Critical: revoke everything |
| Audit corruption | Merkle verification failure | Restore from backup | Compliance violation |
| Policy sync failure | Health check timeout | Retry with exponential backoff | Fail-closed (deny all) |

### CA Key Compromise Response

```
1. IMMEDIATE: Revoke CA certificate (CRL/OCSP)
2. Generate new CA key pair (offline ceremony)
3. Re-issue all service certificates
4. Update trust stores in all services
5. Audit: Log incident with new root hash

Timeline: < 30 minutes (RTO target)
```

---

## 7. Security Threat Model

### STRIDE Analysis

| Threat | Asset | Mitigation | Implementation |
|--------|-------|------------|----------------|
| **S**poofing | Service identity | mTLS with certificate pinning | Verify cert chain to trusted CA |
| **T**ampering | Audit logs | Merkle tree with published roots | Hash chain prevents modification |
| **R**epudiation | Actions | Cryptographic signatures | Every action signed with service key |
| **I**nformation Disclosure | Secrets | TLS 1.3, no plaintext | Forward secrecy, strong ciphers |
| **D**enial of Service | Availability | Rate limiting, quotas | Per-service request limits |
| **E**levation of Privilege | Access | Least privilege ABAC | Default deny, explicit allow |

### mTLS Configuration

```yaml
# Secure TLS configuration
tls:
  min_version: TLS_1_3
  cipher_suites:
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
  client_auth: require  # mTLS: both sides present certs
  verify_depth: 2       # Service cert → Intermediate CA → Root CA
```

---

## 8. Performance Targets

| Metric | Target | How Achieved |
|--------|--------|--------------|
| Cert rotation RTO | < 30 seconds | Pre-generated keys, async issuance |
| Policy eval latency | < 1ms p99 | In-memory policy cache, efficient matching |
| Audit write latency | < 5ms p99 | Append-only, batched disk writes |
| mTLS handshake | < 50ms | Session resumption, TLS 1.3 |

### Latency Breakdown

```
mTLS Handshake (first connection):
├── TCP connect:      5ms
├── TLS ClientHello:  5ms
├── Cert exchange:   20ms  (2 certs, signature verify)
├── Key derivation:   5ms
└── Application:      5ms
                     ────
                     40ms typical

With session resumption: 15ms (skip cert exchange)
```

---

## 9. Operational Concerns

### Certificate Rotation

```bash
# Manual rotation trigger (for emergency rotation)
secure-platform rotate-cert --subject "spiffe://cluster/service-a"

# Check rotation status
secure-platform cert-status --watch

# List expiring certificates
secure-platform certs list --expiring-in 24h
```

### Audit Verification

```bash
# Verify audit log integrity (daily job)
secure-platform audit verify --from 2024-01-01 --to 2024-12-31

# Export cryptographic proof for compliance
secure-platform audit export-proof --event-id <id> --output proof.json

# Publish Merkle root (for external verification)
secure-platform audit publish-root --to https://transparency.example.com
```

### Policy Management

```bash
# Add policy rule
secure-platform policy add --subject "role:admin" --action "*" --resource "*"

# Test policy (dry-run)
secure-platform policy test --subject "service-a" --action "read" --resource "db-x"

# Export policies for review
secure-platform policy export --format yaml > policies.yaml
```

---

## 10. Alternatives Considered

### Identity Provider

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Custom PKI** | Simple, educational, no deps | Limited features | **Selected** for learning |
| HashiCorp Vault | Production-ready, HSM support | External dependency | Production alternative |
| SPIRE | Standard SPIFFE, Kubernetes-native | Complexity | Future enhancement |

**Source**: HashiCorp Vault is used by Stripe, Adobe. SPIRE is CNCF graduated project.

### Policy Engine

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Casbin** | Flexible PERM, Python native | Performance at scale | **Selected** |
| Open Policy Agent | Industry standard, Rego | Rego learning curve | Production alternative |
| Custom rules | Tailored to needs | Maintenance burden | Rejected |

**Source**: OPA is used by Netflix, Goldman Sachs for policy enforcement.

### Audit Log

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Merkle Tree** | Tamper-evident, verifiable | Complexity | **Selected** |
| Append-only log | Simple | No integrity proof | Insufficient |
| Blockchain | Distributed consensus | Overkill for single-org | Rejected |

**Source**: Certificate Transparency (RFC 6962) uses Merkle trees. Used by all CAs.

---

## Further Reading

1. **Zero Trust**: NIST SP 800-207 "Zero Trust Architecture" (2020)
2. **BeyondCorp**: Ward, R. & Beyer, B. "BeyondCorp: A New Approach to Enterprise Security" ;login: (2014)
3. **SPIFFE**: spiffe.io - Secure Production Identity Framework for Everyone
4. **Certificate Transparency**: RFC 6962, certificate-transparency.org
5. **Merkle Trees**: Crosby, S. & Wallach, D. "Efficient Data Structures for Tamper-Evident Logging" USENIX Security (2009)
6. **TLS 1.3**: RFC 8446 "The Transport Layer Security Protocol Version 1.3"
