# Container Runtime & Scheduler Design Document

## High-Level Component Overview

This document describes a container runtime implementing process isolation using Linux kernel primitives. The system is designed following the principles established by Docker, containerd, and the Open Container Initiative (OCI).

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Container Runtime                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   API Layer  │───▶│  Scheduler   │───▶│  Container   │                   │
│  │   (gRPC)     │    │  (Bin-pack)  │    │   Manager    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Linux Kernel Interfaces                        │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │ Namespaces │  │  Cgroups   │  │  Seccomp   │  │   NVML     │  │       │
│  │  │  (clone)   │  │    v2      │  │  (BPF)     │  │   (GPU)    │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach |
|-----------|---------|-------------------|
| **Namespaces** | Process isolation | Linux namespaces provide kernel-level isolation without hypervisor overhead. Used by Docker, LXC, and all modern container runtimes [1] |
| **Cgroups v2** | Resource limits | Unified cgroup hierarchy simplifies resource management. Mandatory for systemd-based systems since 2020 [2] |
| **Bin-packing Scheduler** | Efficient placement | Minimizes resource fragmentation. Google Borg uses similar algorithms for datacenter efficiency [3] |
| **OCI Images** | Portable containers | Industry standard ensures interoperability with Docker Hub, registries [4] |
| **gRPC API** | High-performance RPC | Protocol buffers provide schema evolution; used by Kubernetes, containerd [5] |

### References

1. **Namespaces**: Kerrisk, M. "Namespaces in operation" LWN.net (2013). Linux kernel documentation: `Documentation/namespaces/`
2. **Cgroups v2**: Heo, T. "Control Group v2" kernel.org (2015). Adopted by systemd, Docker 20.10+, Kubernetes 1.25+
3. **Bin-packing**: Verma, A. et al. "Large-scale cluster management at Google with Borg" EuroSys (2015)
4. **OCI Specification**: Open Container Initiative. "Runtime Specification v1.0" opencontainers.org (2017)
5. **gRPC**: Google. "gRPC: A high-performance RPC framework" grpc.io

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a container runtime implementing process isolation using Linux namespaces and cgroups, with deterministic scheduling supporting CPU, memory, and GPU awareness.

**Why build this?** Understanding container internals is essential for:
- Debugging production container issues (namespace leaks, cgroup limits)
- Designing secure multi-tenant systems
- Optimizing resource utilization in GPU clusters

### Goals

- Implement Linux namespace isolation (PID, NET, MNT, UTS, IPC, USER)
- Manage resources via cgroups v2
- Pull and unpack OCI-compliant container images
- Provide deterministic job placement with bin-packing
- Support NVIDIA GPU discovery and allocation

### Non-Goals

- Full OCI runtime compliance (subset only)
- Kubernetes CRI compatibility (simplified API)
- Network plugins (basic bridge networking only)
- Storage drivers beyond overlayfs
- Windows container support

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  gRPC API   │  │   CLI       │  │    Prometheus Metrics       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Scheduler Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Job Queue  │  │  Placement  │  │    Resource Tracker         │  │
│  │             │  │  Algorithm  │  │    (CPU/Mem/GPU)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Container Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Lifecycle  │  │   Image     │  │    Exec                     │  │
│  │  Manager    │  │   Manager   │  │    Handler                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Linux Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Namespaces  │  │  Cgroups    │  │    Network                  │  │
│  │ (unshare)   │  │    v2       │  │    (veth/bridge)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         GPU Layer                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              NVIDIA Management (nvml)                           ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Request Flow**: Client → gRPC API → Scheduler → Container Manager → Linux Primitives
2. **Resource Flow**: Container Manager → Cgroup Manager → Kernel cgroup filesystem
3. **Network Flow**: Container Manager → Network Manager → veth pairs → Bridge

---

## 3. Core Components & APIs

### Container Lifecycle Manager

**Why Protocol-based interfaces?** Dependency injection enables testing with mocks and swapping implementations (e.g., different cgroup backends).

```python
class ContainerManager(Protocol):
    def create(self, config: ContainerConfig) -> Container: ...
    def start(self, container_id: str) -> None: ...
    def stop(self, container_id: str, timeout: int = 10) -> None: ...
    def delete(self, container_id: str) -> None: ...
    def exec(self, container_id: str, command: list[str]) -> ExecResult: ...
    def list(self, filters: dict | None = None) -> list[Container]: ...
```

### Scheduler

**Why single-threaded scheduler?** Determinism is critical for debugging and replay. Kubernetes scheduler follows similar design for consistency.

```python
class Scheduler(Protocol):
    def submit(self, job: Job) -> JobId: ...
    def cancel(self, job_id: JobId) -> None: ...
    def get_status(self, job_id: JobId) -> JobStatus: ...
    def place(self, job: Job) -> Placement: ...
```

### Namespace Manager

**How it works**: Uses `clone(2)` with `CLONE_NEW*` flags or `unshare(2)` to create isolated namespaces. Each namespace type isolates a specific kernel resource:

| Namespace | Flag | Isolates |
|-----------|------|----------|
| PID | `CLONE_NEWPID` | Process IDs |
| NET | `CLONE_NEWNET` | Network stack |
| MNT | `CLONE_NEWNS` | Mount points |
| UTS | `CLONE_NEWUTS` | Hostname |
| IPC | `CLONE_NEWIPC` | IPC objects |
| USER | `CLONE_NEWUSER` | UID/GID mapping |

```python
class NamespaceManager(Protocol):
    def create_namespaces(self, config: NamespaceConfig) -> NamespaceSet: ...
    def enter_namespace(self, ns: Namespace) -> None: ...
    def cleanup(self, container_id: str) -> None: ...
```

### Cgroup Manager

**How cgroups v2 works**: Single unified hierarchy mounted at `/sys/fs/cgroup`. Each container gets a subdirectory with controller files:

```
/sys/fs/cgroup/
└── containers/
    └── <container-id>/
        ├── cgroup.controllers    # Available controllers
        ├── cpu.max               # CPU quota (quota period)
        ├── memory.max            # Memory limit in bytes
        ├── memory.current        # Current memory usage
        └── pids.max              # Process limit
```

```python
class CgroupManager(Protocol):
    def create_cgroup(self, container_id: str, limits: ResourceLimits) -> Cgroup: ...
    def update_limits(self, cgroup: Cgroup, limits: ResourceLimits) -> None: ...
    def get_stats(self, cgroup: Cgroup) -> ResourceStats: ...
    def cleanup(self, cgroup: Cgroup) -> None: ...
```

---

## 4. Data Models & State Machines

### Container State Machine

**Why this state model?** Matches OCI runtime lifecycle, enabling compatibility with standard tooling.

```
                    create()
    ┌─────────┐ ──────────────▶ ┌─────────────┐
    │  None   │                 │  CREATED    │
    └─────────┘                 └─────────────┘
                                      │
                                 start()
                                      │
                                      ▼
                                ┌─────────────┐
                         ┌──────│   RUNNING   │──────┐
                         │      └─────────────┘      │
                    stop()           │          OOM/crash
                         │        exit()             │
                         ▼           │               ▼
                   ┌─────────────┐   │         ┌─────────────┐
                   │   STOPPED   │◀──┘         │   FAILED    │
                   └─────────────┘             └─────────────┘
                         │                           │
                    delete()                    delete()
                         │                           │
                         ▼                           ▼
                   ┌─────────────┐             ┌─────────────┐
                   │   DELETED   │             │   DELETED   │
                   └─────────────┘             └─────────────┘
```

### Resource Limits Model

```python
@dataclass
class ResourceLimits:
    cpu_shares: int = 1024           # Relative CPU weight (default: 1024)
    cpu_quota: int | None = None     # Microseconds per period
    cpu_period: int = 100000         # Period in microseconds (100ms)
    memory_limit: int | None = None  # Bytes, None = unlimited
    memory_swap: int | None = None   # Bytes, None = same as memory
    pids_limit: int | None = None    # Max processes
    gpu_ids: list[str] | None = None # GPU UUIDs to expose
    gpu_memory: int | None = None    # GPU memory limit per device
```

**How CPU limits work**:
- `cpu_quota / cpu_period` = fraction of CPU (e.g., 50000/100000 = 50% of one core)
- `cpu_shares` = relative weight when competing for CPU

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| API Server | asyncio | High concurrency, I/O-bound. Python's asyncio handles 10K+ concurrent connections efficiently |
| Container Init | multiprocessing (fork) | Isolation requires new process. `clone()` creates child in new namespaces |
| Image Pulls | asyncio + httpx | Concurrent layer downloads reduce pull time by parallelizing HTTP requests |
| Resource Monitoring | Threading | Periodic polling of cgroup stats files. Threading avoids blocking event loop |
| Scheduler | Single-threaded | **Critical**: Determinism enables debugging via replay |

### Container Creation Flow

```python
async def create_container(config: ContainerConfig) -> Container:
    # 1. Validate config (async)
    # Why: Early validation prevents wasted resources
    await validate_config(config)

    # 2. Pull image if needed (async I/O)
    # Why: Parallel layer downloads, registry auth
    await ensure_image(config.image)

    # 3. Fork container init process (multiprocessing)
    # Why: clone() with CLONE_NEW* flags requires new process
    process = await run_in_executor(fork_container, config)

    # 4. Setup cgroups (sync, fast)
    # Why: Must happen before container starts to enforce limits
    cgroup = cgroup_manager.create_cgroup(container_id, config.limits)

    # 5. Setup network (sync)
    # Why: veth pair must exist before container networking works
    network = network_manager.setup(container_id, config.network)

    return Container(id=container_id, process=process, cgroup=cgroup)
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Source |
|---------|-----------|----------|--------|
| Container crash | SIGCHLD / waitpid | Update state, emit event | POSIX signal handling |
| OOM kill | `memory.events` file | Mark failed, emit OOM event | cgroups v2 spec |
| Image pull failure | HTTP error | Retry with exponential backoff | Standard retry pattern |
| Namespace creation fail | syscall error (EPERM) | Cleanup partial state | Linux namespace semantics |
| GPU allocation fail | NVML error | Return to pool, retry | NVIDIA NVML API |
| Scheduler crash | Health check timeout | Replay from persistent queue | Event sourcing pattern |

### Cleanup on Failure

**Why reverse order?** Dependencies must be cleaned in reverse creation order. Network depends on namespaces, which depend on cgroups.

```python
def cleanup_container(container_id: str) -> None:
    """Cleanup in reverse order of creation."""
    # 1. Network (depends on namespace)
    try:
        network_manager.cleanup(container_id)
    except Exception as e:
        log.warning("Network cleanup failed", error=e)

    # 2. Cgroups (independent, but processes must be gone)
    try:
        cgroup_manager.cleanup(container_id)
    except Exception as e:
        log.warning("Cgroup cleanup failed", error=e)

    # 3. Namespaces (kernel cleans up when last process exits)
    try:
        namespace_manager.cleanup(container_id)
    except Exception as e:
        log.warning("Namespace cleanup failed", error=e)

    # 4. Filesystem (overlayfs mount)
    try:
        filesystem_manager.cleanup(container_id)
    except Exception as e:
        log.warning("Filesystem cleanup failed", error=e)
```

---

## 7. Security Threat Model

### STRIDE Analysis

**Why STRIDE?** Microsoft's threat modeling framework systematically covers major threat categories. Used by AWS, Google for security reviews.

| Threat | Asset | Mitigation |
|--------|-------|------------|
| **S**poofing | Container identity | Unique container IDs, namespace isolation |
| **T**ampering | Container filesystem | Read-only layers, signed images (Notary v2) |
| **R**epudiation | Operations | Audit logging with timestamps, immutable logs |
| **I**nformation Disclosure | Host filesystem | Mount namespace, seccomp filters |
| **D**enial of Service | Host resources | Cgroup limits, pids limit, OOM killer |
| **E**levation of Privilege | Root access | User namespaces, dropped capabilities |

### Security Controls

**Seccomp** (Secure Computing Mode):
- **Why**: Kernel-level syscall filtering reduces attack surface
- **How**: BPF program loaded via `seccomp(2)` syscall, filters syscalls by number

**Capabilities**:
- **Why**: Fine-grained privileges instead of root/non-root binary
- **How**: Drop all capabilities, add only required (e.g., `CAP_NET_BIND_SERVICE`)

**No New Privileges**:
- **Why**: Prevents setuid binaries from escalating privileges
- **How**: `PR_SET_NO_NEW_PRIVS` via `prctl(2)`

---

## 8. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Container startup (cold) | < 500ms | Competitive with Docker cold start |
| Container startup (warm) | < 100ms | Image cached, only namespace/cgroup setup |
| Image layer pull | > 50 MB/s | Saturate typical network bandwidth |
| Scheduling decision | < 10ms | Sub-frame latency for interactive workloads |
| Memory overhead per container | < 10 MB | Metadata only, no VM overhead |

### Resource Limits

| Resource | Default | Max | Why |
|----------|---------|-----|-----|
| Containers per host | - | 1000 | cgroup hierarchy depth limits |
| Memory per container | 512 MB | Host limit | Prevent OOM of other containers |
| CPU shares | 1024 | 262144 | cgroups v2 max |
| PIDs per container | 100 | 32768 | Prevent fork bombs |

---

## 9. Operational Concerns

### Deployment

```bash
# Requires root or CAP_SYS_ADMIN for namespace creation
# Production: use systemd unit with minimal capabilities
docker run --privileged -v /sys/fs/cgroup:/sys/fs/cgroup \
    container_runtime:latest
```

### Monitoring

Key metrics exposed via Prometheus:
- `container_count{state="running|stopped|failed"}` - Container state distribution
- `container_startup_seconds` - Startup latency histogram
- `resource_cpu_usage_percent{container_id}` - CPU utilization
- `resource_memory_bytes{container_id}` - Memory usage
- `oom_events_total` - OOM kill counter (alert threshold: >0)

### Debugging

```bash
# Inspect container namespaces
nsenter -t <pid> -a /bin/sh

# Check cgroup limits
cat /sys/fs/cgroup/containers/<id>/memory.max
cat /sys/fs/cgroup/containers/<id>/memory.current

# View container logs
journalctl -u container-runtime -f

# Debug namespace creation
strace -f -e clone,unshare container_runtime create test
```

---

## 10. Alternatives Considered & Tradeoffs

### Container Runtime Implementation

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Custom (Python)** | Educational, portable, easy debugging | Performance overhead | **Selected** for learning |
| runc (Go) | Production-ready, OCI compliant | External dependency | Production alternative |
| crun (C) | Fast, low memory | C complexity | Performance-critical use |

**Source**: runc is the reference OCI implementation used by Docker, containerd, CRI-O.

### Cgroup Version

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Cgroups v2** | Unified hierarchy, better accounting | Requires kernel 5.2+ | **Selected** |
| Cgroups v1 | Wide support | Complex, deprecated | Legacy only |

**Source**: Kubernetes 1.25+ defaults to cgroups v2. systemd requires v2 for unified resource management.

### Scheduling Algorithm

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Bin-packing** | Maximizes utilization, fewer nodes | Can create hotspots | **Selected** |
| Spread | Even distribution, fault tolerance | Lower utilization | Multi-AZ deployments |
| Random | Simple, fast | Unpredictable | Testing only |

**Source**: Google Borg paper (EuroSys 2015) demonstrates bin-packing effectiveness at scale.

### Network Mode

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Bridge** | Simple NAT, port mapping | Performance overhead (~5%) | **Selected** for isolation |
| Host | Native performance | No network isolation | Performance-critical |
| None | Maximum security | No connectivity | Air-gapped workloads |

---

## Further Reading

1. **Linux Containers**: Kerrisk, M. "Namespaces in operation" LWN.net series (2013)
2. **Cgroups v2**: kernel.org Documentation/admin-guide/cgroup-v2.rst
3. **OCI Runtime Spec**: github.com/opencontainers/runtime-spec
4. **Borg Paper**: Verma, A. et al. "Large-scale cluster management at Google with Borg" (2015)
5. **Seccomp BPF**: kernel.org Documentation/userspace-api/seccomp_filter.rst
6. **Docker Internals**: docker.com/blog/docker-networking-design-philosophy/
