# GPU-Aware ML Platform Design Document

## High-Level Component Overview

This document describes a GPU scheduling platform implementing datacenter-level resource management principles from Google's Borg, Microsoft's Gandiva, and NVIDIA's GPU virtualization technologies.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GPU Platform                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Scheduler  │───▶│  Allocator   │───▶│   Executor   │                   │
│  │  (NUMA-aware)│    │  (MPS/MIG)   │    │  (Isolation) │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Hardware Abstraction                           │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │       │
│  │  │   NVML     │  │   hwloc    │  │   CUDA     │  │  PCIe/     │  │       │
│  │  │  (Monitor) │  │ (Topology) │  │  (Compute) │  │  NVLink    │  │       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Each Component Exists

| Component | Purpose | Why This Approach | Source |
|-----------|---------|-------------------|--------|
| **NUMA-aware Placement** | Memory bandwidth | GPU-to-CPU affinity critical for data transfer. Wrong NUMA = 2-3x slowdown [1] | Intel/AMD NUMA docs |
| **Gang Scheduling** | Multi-GPU jobs | All GPUs must be allocated atomically or job fails [2] | Google Borg paper |
| **MPS/MIG Sharing** | GPU utilization | Single jobs rarely use 100% GPU. Sharing increases utilization 2-3x [3] | NVIDIA docs |
| **NVML Monitoring** | Health & metrics | Detect ECC errors, temperature, utilization before failure [4] | NVIDIA NVML API |
| **No Oversubscription** | Predictability | GPU memory oversubscription causes OOM kills, unlike CPU [5] | Industry practice |
| **XID Error Handling** | Fault tolerance | Hardware errors need immediate response to prevent cascade [6] | NVIDIA XID guide |

### How GPU Scheduling Differs from CPU

| Aspect | CPU Scheduling | GPU Scheduling |
|--------|----------------|----------------|
| **Oversubscription** | Common (virtual memory) | Never (GPU OOM = job death) |
| **Preemption** | Fast (context switch) | Slow (kernel must complete) |
| **Topology** | NUMA optional | NUMA critical (NVLink, PCIe) |
| **Sharing** | Time-slice default | Requires MPS/MIG setup |
| **Failures** | Software recovery | Hardware errors (ECC, XID) |

### References

1. **NUMA Effects**: Lameter, C. "NUMA (Non-Uniform Memory Access): An Overview" ACM Queue (2013)
2. **Gang Scheduling**: Feitelson, D. & Rudolph, L. "Gang Scheduling Performance Benefits for Fine-Grain Synchronization" JPDC (1992)
3. **GPU Sharing**: Xiao, W. et al. "Gandiva: Introspective Cluster Scheduling for Deep Learning" OSDI (2018)
4. **NVML**: NVIDIA. "NVIDIA Management Library (NVML)" developer.nvidia.com
5. **GPU Memory**: Yu, P. & Chowdhury, M. "Salus: Fine-Grained GPU Sharing Primitives for Deep Learning" MLSys (2020)
6. **XID Errors**: NVIDIA. "XID Errors" docs.nvidia.com/deploy/xid-errors

---

## 1. Problem Statement & Non-Goals

### Problem Statement

Build a GPU scheduling platform with NUMA-aware placement, fault handling, and support for MPS/MIG GPU sharing.

**Why build this?** GPU scheduling is fundamentally different from CPU:
- GPUs cannot be oversubscribed (no virtual memory equivalent)
- Topology matters: wrong NUMA placement = 50% performance loss
- Hardware failures are common: ECC errors, XID errors, thermal throttling
- Sharing requires explicit configuration (MPS or MIG)

### Goals

- GPU discovery and topology mapping via NVML
- NUMA-aware job placement for optimal memory bandwidth
- Gang scheduling for multi-GPU jobs
- ECC/XID error detection and job recovery
- No GPU oversubscription (100% allocation accuracy)

### Non-Goals

- Kubernetes device plugin compatibility
- AMD/Intel GPU support
- Distributed training orchestration
- Model serving runtime

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  gRPC API   │  │  REST API   │  │    Prometheus Metrics       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Scheduler Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ NUMA-Aware  │  │    Gang     │  │    Preemption               │  │
│  │  Placement  │  │  Scheduler  │  │    Manager                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Allocation Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  Allocator  │  │  MPS/MIG    │  │    Fragmentation            │  │
│  │             │  │   Sharing   │  │    Manager                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Discovery Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │   Scanner   │  │  Topology   │  │    Health Monitor           │  │
│  │   (NVML)    │  │   (hwloc)   │  │    (ECC/XID)                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Fault Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Detector → Handler → Recovery                      ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Discovery**: NVML scan → Topology mapping → Health baseline
2. **Scheduling**: Job request → NUMA-aware placement → Gang allocation
3. **Execution**: GPU assignment → Isolation (cgroups) → Monitoring
4. **Fault Handling**: XID detection → Job migration → GPU recovery

---

## 3. Core Components & APIs

### GPU Discovery

**Why topology matters**:
```
NUMA Node 0          NUMA Node 1
┌─────────────┐      ┌─────────────┐
│   CPU 0-15  │      │  CPU 16-31  │
│   128GB RAM │      │   128GB RAM │
└──────┬──────┘      └──────┬──────┘
       │ PCIe                │ PCIe
       ▼                     ▼
┌─────────────┐      ┌─────────────┐
│   GPU 0-1   │      │   GPU 2-3   │
│  (NVLinked) │      │  (NVLinked) │
└─────────────┘      └─────────────┘

Job using GPU 0 should run on CPU 0-15 for optimal DMA.
Job using GPUs 0+2 = cross-NUMA = slow data transfer.
```

```python
class GPUDiscovery(Protocol):
    def scan(self) -> list[GPUDevice]: ...
    def get_topology(self) -> NUMATopology: ...
    def get_health(self, gpu_id: str) -> GPUHealth: ...
```

### Scheduler

**Why gang scheduling?**
```
Multi-GPU training requires synchronized GPUs:

Without gang scheduling:
  t=0: GPU 0 allocated, waiting for GPU 1
  t=5: GPU 1 allocated, waiting for GPU 2
  t=10: GPU 0 times out, released
  Result: DEADLOCK or STARVATION

With gang scheduling:
  t=0: Request GPUs [0,1,2] atomically
  t=0: Either ALL allocated or NONE
  Result: No partial allocations
```

```python
class GPUScheduler(Protocol):
    def submit(self, job: Job) -> JobId: ...
    def place(self, job: Job) -> Placement: ...
    def preempt(self, job_id: JobId) -> None: ...
```

### Allocator

**MPS vs MIG comparison**:

| Feature | MPS (Multi-Process Service) | MIG (Multi-Instance GPU) |
|---------|----------------------------|--------------------------|
| Isolation | Soft (shared memory) | Hard (separate engines) |
| Overhead | Low | Higher |
| Granularity | Any % | Fixed slices (1/7, 2/7...) |
| Use case | Trusted workloads | Multi-tenant |
| GPU support | All CUDA GPUs | A100, H100+ |

```python
class GPUAllocator(Protocol):
    def allocate(self, request: GPURequest) -> Allocation: ...
    def release(self, allocation: Allocation) -> None: ...
    def get_available(self) -> list[GPUDevice]: ...
```

---

## 4. Data Models & State Machines

### Job State Machine

```
    ┌─────────────┐   submit()   ┌─────────────┐
    │   PENDING   │ ───────────▶ │   QUEUED    │
    └─────────────┘              └─────────────┘
                                       │
                                  schedule()
                                       │
                                       ▼
                                ┌─────────────┐
                         ┌──────│   RUNNING   │──────┐
                         │      └─────────────┘      │
                    complete()         │          fail()
                         │         preempt()         │
                         ▼             │             ▼
                   ┌─────────────┐     │       ┌─────────────┐
                   │  COMPLETED  │     │       │   FAILED    │
                   └─────────────┘     │       └─────────────┘
                                       ▼
                                ┌─────────────┐
                                │  PREEMPTED  │───▶ Re-queue
                                └─────────────┘
```

### GPU State Machine

```
    ┌─────────────┐
    │  AVAILABLE  │◀──────────────────────────┐
    └─────────────┘                           │
          │                                   │
     allocate()                          release()
          │                                   │
          ▼                                   │
    ┌─────────────┐                    ┌──────┴──────┐
    │  ALLOCATED  │───────────────────▶│   IN_USE    │
    └─────────────┘                    └─────────────┘
          │                                   │
     health_fail()                       xid_error()
          │                                   │
          ▼                                   ▼
    ┌─────────────┐                    ┌─────────────┐
    │  DEGRADED   │                    │  RESETTING  │
    └─────────────┘                    └─────────────┘
          │                                   │
     admin_action()                      reset_complete()
          │                                   │
          ▼                                   ▼
    ┌─────────────┐                    ┌─────────────┐
    │   OFFLINE   │                    │  AVAILABLE  │
    └─────────────┘                    └─────────────┘
```

---

## 5. Concurrency Model

| Component | Model | Rationale |
|-----------|-------|-----------|
| API Server | asyncio | High concurrency, non-blocking |
| Job Execution | multiprocessing | Process isolation per job |
| Health Polling | Threading | Periodic NVML queries (100ms) |
| Scheduler | Single-threaded | Determinism for debugging |

### NUMA-Aware Placement Algorithm

```python
def place_job(job: Job, cluster: Cluster) -> Placement | None:
    """
    NUMA-aware bin-packing placement.

    Priority:
    1. Same NUMA node (best: NVLink between GPUs)
    2. Same socket (good: PCIe within socket)
    3. Cross-socket (acceptable: QPI/UPI link)
    4. Cross-node (last resort: network)
    """
    required_gpus = job.gpu_count

    # Try placement within single NUMA node
    for node in cluster.numa_nodes:
        available = node.get_available_gpus()
        if len(available) >= required_gpus:
            # Prefer NVLink-connected GPUs
            gpus = select_nvlink_group(available, required_gpus)
            return Placement(gpus=gpus, numa_node=node.id)

    # Fall back to cross-NUMA (with warning)
    if job.allow_cross_numa:
        return cross_numa_placement(job, cluster)

    return None  # Cannot place, queue
```

---

## 6. Failure Modes & Recovery

| Failure | Detection | Recovery | Source |
|---------|-----------|----------|--------|
| ECC error (correctable) | NVML polling | Log, continue | XID 48 |
| ECC error (uncorrectable) | NVML event | Fail job, offline GPU | XID 48 |
| XID 31 (GPU memory error) | NVML event | Reset GPU, restart job | XID 31 |
| XID 43 (GPU stopped) | NVML event | Hard reset, fail jobs | XID 43 |
| Thermal throttle | Temperature > 85°C | Pause scheduling | Monitoring |
| GPU hang | Watchdog timeout | Force reset via nvidia-smi | Custom |

### XID Error Response Flow

```
XID Error Detected
        │
        ▼
┌───────────────────┐
│ Classify Severity │
│  - Critical: 31,43│
│  - Warning: 48    │
└───────────────────┘
        │
        ├── Critical ──▶ Kill jobs ──▶ Reset GPU ──▶ Health check
        │
        └── Warning ───▶ Log ──▶ Increment counter ──▶ Alert if > threshold
```

---

## 7. Security Threat Model

| Threat | Asset | Mitigation | Implementation |
|--------|-------|------------|----------------|
| GPU memory snooping | Training data | MIG isolation | Separate GPU engines |
| Resource exhaustion | Cluster capacity | Quotas, no oversub | Per-user GPU limits |
| Unauthorized access | GPU devices | cgroups device whitelist | Only assigned GPUs visible |
| Side-channel attacks | Model weights | Process isolation | Separate CUDA contexts |

### GPU Isolation with cgroups

```bash
# Restrict container to GPU 0 only
echo "c 195:0 rwm" > /sys/fs/cgroup/devices/container/devices.allow
# 195 = nvidia major device number
# 0 = GPU index

# With NVIDIA container runtime
docker run --gpus '"device=0"' ...
```

---

## 8. Performance Targets

| Metric | Target | How Measured |
|--------|--------|--------------|
| GPU discovery | < 1s | NVML scan time |
| Scheduling decision | < 100ms | Placement algorithm |
| Job startup | < 5s | Allocation to first CUDA call |
| Job recovery | < 30s | Failure to re-running |
| Utilization target | > 80% | GPU-hours used / available |
| Oversubscription | 0% | Allocated > physical = bug |

### Utilization Metrics

```
Cluster utilization = Σ(gpu_utilization × time) / Σ(total_gpu_hours)

Target breakdown:
- Training jobs: 90%+ GPU utilization (compute bound)
- Inference jobs: 30-50% utilization (latency bound)
- Interactive: 10-30% utilization (bursty)

Overall target: 80% average (industry benchmark)
```

---

## 9. Operational Concerns

### GPU Health Monitoring

```bash
# Show all GPUs with health status
gpu-platform status --all

# Detailed health for specific GPU
gpu-platform health --gpu-id GPU-UUID-1234

# Watch mode for debugging
gpu-platform watch --interval 1s
```

### Job Management

```bash
# List all jobs with GPU assignments
gpu-platform jobs list

# Show job details including NUMA placement
gpu-platform jobs describe <job-id>

# Cancel job (graceful shutdown)
gpu-platform jobs cancel <job-id>

# Force kill (immediate)
gpu-platform jobs kill <job-id>
```

### Maintenance Operations

```bash
# Drain GPU for maintenance (finish running, no new jobs)
gpu-platform drain --gpu-id GPU-UUID-1234

# Offline GPU immediately
gpu-platform offline --gpu-id GPU-UUID-1234 --force

# Return GPU to pool
gpu-platform online --gpu-id GPU-UUID-1234
```

---

## 10. Alternatives Considered

### GPU Management API

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **NVML (Python)** | Simple, direct, portable | Limited to monitoring | **Selected** |
| CUDA Driver API | Full control | C complexity | For advanced features |
| DCGM | Enterprise features | NVIDIA license | Production alternative |

**Source**: NVML is used by nvidia-smi, Prometheus exporters, Kubernetes device plugins.

### Scheduling Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **NUMA-aware bin-pack** | Optimal bandwidth | Complex | **Selected** |
| Random | Simple | Poor performance | Testing only |
| Round-robin | Fair | Ignores topology | Not for GPUs |

**Source**: Google's Borg and Microsoft's Gandiva both use topology-aware placement.

### GPU Sharing

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **MPS** | Works on all GPUs | Soft isolation | **Selected** for single-tenant |
| MIG | Hard isolation | A100+ only | For multi-tenant |
| Time-slicing | No setup | Context switch overhead | Fallback |

**Source**: MIG introduced in A100 (2020). MPS available since Kepler (2012).

---

## Further Reading

1. **GPU Scheduling**: Xiao, W. et al. "Gandiva: Introspective Cluster Scheduling for Deep Learning" OSDI (2018)
2. **NUMA Effects**: Lameter, C. "NUMA: An Overview" ACM Queue (2013)
3. **NVML Guide**: NVIDIA "NVML API Reference" developer.nvidia.com
4. **MIG Documentation**: NVIDIA "Multi-Instance GPU User Guide" docs.nvidia.com
5. **XID Errors**: NVIDIA "XID Errors Reference" docs.nvidia.com/deploy
6. **Kubernetes GPU**: kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
