"""Job entities representing workload requests and their state.

Jobs model ML training/inference tasks submitted to the GPU scheduler.
They track allocation state, GPU requirements, and resource quotas.

References:
    - design.md Section 4 (Job State Machine)
    - design.md Section 5 (Job Placement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from gpu_platform.domain.value_objects.gpu_identifiers import GPUId, JobId, NUMANodeId


class JobState(Enum):
    """Job lifecycle state."""
    PENDING = "pending"         # Created but not queued
    QUEUED = "queued"           # Waiting for resource availability
    SCHEDULED = "scheduled"     # Resources allocated, starting
    RUNNING = "running"         # Actively executing
    PREEMPTED = "preempted"     # Paused for higher priority job
    COMPLETED = "completed"     # Finished successfully
    FAILED = "failed"           # Error or timeout
    CANCELLED = "cancelled"     # User cancelled the job


class JobPriority(Enum):
    """Job scheduling priority (higher number = higher priority)."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class GPURequest:
    """GPU resource requirements for a job."""
    gpu_count: int               # Number of GPUs needed
    min_memory_mb: int          # Minimum memory per GPU
    supports_mig: bool = False  # Can use MIG partitions
    allow_cross_numa: bool = False  # OK if GPUs on different NUMA nodes
    prefer_nvlink: bool = False # Prefer NVLink interconnect


@dataclass
class Job:
    """ML job submitted to the scheduler."""
    job_id: JobId
    user_id: str                # User who submitted job
    priority: JobPriority       # Scheduling priority
    gpu_request: GPURequest     # GPU requirements
    state: JobState = JobState.PENDING
    
    # Allocation state
    allocated_gpus: list[GPUId] = field(default_factory=list)
    allocated_numa_node: Optional[NUMANodeId] = None
    
    # Resource quotas (per-user limits)
    max_gpu_hours: Optional[float] = None  # Budget in GPU-hours
    max_concurrent_gpus: int = 4           # Max GPUs this user can use
    
    # Metadata
    submitted_timestamp: float = 0.0
    started_timestamp: Optional[float] = None
    completed_timestamp: Optional[float] = None
    preemption_count: int = 0
    
    @property
    def is_multi_gpu(self) -> bool:
        """Check if job requires multiple GPUs."""
        return self.gpu_request.gpu_count > 1
    
    @property
    def is_allocated(self) -> bool:
        """Check if job has GPU allocation."""
        return len(self.allocated_gpus) == self.gpu_request.gpu_count
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since job started."""
        if self.started_timestamp is None:
            return 0.0
        end = self.completed_timestamp or self.started_timestamp
        return end - self.started_timestamp


@dataclass
class Placement:
    """GPU placement decision for a job."""
    job_id: JobId
    gpu_ids: list[GPUId]        # Assigned GPUs
    numa_node: NUMANodeId       # Target NUMA node
    is_cross_numa: bool         # True if GPUs span NUMA nodes
    mig_slices: Optional[list[int]] = None  # MIG slice assignments if used
    
    @property
    def gpu_count(self) -> int:
        """Number of GPUs in this placement."""
        return len(self.gpu_ids)
