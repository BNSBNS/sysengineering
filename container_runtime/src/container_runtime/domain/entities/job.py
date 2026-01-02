"""Job and scheduling entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class JobState(Enum):
    """Job lifecycle state."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Placement:
    """Container placement decision."""
    container_id: str
    assigned: bool  # True if successfully placed
    reason: str = ""  # Why failed if not assigned
    cpu_allocation: int = 0  # CPU shares allocated
    memory_allocation: int = 0  # Memory bytes allocated
    gpu_allocation: list[str] = field(default_factory=list)  # GPU IDs


@dataclass
class Job:
    """Scheduled job entity."""
    job_id: str
    image_id: str
    container_id: str
    state: JobState = JobState.PENDING
    priority: int = 50  # 0-100, higher = more important
    cpu_required: int = 1024  # CPU shares
    memory_required: int = 256_000_000  # 256MB default
    gpu_required: int = 0  # Number of GPUs
    timeout_seconds: int | None = None  # Job timeout
    
    created_at: float = field(default_factory=time.time)
    scheduled_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    
    placement: Optional[Placement] = None
    error_message: str = ""
    
    def schedule(self, placement: Placement) -> None:
        """Mark job as scheduled.
        
        Args:
            placement: Placement decision.
        """
        if self.state != JobState.PENDING:
            raise Exception(f"Cannot schedule job in state {self.state}")
        self.state = JobState.SCHEDULED
        self.placement = placement
        self.scheduled_at = time.time()
    
    def run(self) -> None:
        """Mark job as running."""
        if self.state != JobState.SCHEDULED:
            raise Exception(f"Cannot run job in state {self.state}")
        self.state = JobState.RUNNING
        self.started_at = time.time()
    
    def complete(self) -> None:
        """Mark job as completed."""
        if self.state != JobState.RUNNING:
            raise Exception(f"Cannot complete job in state {self.state}")
        self.state = JobState.COMPLETED
        self.completed_at = time.time()
    
    def fail(self, error: str) -> None:
        """Mark job as failed.
        
        Args:
            error: Error message.
        """
        self.state = JobState.FAILED
        self.error_message = error
        self.completed_at = time.time()
    
    def cancel(self) -> None:
        """Cancel job."""
        if self.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
            raise Exception(f"Cannot cancel job in state {self.state}")
        self.state = JobState.CANCELLED
        self.completed_at = time.time()
    
    def is_active(self) -> bool:
        """Check if job is still active.
        
        Returns:
            True if job is pending, scheduled, or running.
        """
        return self.state in [JobState.PENDING, JobState.SCHEDULED, JobState.RUNNING]


@dataclass
class ResourceAllocation:
    """Allocated resources on a node."""
    total_cpu_shares: int = 0
    total_memory_bytes: int = 0
    total_gpu_count: int = 0
    
    # Available/free resources
    available_cpu_shares: int = 0
    available_memory_bytes: int = 0
    available_gpu_ids: list[str] = field(default_factory=list)
    
    def has_capacity(self, cpu: int, memory: int, gpu: int) -> bool:
        """Check if node has capacity for resources.
        
        Args:
            cpu: CPU shares needed.
            memory: Memory bytes needed.
            gpu: GPU count needed.
            
        Returns:
            True if resources available.
        """
        return (
            self.available_cpu_shares >= cpu
            and self.available_memory_bytes >= memory
            and len(self.available_gpu_ids) >= gpu
        )
    
    def allocate(self, cpu: int, memory: int, gpu: int, gpu_ids: list[str]) -> None:
        """Allocate resources.
        
        Args:
            cpu: CPU shares to allocate.
            memory: Memory bytes to allocate.
            gpu: GPU count to allocate.
            gpu_ids: GPU IDs to allocate.
        """
        self.available_cpu_shares -= cpu
        self.available_memory_bytes -= memory
        for gpu_id in gpu_ids:
            if gpu_id in self.available_gpu_ids:
                self.available_gpu_ids.remove(gpu_id)
    
    def deallocate(self, cpu: int, memory: int, gpu_ids: list[str]) -> None:
        """Deallocate resources.
        
        Args:
            cpu: CPU shares to deallocate.
            memory: Memory bytes to deallocate.
            gpu_ids: GPU IDs to deallocate.
        """
        self.available_cpu_shares += cpu
        self.available_memory_bytes += memory
        self.available_gpu_ids.extend(gpu_ids)
    
    def utilization_percent(self) -> float:
        """Get resource utilization percentage.
        
        Returns:
            Utilization 0-100%.
        """
        if self.total_memory_bytes == 0:
            return 0
        used = self.total_memory_bytes - self.available_memory_bytes
        return (used / self.total_memory_bytes) * 100


@dataclass
class SchedulingDecision:
    """Decision from scheduler for a job."""
    job_id: str
    placement: Placement
    estimated_wait_seconds: int = 0
    reason: str = ""
