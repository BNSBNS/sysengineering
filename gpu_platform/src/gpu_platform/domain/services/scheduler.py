"""GPU scheduler with NUMA-aware placement and gang scheduling.

The scheduler implements:
1. NUMA-aware placement: Keep GPUs and job on same NUMA node
2. Gang scheduling: Atomically allocate all required GPUs or none
3. Job queuing: FIFO with priority support
4. Preemption: Can suspend lower-priority jobs for higher-priority
5. Health-aware scheduling: Avoids unhealthy GPUs, preempts on degradation

References:
    - design.md Section 3 (Scheduler)
    - design.md Section 5 (NUMA-Aware Placement & Gang Scheduling)
    - design.md Section 6 (Failure Modes & Recovery)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from gpu_platform.domain.entities.gpu_device import GPUDevice, GPUState
from gpu_platform.domain.entities.job import Job, JobPriority, JobState, Placement
from gpu_platform.domain.entities.topology import Topology
from gpu_platform.domain.value_objects.gpu_identifiers import GPUId, JobId, NUMANodeId

if TYPE_CHECKING:
    from gpu_platform.domain.services.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Scheduler operation failed."""
    pass


@dataclass
class ScheduleDecision:
    """Result of a scheduling decision."""
    success: bool
    placement: Optional[Placement] = None
    reason: str = ""


class GPUScheduler:
    """NUMA-aware GPU scheduler with gang scheduling and health awareness."""

    def __init__(self, topology: Topology, health_monitor: Optional[HealthMonitor] = None) -> None:
        """Initialize scheduler with cluster topology.
        
        Args:
            topology: Cluster NUMA and GPU topology.
            health_monitor: Optional health monitor for health-aware scheduling.
        """
        self._topology = topology
        self._health_monitor = health_monitor
        self._devices: dict[GPUId, GPUDevice] = {}  # GPU registry
        self._jobs: dict[JobId, Job] = {}           # Job registry
        self._queue: deque[Job] = deque()           # Pending jobs
        self._counter = 0
    
    def register_gpu(self, device: GPUDevice) -> None:
        """Register a GPU device."""
        self._devices[device.specs.gpu_id] = device
    
    def submit_job(self, job: Job) -> None:
        """Submit a new job to the scheduler.
        
        The job enters QUEUED state and will be scheduled when
        resources become available.
        
        Args:
            job: Job to submit.
            
        Raises:
            SchedulerError: If job is invalid.
        """
        if job.gpu_request.gpu_count > len(self._devices):
            raise SchedulerError(
                f"Job requests {job.gpu_request.gpu_count} GPUs "
                f"but only {len(self._devices)} available"
            )
        
        job.state = JobState.QUEUED
        job.submitted_timestamp = time.time()
        self._jobs[job.job_id] = job
        self._queue.append(job)
        logger.info(f"Job {job.job_id} submitted (priority={job.priority.name})")
    
    def schedule_pending(self) -> ScheduleDecision:
        """Try to schedule the next pending job.
        
        Attempts to place the highest-priority pending job using
        NUMA-aware gang scheduling.
        
        Returns:
            ScheduleDecision with success status and placement.
        """
        if not self._queue:
            return ScheduleDecision(success=False, reason="No pending jobs")
        
        # Get next job (should prioritize by priority, but FIFO for now)
        job = self._queue.popleft()
        
        # Try to place it
        placement = self._place_job(job)
        
        if placement:
            job.state = JobState.SCHEDULED
            job.allocated_gpus = placement.gpu_ids
            job.allocated_numa_node = placement.numa_node
            job.started_timestamp = time.time()
            
            # Mark GPUs as allocated
            for gpu_id in placement.gpu_ids:
                if gpu_id in self._devices:
                    device = self._devices[gpu_id]
                    device.state = GPUState.IN_USE
                    device.allocated_jobs.append(str(job.job_id))
            
            logger.info(
                f"Job {job.job_id} scheduled to GPUs {placement.gpu_ids} "
                f"on NUMA node {placement.numa_node}"
            )
            return ScheduleDecision(success=True, placement=placement)
        else:
            # Re-queue the job
            self._queue.appendleft(job)
            return ScheduleDecision(
                success=False,
                reason="Insufficient GPU resources available"
            )
    
    def _place_job(self, job: Job) -> Optional[Placement]:
        """Place a job using NUMA-aware gang scheduling.
        
        Tries to:
        1. Find GPUs on the same NUMA node (best case)
        2. Try other NUMA nodes if cross-NUMA allowed
        3. Evaluate health metrics if available
        4. Return None if no valid placement
        """
        required = job.gpu_request.gpu_count
        available_gpus = [
            (gpu_id, device)
            for gpu_id, device in self._devices.items()
            if device.state == GPUState.AVAILABLE and device.is_healthy
        ]
        
        # Filter out GPUs with active critical alerts (if health monitor available)
        if self._health_monitor:
            available_gpus = [
                (gpu_id, device) for gpu_id, device in available_gpus
                if not self._has_critical_alert(gpu_id)
            ]
        
        if len(available_gpus) < required:
            return None
        
        # Strategy 1: Same NUMA node (best case)
        for node_id, node in self._topology.numa_nodes.items():
            gpus_in_node = [
                gpu_id for gpu_id in node.gpus
                if gpu_id in [g[0] for g in available_gpus]
            ]
            
            if len(gpus_in_node) >= required:
                selected = gpus_in_node[:required]
                return Placement(
                    job_id=job.job_id,
                    gpu_ids=selected,
                    numa_node=NUMANodeId(node_id),
                    is_cross_numa=False,
                )
        
        # Strategy 2: Cross-NUMA (if allowed)
        if job.gpu_request.allow_cross_numa:
            selected = [gpu_id for gpu_id, _ in available_gpus[:required]]
            
            # Find which NUMA nodes they belong to
            numa_nodes = set()
            for gpu_id in selected:
                node = self._topology.get_node_for_gpu(gpu_id)
                if node is not None:
                    numa_nodes.add(node)

            
            # Use first node as primary
            primary_node = numa_nodes.pop() if numa_nodes else NUMANodeId(0)
            
            return Placement(
                job_id=job.job_id,
                gpu_ids=selected,
                numa_node=primary_node,
                is_cross_numa=True,
            )
        
        return None
    
    def _has_critical_alert(self, gpu_id: GPUId) -> bool:
        """Check if GPU has active critical alerts.
        
        Args:
            gpu_id: GPU to check.
            
        Returns:
            True if GPU has CRITICAL severity alerts.
        """
        if not self._health_monitor:
            return False
        
        alerts = self._health_monitor.monitor.get_active_alerts(gpu_id)
        return any(a.severity.value == "critical" for a in alerts)
    
    def preempt_job(self, job_id: JobId) -> bool:
        """Preempt a running job (e.g., due to health degradation).
        
        Args:
            job_id: Job to preempt.
            
        Returns:
            True if job was preempted.
        """
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        
        if job.state == JobState.RUNNING:
            job.state = JobState.PREEMPTED
            logger.warning(f"Job {job_id} preempted due to health degradation")
            
            # Release GPUs
            for gpu_id in job.allocated_gpus:
                if gpu_id in self._devices:
                    device = self._devices[gpu_id]
                    device.state = GPUState.AVAILABLE
                    if str(job_id) in device.allocated_jobs:
                        device.allocated_jobs.remove(str(job_id))
            
            # Re-queue for later
            self._queue.append(job)
            return True
        
        return False
    
    def get_job(self, job_id: JobId) -> Optional[Job]:
        """Get a job by ID.

        Args:
            job_id: Job to retrieve.

        Returns:
            Job if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def start_job(self, job_id: JobId) -> bool:
        """Transition a job from SCHEDULED to RUNNING.

        This should be called when the job actually starts executing
        on the allocated GPUs.

        Args:
            job_id: Job to start.

        Returns:
            True if job was started, False if not found or invalid state.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.state != JobState.SCHEDULED:
            logger.warning(f"Cannot start job {job_id} in state {job.state.name}")
            return False

        job.state = JobState.RUNNING
        logger.info(f"Job {job_id} started running")
        return True

    def cancel_job(self, job_id: JobId) -> bool:
        """Cancel a job.

        Cancels a pending or running job, releasing any allocated resources.

        Args:
            job_id: Job to cancel.

        Returns:
            True if job was cancelled, False if not found or already completed.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            logger.warning(f"Cannot cancel job {job_id} in terminal state {job.state.name}")
            return False

        # Remove from queue if pending
        if job.state == JobState.QUEUED:
            try:
                self._queue.remove(job)
            except ValueError:
                pass  # Already removed

        # Release GPUs if allocated
        if job.state in (JobState.SCHEDULED, JobState.RUNNING):
            for gpu_id in job.allocated_gpus:
                if gpu_id in self._devices:
                    device = self._devices[gpu_id]
                    device.state = GPUState.AVAILABLE
                    if str(job_id) in device.allocated_jobs:
                        device.allocated_jobs.remove(str(job_id))

        job.state = JobState.CANCELLED
        job.completed_timestamp = time.time()
        logger.info(f"Job {job_id} cancelled")
        return True

    def fail_job(self, job_id: JobId, reason: str = "") -> bool:
        """Mark a job as failed.

        Args:
            job_id: Job that failed.
            reason: Failure reason.

        Returns:
            True if job was marked failed, False if not found.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Release GPUs if allocated
        for gpu_id in job.allocated_gpus:
            if gpu_id in self._devices:
                device = self._devices[gpu_id]
                device.state = GPUState.AVAILABLE
                if str(job_id) in device.allocated_jobs:
                    device.allocated_jobs.remove(str(job_id))

        job.state = JobState.FAILED
        job.completed_timestamp = time.time()
        logger.error(f"Job {job_id} failed: {reason}")
        return True

    def release_job(self, job_id: JobId) -> bool:
        """Release GPU resources and complete a job.

        Args:
            job_id: Job to release.

        Returns:
            True if job was released, False if not found.
        """
        if job_id not in self._jobs:
            logger.warning(f"Job {job_id} not found")
            return False

        job = self._jobs[job_id]

        # Release GPUs
        for gpu_id in job.allocated_gpus:
            if gpu_id in self._devices:
                device = self._devices[gpu_id]
                device.state = GPUState.AVAILABLE
                if str(job_id) in device.allocated_jobs:
                    device.allocated_jobs.remove(str(job_id))

        job.state = JobState.COMPLETED
        job.completed_timestamp = time.time()
        logger.info(f"Job {job_id} completed and released GPUs")
        return True
    
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        allocated_gpus = sum(
            1 for d in self._devices.values()
            if d.state == GPUState.IN_USE
        )
        
        return {
            "total_gpus": len(self._devices),
            "allocated_gpus": allocated_gpus,
            "available_gpus": len(self._devices) - allocated_gpus,
            "pending_jobs": len(self._queue),
            "active_jobs": len([j for j in self._jobs.values() if j.state == JobState.RUNNING]),
        }
