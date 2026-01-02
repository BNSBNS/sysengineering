"""Scheduler service with bin-packing algorithm."""

from __future__ import annotations

from typing import Optional
from container_runtime.domain.entities.job import (
    Job,
    JobState,
    Placement,
    ResourceAllocation,
)


class Scheduler:
    """Deterministic scheduler with bin-packing placement.
    
    Uses First-Fit Decreasing (FFD) algorithm:
    1. Sort jobs by resource requirement (descending)
    2. For each job, find first node with capacity
    3. Allocate to that node
    
    Ensures deterministic placement for debugging/replay.
    """
    
    def __init__(self, nodes: dict[str, ResourceAllocation] = None):
        """Initialize scheduler.
        
        Args:
            nodes: Available nodes with resource allocations.
        """
        self._jobs: dict[str, Job] = {}
        self._pending_queue: list[str] = []  # Job IDs waiting for placement
        self._nodes = nodes or {"node-0": ResourceAllocation()}
        self._placements: dict[str, Placement] = {}
        self._next_container_id = 0
    
    def submit(self, job: Job) -> str:
        """Submit a job for scheduling.
        
        Args:
            job: Job to schedule.
            
        Returns:
            Job ID.
        """
        if job.job_id in self._jobs:
            raise ValueError(f"Job {job.job_id} already exists")
        
        self._jobs[job.job_id] = job
        self._pending_queue.append(job.job_id)
        
        return job.job_id
    
    def schedule_all(self) -> dict[str, Placement]:
        """Schedule all pending jobs.
        
        Uses bin-packing to minimize resource fragmentation.
        
        Returns:
            Dict mapping job_id -> placement.
        """
        scheduled = {}
        failed = []
        
        # Sort by memory requirement (descending) for FFD
        pending = [self._jobs[jid] for jid in self._pending_queue]
        pending.sort(key=lambda j: j.memory_required, reverse=True)
        
        for job in pending:
            placement = self._find_placement(job)
            
            if placement.assigned:
                # Allocate resources
                node_id = "node-0"  # Simplified: single node
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node.allocate(
                        job.cpu_required,
                        job.memory_required,
                        job.gpu_required,
                        placement.gpu_allocation,
                    )
                
                job.schedule(placement)
                scheduled[job.job_id] = placement
                self._placements[job.job_id] = placement
            else:
                # Keep in pending queue
                failed.append(job.job_id)
                # Still record the placement (failed)
                scheduled[job.job_id] = placement
                self._placements[job.job_id] = placement
        
        # Update pending queue
        self._pending_queue = failed
        
        return scheduled
    
    def get_placement(self, job_id: str) -> Optional[Placement]:
        """Get placement for a job.
        
        Args:
            job_id: Job ID.
            
        Returns:
            Placement or None.
        """
        return self._placements.get(job_id)
    
    def cancel(self, job_id: str) -> None:
        """Cancel a job.
        
        Args:
            job_id: Job ID.
        """
        job = self._jobs.get(job_id)
        if not job:
            return
        
        if job.is_active():
            job.cancel()
            
            # Deallocate resources
            if job.placement:
                node_id = "node-0"  # Simplified
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node.deallocate(
                        job.cpu_required,
                        job.memory_required,
                        job.placement.gpu_allocation,
                    )
        
        if job.job_id in self._pending_queue:
            self._pending_queue.remove(job.job_id)
    
    def _find_placement(self, job: Job) -> Placement:
        """Find placement for a job using bin-packing.
        
        Args:
            job: Job to place.
            
        Returns:
            Placement decision.
        """
        # First-Fit: check each node
        for node_id, node in self._nodes.items():
            if node.has_capacity(job.cpu_required, job.memory_required, job.gpu_required):
                # Allocate GPU IDs
                gpu_ids = node.available_gpu_ids[:job.gpu_required]
                
                placement = Placement(
                    container_id=f"container-{self._next_container_id}",
                    assigned=True,
                    cpu_allocation=job.cpu_required,
                    memory_allocation=job.memory_required,
                    gpu_allocation=gpu_ids,
                )
                self._next_container_id += 1
                return placement
        
        # No capacity found
        return Placement(
            container_id="",
            assigned=False,
            reason="Insufficient resources",
        )
    
    def get_stats(self) -> dict:
        """Get scheduler statistics.
        
        Returns:
            Scheduler stats.
        """
        total_jobs = len(self._jobs)
        pending = len([j for j in self._jobs.values() if j.state == JobState.PENDING])
        scheduled = len([j for j in self._jobs.values() if j.state == JobState.SCHEDULED])
        running = len([j for j in self._jobs.values() if j.state == JobState.RUNNING])
        completed = len([j for j in self._jobs.values() if j.state == JobState.COMPLETED])
        failed = len([j for j in self._jobs.values() if j.state == JobState.FAILED])
        
        return {
            "total_jobs": total_jobs,
            "pending": pending,
            "scheduled": scheduled,
            "running": running,
            "completed": completed,
            "failed": failed,
            "nodes": len(self._nodes),
        }
