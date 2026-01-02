"""Inbound port interfaces for the GPU platform.

Inbound ports define what the system offers to external clients.
Adapters will implement these with gRPC, REST, CLI, etc.
"""

from __future__ import annotations

from typing import Protocol

from gpu_platform.domain.entities.job import Job, Placement
from gpu_platform.domain.entities.topology import Topology
from gpu_platform.domain.value_objects.gpu_identifiers import JobId


class GPUPlatformAPI(Protocol):
    """Main API offered by the GPU platform."""
    
    def submit_job(self, job: Job) -> JobId:
        """Submit a new job to the scheduler.
        
        Args:
            job: Job to submit.
            
        Returns:
            JobId assigned to the job.
        """
        ...
    
    def cancel_job(self, job_id: JobId) -> bool:
        """Cancel a running or queued job.
        
        Args:
            job_id: Job to cancel.
            
        Returns:
            True if cancelled, False if not found.
        """
        ...
    
    def get_job_status(self, job_id: JobId) -> Job | None:
        """Get current status of a job.
        
        Args:
            job_id: Job to query.
            
        Returns:
            Job with current state, or None if not found.
        """
        ...
    
    def get_cluster_topology(self) -> Topology:
        """Get cluster GPU and NUMA topology.
        
        Returns:
            Cluster topology including GPUs and NUMA nodes.
        """
        ...
    
    def get_cluster_stats(self) -> dict:
        """Get cluster-wide statistics.
        
        Returns:
            Dictionary with stats like GPU utilization, pending jobs, etc.
        """
        ...
