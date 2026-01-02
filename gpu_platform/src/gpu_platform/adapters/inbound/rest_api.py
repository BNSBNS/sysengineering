"""FastAPI REST adapter for the GPU Platform.

Provides HTTP endpoints for job submission, management, and cluster monitoring.

Usage:
    from gpu_platform.adapters.inbound.rest_api import create_app

    app = create_app(coordinator)
    # Run with: uvicorn module:app --host 0.0.0.0 --port 8080

References:
    - design.md Section 8 (API Design)
    - ports/inbound/api.py (GPUPlatformAPI interface)
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class GPURequestModel(BaseModel):
        """GPU resource requirements."""

        gpu_count: int = Field(..., ge=1, le=8, description="Number of GPUs required")
        min_memory_mb: int = Field(
            ..., ge=1024, description="Minimum memory per GPU in MB"
        )
        supports_mig: bool = Field(default=False, description="Can use MIG partitions")
        allow_cross_numa: bool = Field(
            default=False, description="Allow GPUs from different NUMA nodes"
        )
        prefer_nvlink: bool = Field(
            default=False, description="Prefer NVLink interconnect"
        )

    class JobSubmitRequest(BaseModel):
        """Request to submit a new job."""

        job_id: str = Field(..., min_length=1, description="Unique job identifier")
        user_id: str = Field(..., min_length=1, description="User submitting the job")
        priority: str = Field(
            default="normal", description="Priority: low, normal, high, critical"
        )
        gpu_request: GPURequestModel

    class JobStatusResponse(BaseModel):
        """Job status response."""

        job_id: str
        user_id: str
        state: str
        priority: str
        gpu_count: int
        allocated_gpus: list[str]
        submitted_timestamp: float
        started_timestamp: Optional[float]
        completed_timestamp: Optional[float]

    class ClusterStatsResponse(BaseModel):
        """Cluster statistics response."""

        total_gpus: int
        available_gpus: int
        allocated_gpus: int
        queued_jobs: int
        running_jobs: int
        completed_jobs: int = 0
        failed_jobs: int = 0

    class TopologyNodeResponse(BaseModel):
        """NUMA node topology."""

        node_id: int
        cpu_cores: list[int]
        memory_gb: int
        gpus: list[str]

    class TopologyResponse(BaseModel):
        """Cluster topology response."""

        node_count: int
        total_gpus: int
        total_memory_gb: int
        nodes: list[TopologyNodeResponse]

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        initialized: bool
        total_gpus: int
        available_gpus: int


def create_app(coordinator) -> "FastAPI":
    """Create FastAPI application with GPU platform endpoints.

    Args:
        coordinator: GPUPlatformCoordinator instance.

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    from gpu_platform.domain.entities.job import (
        GPURequest,
        Job,
        JobPriority,
    )

    app = FastAPI(
        title="GPU Platform API",
        description="NUMA-aware GPU cluster management for ML workloads",
        version="1.0.0",
    )

    # Priority mapping
    PRIORITY_MAP = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "critical": JobPriority.CRITICAL,
    }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check platform health status."""
        stats = coordinator.get_cluster_stats()
        return HealthResponse(
            status="healthy" if coordinator._initialized else "initializing",
            initialized=coordinator._initialized,
            total_gpus=stats.get("total_gpus", 0),
            available_gpus=stats.get("available_gpus", 0),
        )

    @app.get("/stats", response_model=ClusterStatsResponse, tags=["Cluster"])
    async def get_cluster_stats():
        """Get cluster-wide statistics."""
        stats = coordinator.get_cluster_stats()
        return ClusterStatsResponse(
            total_gpus=stats.get("total_gpus", 0),
            available_gpus=stats.get("available_gpus", 0),
            allocated_gpus=stats.get("allocated_gpus", 0),
            queued_jobs=stats.get("queued_jobs", 0),
            running_jobs=stats.get("running_jobs", 0),
            completed_jobs=stats.get("completed_jobs", 0),
            failed_jobs=stats.get("failed_jobs", 0),
        )

    @app.get("/topology", response_model=TopologyResponse, tags=["Cluster"])
    async def get_cluster_topology():
        """Get cluster GPU and NUMA topology."""
        topology = coordinator.get_cluster_topology()
        nodes = []
        for node_id, node in topology.numa_nodes.items():
            nodes.append(
                TopologyNodeResponse(
                    node_id=node.node_id,
                    cpu_cores=list(node.cpu_cores),
                    memory_gb=node.memory_gb,
                    gpus=list(node.gpus),
                )
            )
        return TopologyResponse(
            node_count=topology.node_count,
            total_gpus=topology.total_gpus,
            total_memory_gb=topology.total_memory_gb,
            nodes=nodes,
        )

    @app.post("/jobs", response_model=dict, status_code=status.HTTP_201_CREATED, tags=["Jobs"])
    async def submit_job(request: JobSubmitRequest):
        """Submit a new GPU job."""
        priority = PRIORITY_MAP.get(request.priority.lower(), JobPriority.NORMAL)

        gpu_request = GPURequest(
            gpu_count=request.gpu_request.gpu_count,
            min_memory_mb=request.gpu_request.min_memory_mb,
            supports_mig=request.gpu_request.supports_mig,
            allow_cross_numa=request.gpu_request.allow_cross_numa,
            prefer_nvlink=request.gpu_request.prefer_nvlink,
        )

        job = Job(
            job_id=request.job_id,
            user_id=request.user_id,
            priority=priority,
            gpu_request=gpu_request,
        )

        try:
            job_id = coordinator.submit_job(job)
            return {"job_id": str(job_id), "status": "submitted"}
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
    async def get_job_status(job_id: str):
        """Get status of a specific job."""
        job = coordinator.get_job_status(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
            )
        return JobStatusResponse(
            job_id=str(job.job_id),
            user_id=job.user_id,
            state=job.state.value,
            priority=job.priority.name.lower(),
            gpu_count=job.gpu_request.gpu_count,
            allocated_gpus=[str(g) for g in job.allocated_gpus],
            submitted_timestamp=job.submitted_timestamp,
            started_timestamp=job.started_timestamp,
            completed_timestamp=job.completed_timestamp,
        )

    @app.delete("/jobs/{job_id}", response_model=dict, tags=["Jobs"])
    async def cancel_job(job_id: str):
        """Cancel a running or queued job."""
        result = coordinator.cancel_job(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be cancelled",
            )
        return {"job_id": job_id, "status": "cancelled"}

    @app.post("/jobs/{job_id}/start", response_model=dict, tags=["Jobs"])
    async def start_job(job_id: str):
        """Transition a scheduled job to running state."""
        result = coordinator.start_job(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} not found or not in scheduled state",
            )
        return {"job_id": job_id, "status": "running"}

    @app.post("/jobs/{job_id}/complete", response_model=dict, tags=["Jobs"])
    async def complete_job(job_id: str):
        """Mark a job as completed and release resources."""
        result = coordinator.complete_job(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} not found or not running",
            )
        return {"job_id": job_id, "status": "completed"}

    @app.post("/jobs/{job_id}/fail", response_model=dict, tags=["Jobs"])
    async def fail_job(job_id: str, reason: str = ""):
        """Mark a job as failed."""
        result = coordinator.fail_job(job_id, reason)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} not found",
            )
        return {"job_id": job_id, "status": "failed", "reason": reason}

    return app
