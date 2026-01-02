"""FastAPI REST adapter for the Container Runtime.

Provides HTTP endpoints for container and job management.

Usage:
    from container_runtime.adapters.inbound.rest_api import create_app

    app = create_app()
    # Run with: uvicorn module:app --host 0.0.0.0 --port 8080

References:
    - design.md Section 2.4 (API Services)
    - ports/inbound/api.py (API contracts)
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from container_runtime.domain.entities.container import (
    Container,
    ContainerConfig,
    ContainerState,
    ResourceLimits,
)
from container_runtime.domain.entities.job import (
    Job,
    JobState,
    ResourceAllocation,
)
from container_runtime.domain.services.container_manager import ContainerManager
from container_runtime.domain.services.image_manager import ImageManager
from container_runtime.domain.services.scheduler import Scheduler


# Pydantic models for request/response serialization
if _HAS_FASTAPI:

    class ResourceLimitsModel(BaseModel):
        """Resource limits configuration."""

        cpu_shares: int = Field(default=1024, ge=2, description="CPU shares")
        memory_limit: int = Field(default=536870912, ge=4194304, description="Memory limit bytes")
        cpu_quota: int = Field(default=-1, description="CPU quota microseconds")
        cpu_period: int = Field(default=100000, ge=1000, description="CPU period microseconds")

    class CreateContainerRequest(BaseModel):
        """Request to create a container."""

        container_id: str = Field(..., min_length=1, description="Container ID")
        image_id: str = Field(..., min_length=1, description="Image ID (e.g., ubuntu:22.04)")
        command: list[str] = Field(default_factory=list, description="Command to run")
        environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")
        limits: Optional[ResourceLimitsModel] = Field(default=None, description="Resource limits")

    class ContainerResponse(BaseModel):
        """Container details response."""

        container_id: str
        image_id: str
        state: str
        pid: Optional[int] = None
        exit_code: Optional[int] = None
        error_message: Optional[str] = None
        uptime_seconds: float = 0.0

    class ContainerListResponse(BaseModel):
        """List of containers."""

        containers: list[ContainerResponse]
        count: int

    class ContainerStatsResponse(BaseModel):
        """Container statistics."""

        container_id: str
        cpu_percent: float
        memory_mb: float
        memory_percent: float
        io_read_mb: float
        io_write_mb: float

    class SubmitJobRequest(BaseModel):
        """Request to submit a job."""

        job_id: str = Field(..., min_length=1, description="Job ID")
        image_id: str = Field(..., min_length=1, description="Image ID")
        cpu_required: int = Field(default=1024, ge=1, description="CPU shares required")
        memory_required: int = Field(default=536870912, ge=1, description="Memory bytes required")
        gpu_required: int = Field(default=0, ge=0, description="Number of GPUs required")
        priority: int = Field(default=0, description="Job priority")

    class JobResponse(BaseModel):
        """Job details response."""

        job_id: str
        image_id: str
        container_id: Optional[str] = None
        state: str
        cpu_required: int
        memory_required: int
        gpu_required: int
        priority: int
        placement_assigned: bool = False
        placement_reason: Optional[str] = None

    class JobListResponse(BaseModel):
        """List of jobs."""

        jobs: list[JobResponse]
        count: int

    class PlacementResponse(BaseModel):
        """Placement decision response."""

        job_id: str
        container_id: str
        assigned: bool
        reason: Optional[str] = None
        cpu_allocation: int
        memory_allocation: int
        gpu_allocation: list[str]

    class ScheduleResponse(BaseModel):
        """Scheduling result response."""

        placements: list[PlacementResponse]
        scheduled_count: int
        failed_count: int

    class PullImageRequest(BaseModel):
        """Request to pull an image."""

        registry: str = Field(default="docker.io", description="Container registry")
        name: str = Field(..., min_length=1, description="Image name")
        tag: str = Field(default="latest", description="Image tag")

    class ImageResponse(BaseModel):
        """Image details response."""

        image_id: str
        name: str
        tag: str
        registry: str
        size_mb: float

    class ImageListResponse(BaseModel):
        """List of images."""

        images: list[ImageResponse]
        count: int

    class NodeResourceModel(BaseModel):
        """Node resource configuration."""

        total_cpu_shares: int = Field(default=4096, description="Total CPU shares")
        total_memory_bytes: int = Field(default=4000000000, description="Total memory bytes")
        available_cpu_shares: int = Field(default=4096, description="Available CPU shares")
        available_memory_bytes: int = Field(default=4000000000, description="Available memory bytes")
        available_gpu_ids: list[str] = Field(default_factory=list, description="Available GPU IDs")

    class AddNodeRequest(BaseModel):
        """Request to add a node."""

        node_id: str = Field(..., min_length=1, description="Node ID")
        resources: NodeResourceModel

    class SchedulerStatsResponse(BaseModel):
        """Scheduler statistics."""

        total_jobs: int
        pending: int
        scheduled: int
        running: int
        completed: int
        failed: int
        nodes: int

    class MetricsResponse(BaseModel):
        """Runtime metrics."""

        containers_total: int
        containers_running: int
        containers_stopped: int
        containers_failed: int
        jobs_total: int
        jobs_pending: int
        jobs_scheduled: int
        images_cached: int

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str = "0.1.0"


def create_app(
    container_manager: ContainerManager | None = None,
    scheduler: Scheduler | None = None,
    image_manager: ImageManager | None = None,
) -> "FastAPI":
    """Create FastAPI application with Container Runtime endpoints.

    Args:
        container_manager: Optional ContainerManager instance.
        scheduler: Optional Scheduler instance.
        image_manager: Optional ImageManager instance.

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    # Initialize services
    containers = container_manager or ContainerManager()
    sched = scheduler or Scheduler()
    images = image_manager or ImageManager()

    app = FastAPI(
        title="Container Runtime API",
        description="Container orchestration with deterministic scheduling and resource management",
        version="1.0.0",
    )

    # Health endpoints
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check runtime health status."""
        return HealthResponse(status="healthy")

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def get_metrics():
        """Get runtime metrics."""
        container_list = containers.list()
        running = sum(1 for c in container_list if c.state == ContainerState.RUNNING)
        stopped = sum(1 for c in container_list if c.state == ContainerState.STOPPED)
        failed = sum(1 for c in container_list if c.state == ContainerState.FAILED)

        scheduler_stats = sched.get_stats()

        return MetricsResponse(
            containers_total=len(container_list),
            containers_running=running,
            containers_stopped=stopped,
            containers_failed=failed,
            jobs_total=scheduler_stats["total_jobs"],
            jobs_pending=scheduler_stats["pending"],
            jobs_scheduled=scheduler_stats["scheduled"],
            images_cached=len(images.list()),
        )

    # Container endpoints
    @app.post(
        "/containers",
        response_model=ContainerResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Containers"],
    )
    async def create_container(request: CreateContainerRequest):
        """Create a new container."""
        limits = None
        if request.limits:
            limits = ResourceLimits(
                cpu_shares=request.limits.cpu_shares,
                memory_limit=request.limits.memory_limit,
                cpu_quota=request.limits.cpu_quota,
                cpu_period=request.limits.cpu_period,
            )

        config = ContainerConfig(
            image_id=request.image_id,
            container_id=request.container_id,
            command=request.command,
            environment=request.environment,
            limits=limits,
        )

        if not config.validate():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid container configuration",
            )

        try:
            container = containers.create(config)
            return ContainerResponse(
                container_id=container.container_id,
                image_id=container.image_id,
                state=container.state.value,
                pid=container.pid,
                uptime_seconds=container.uptime,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )

    @app.get("/containers", response_model=ContainerListResponse, tags=["Containers"])
    async def list_containers():
        """List all containers."""
        container_list = containers.list()
        return ContainerListResponse(
            containers=[
                ContainerResponse(
                    container_id=c.container_id,
                    image_id=c.image_id,
                    state=c.state.value,
                    pid=c.pid,
                    exit_code=c.exit_code,
                    error_message=c.error_message,
                    uptime_seconds=c.uptime,
                )
                for c in container_list
            ],
            count=len(container_list),
        )

    @app.get("/containers/{container_id}", response_model=ContainerResponse, tags=["Containers"])
    async def get_container(container_id: str):
        """Get container details."""
        container = containers.get(container_id)
        if not container:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {container_id} not found",
            )
        return ContainerResponse(
            container_id=container.container_id,
            image_id=container.image_id,
            state=container.state.value,
            pid=container.pid,
            exit_code=container.exit_code,
            error_message=container.error_message,
            uptime_seconds=container.uptime,
        )

    @app.post("/containers/{container_id}/start", response_model=ContainerResponse, tags=["Containers"])
    async def start_container(container_id: str):
        """Start a container."""
        container = containers.get(container_id)
        if not container:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {container_id} not found",
            )
        try:
            containers.start(container_id)
            container = containers.get(container_id)
            return ContainerResponse(
                container_id=container.container_id,
                image_id=container.image_id,
                state=container.state.value,
                pid=container.pid,
                uptime_seconds=container.uptime,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.post("/containers/{container_id}/stop", response_model=ContainerResponse, tags=["Containers"])
    async def stop_container(container_id: str):
        """Stop a container."""
        container = containers.get(container_id)
        if not container:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {container_id} not found",
            )
        try:
            containers.stop(container_id)
            container = containers.get(container_id)
            return ContainerResponse(
                container_id=container.container_id,
                image_id=container.image_id,
                state=container.state.value,
                pid=container.pid,
                exit_code=container.exit_code,
                uptime_seconds=container.uptime,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.delete("/containers/{container_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Containers"])
    async def delete_container(container_id: str):
        """Delete a container."""
        container = containers.get(container_id)
        if not container:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {container_id} not found",
            )
        try:
            containers.delete(container_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.get("/containers/{container_id}/stats", response_model=ContainerStatsResponse, tags=["Containers"])
    async def get_container_stats(container_id: str):
        """Get container resource statistics."""
        container = containers.get(container_id)
        if not container:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {container_id} not found",
            )
        stats = containers.get_stats(container_id)
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Container {container_id} not running",
            )
        return ContainerStatsResponse(
            container_id=container_id,
            cpu_percent=stats.cpu_percent,
            memory_mb=stats.memory_bytes / (1024 * 1024),
            memory_percent=stats.memory_percent,
            io_read_mb=stats.io_read_bytes / (1024 * 1024),
            io_write_mb=stats.io_write_bytes / (1024 * 1024),
        )

    # Job endpoints
    @app.post(
        "/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Jobs"],
    )
    async def submit_job(request: SubmitJobRequest):
        """Submit a job for scheduling."""
        job = Job(
            job_id=request.job_id,
            image_id=request.image_id,
            container_id="",  # Will be assigned by scheduler
            cpu_required=request.cpu_required,
            memory_required=request.memory_required,
            gpu_required=request.gpu_required,
            priority=request.priority,
        )
        try:
            sched.submit(job)
            return JobResponse(
                job_id=job.job_id,
                image_id=job.image_id,
                state=job.state.value,
                cpu_required=job.cpu_required,
                memory_required=job.memory_required,
                gpu_required=job.gpu_required,
                priority=job.priority,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )

    @app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
    async def list_jobs():
        """List all jobs."""
        jobs = list(sched._jobs.values())
        return JobListResponse(
            jobs=[
                JobResponse(
                    job_id=j.job_id,
                    image_id=j.image_id,
                    container_id=j.placement.container_id if j.placement else None,
                    state=j.state.value,
                    cpu_required=j.cpu_required,
                    memory_required=j.memory_required,
                    gpu_required=j.gpu_required,
                    priority=j.priority,
                    placement_assigned=j.placement.assigned if j.placement else False,
                    placement_reason=j.placement.reason if j.placement else None,
                )
                for j in jobs
            ],
            count=len(jobs),
        )

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
    async def get_job(job_id: str):
        """Get job details."""
        job = sched._jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        return JobResponse(
            job_id=job.job_id,
            image_id=job.image_id,
            container_id=job.placement.container_id if job.placement else None,
            state=job.state.value,
            cpu_required=job.cpu_required,
            memory_required=job.memory_required,
            gpu_required=job.gpu_required,
            priority=job.priority,
            placement_assigned=job.placement.assigned if job.placement else False,
            placement_reason=job.placement.reason if job.placement else None,
        )

    @app.post("/jobs/{job_id}/cancel", response_model=JobResponse, tags=["Jobs"])
    async def cancel_job(job_id: str):
        """Cancel a job."""
        job = sched._jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        sched.cancel(job_id)
        job = sched._jobs.get(job_id)
        return JobResponse(
            job_id=job.job_id,
            image_id=job.image_id,
            container_id=job.placement.container_id if job.placement else None,
            state=job.state.value,
            cpu_required=job.cpu_required,
            memory_required=job.memory_required,
            gpu_required=job.gpu_required,
            priority=job.priority,
        )

    @app.post("/schedule", response_model=ScheduleResponse, tags=["Scheduling"])
    async def schedule_all():
        """Schedule all pending jobs."""
        placements = sched.schedule_all()
        placement_responses = []
        scheduled_count = 0
        failed_count = 0

        for job_id, placement in placements.items():
            placement_responses.append(
                PlacementResponse(
                    job_id=job_id,
                    container_id=placement.container_id,
                    assigned=placement.assigned,
                    reason=placement.reason,
                    cpu_allocation=placement.cpu_allocation,
                    memory_allocation=placement.memory_allocation,
                    gpu_allocation=placement.gpu_allocation,
                )
            )
            if placement.assigned:
                scheduled_count += 1
            else:
                failed_count += 1

        return ScheduleResponse(
            placements=placement_responses,
            scheduled_count=scheduled_count,
            failed_count=failed_count,
        )

    @app.get("/scheduler/stats", response_model=SchedulerStatsResponse, tags=["Scheduling"])
    async def get_scheduler_stats():
        """Get scheduler statistics."""
        stats = sched.get_stats()
        return SchedulerStatsResponse(**stats)

    @app.post("/scheduler/nodes", status_code=status.HTTP_201_CREATED, tags=["Scheduling"])
    async def add_node(request: AddNodeRequest):
        """Add a node to the scheduler."""
        node = ResourceAllocation(
            total_cpu_shares=request.resources.total_cpu_shares,
            total_memory_bytes=request.resources.total_memory_bytes,
            available_cpu_shares=request.resources.available_cpu_shares,
            available_memory_bytes=request.resources.available_memory_bytes,
            available_gpu_ids=request.resources.available_gpu_ids,
        )
        sched._nodes[request.node_id] = node
        return {"node_id": request.node_id, "status": "added"}

    # Image endpoints
    @app.post(
        "/images",
        response_model=ImageResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Images"],
    )
    async def pull_image(request: PullImageRequest):
        """Pull an image from registry."""
        try:
            image = images.pull(request.registry, request.name, request.tag)
            return ImageResponse(
                image_id=image.image_id,
                name=image.name,
                tag=image.tag,
                registry=image.registry,
                size_mb=images.get_size_mb(image.image_id),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.get("/images", response_model=ImageListResponse, tags=["Images"])
    async def list_images():
        """List all cached images."""
        image_list = images.list()
        return ImageListResponse(
            images=[
                ImageResponse(
                    image_id=img.image_id,
                    name=img.name,
                    tag=img.tag,
                    registry=img.registry,
                    size_mb=images.get_size_mb(img.image_id),
                )
                for img in image_list
            ],
            count=len(image_list),
        )

    @app.get("/images/{image_id}", response_model=ImageResponse, tags=["Images"])
    async def get_image(image_id: str):
        """Get image details."""
        image = images.get(image_id)
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found",
            )
        return ImageResponse(
            image_id=image.image_id,
            name=image.name,
            tag=image.tag,
            registry=image.registry,
            size_mb=images.get_size_mb(image.image_id),
        )

    @app.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Images"])
    async def delete_image(image_id: str):
        """Delete an image."""
        if not images.exists(image_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found",
            )
        images.delete(image_id)

    return app
