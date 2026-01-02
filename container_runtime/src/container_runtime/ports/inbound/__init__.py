"""Inbound ports - API contracts for the container runtime.

Inbound ports define the interfaces that clients and upper layers
use to interact with container management, scheduling, and image handling.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from container_runtime.domain.entities.container import (
    Container,
    ContainerConfig,
    ContainerImage,
    ContainerStats,
)
from container_runtime.domain.entities.job import (
    Job,
    Placement,
)


# =============================================================================
# Container Manager Port
# =============================================================================


@dataclass
class ContainerManagerStats:
    """Statistics for container manager monitoring."""

    total_containers: int
    running_containers: int
    stopped_containers: int
    failed_containers: int

    @property
    def utilization_ratio(self) -> float:
        """Calculate running/total ratio."""
        if self.total_containers == 0:
            return 0.0
        return self.running_containers / self.total_containers


class ContainerManagerPort(Protocol):
    """Protocol for container lifecycle operations.

    The container manager handles creating, starting, stopping,
    and deleting containers.

    Thread Safety:
        All methods must be thread-safe.

    Example:
        container = manager.create(config)
        try:
            manager.start(container.container_id)
            # ... use container
        finally:
            manager.stop(container.container_id)
            manager.delete(container.container_id)
    """

    @abstractmethod
    def create(self, config: ContainerConfig) -> Container:
        """Create a new container.

        Args:
            config: Container configuration.

        Returns:
            The created container.

        Raises:
            ValueError: If config is invalid or container ID exists.
        """
        ...

    @abstractmethod
    def start(self, container_id: str) -> None:
        """Start a container.

        Args:
            container_id: Container to start.

        Raises:
            ValueError: If container not found.
            ContainerError: If container cannot be started.
        """
        ...

    @abstractmethod
    def stop(self, container_id: str, timeout_seconds: int = 10) -> None:
        """Stop a container.

        Sends SIGTERM, waits for timeout, then SIGKILL if needed.

        Args:
            container_id: Container to stop.
            timeout_seconds: Grace period before kill.

        Raises:
            ValueError: If container not found.
        """
        ...

    @abstractmethod
    def delete(self, container_id: str) -> None:
        """Delete a container.

        Args:
            container_id: Container to delete.

        Raises:
            ValueError: If container not found.
            ContainerError: If container is running.
        """
        ...

    @abstractmethod
    def get(self, container_id: str) -> Optional[Container]:
        """Get container by ID.

        Args:
            container_id: Container ID.

        Returns:
            Container or None if not found.
        """
        ...

    @abstractmethod
    def list(self) -> list[Container]:
        """List all containers.

        Returns:
            List of all containers.
        """
        ...

    @abstractmethod
    def get_stats(self, container_id: str) -> ContainerStats:
        """Get container runtime statistics.

        Args:
            container_id: Container ID.

        Returns:
            Container statistics.

        Raises:
            ValueError: If container not found.
        """
        ...


class ContainerError(Exception):
    """Raised when container operation fails."""

    pass


# =============================================================================
# Scheduler Port
# =============================================================================


@dataclass
class SchedulerStats:
    """Statistics for scheduler monitoring."""

    total_jobs: int
    pending_jobs: int
    scheduled_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    node_count: int

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        total_finished = self.completed_jobs + self.failed_jobs
        if total_finished == 0:
            return 0.0
        return self.completed_jobs / total_finished


class SchedulerPort(Protocol):
    """Protocol for job scheduling operations.

    The scheduler handles job submission, placement decisions,
    and resource allocation using bin-packing algorithms.

    Thread Safety:
        All methods must be thread-safe.

    Determinism:
        Given the same jobs and resources, schedule_all() must
        produce identical placements for debugging/replay.

    Example:
        job_id = scheduler.submit(job)
        placements = scheduler.schedule_all()
        if placements[job_id].assigned:
            # Job was placed, start container
            pass
    """

    @abstractmethod
    def submit(self, job: Job) -> str:
        """Submit a job for scheduling.

        Args:
            job: Job to schedule.

        Returns:
            Job ID.

        Raises:
            ValueError: If job ID already exists.
        """
        ...

    @abstractmethod
    def schedule_all(self) -> dict[str, Placement]:
        """Schedule all pending jobs.

        Uses First-Fit Decreasing (FFD) bin-packing algorithm
        for deterministic placement.

        Returns:
            Dict mapping job_id -> placement decision.
        """
        ...

    @abstractmethod
    def get_placement(self, job_id: str) -> Optional[Placement]:
        """Get placement decision for a job.

        Args:
            job_id: Job ID.

        Returns:
            Placement or None if not scheduled.
        """
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a job and release resources.

        Args:
            job_id: Job to cancel.
        """
        ...

    @abstractmethod
    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics.

        Returns:
            Scheduler statistics.
        """
        ...


# =============================================================================
# Image Manager Port
# =============================================================================


@dataclass
class ImageManagerStats:
    """Statistics for image manager monitoring."""

    total_images: int
    total_size_bytes: int
    cache_hit_count: int
    cache_miss_count: int

    @property
    def total_size_mb(self) -> float:
        """Get total size in MB."""
        return self.total_size_bytes / 1_000_000

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return 0.0
        return self.cache_hit_count / total


class ImageManagerPort(Protocol):
    """Protocol for container image operations.

    The image manager handles pulling, caching, and managing
    OCI container images.

    Thread Safety:
        All methods must be thread-safe.

    Example:
        image = manager.pull("docker.io", "nginx", "latest")
        if manager.exists(image.image_id):
            # Use image
            pass
    """

    @abstractmethod
    def pull(self, registry: str, name: str, tag: str = "latest") -> ContainerImage:
        """Pull (or get cached) image from registry.

        Args:
            registry: Registry URL (e.g., "docker.io").
            name: Image name.
            tag: Image tag.

        Returns:
            The pulled image.

        Raises:
            ImagePullError: If pull fails.
        """
        ...

    @abstractmethod
    def get(self, image_id: str) -> Optional[ContainerImage]:
        """Get image by ID.

        Args:
            image_id: Image ID.

        Returns:
            Image or None if not found.
        """
        ...

    @abstractmethod
    def list(self) -> list[ContainerImage]:
        """List all cached images.

        Returns:
            List of all images.
        """
        ...

    @abstractmethod
    def delete(self, image_id: str) -> None:
        """Delete an image from cache.

        Args:
            image_id: Image to delete.
        """
        ...

    @abstractmethod
    def exists(self, image_id: str) -> bool:
        """Check if image exists in cache.

        Args:
            image_id: Image ID.

        Returns:
            True if exists.
        """
        ...


class ImagePullError(Exception):
    """Raised when image pull fails."""

    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Container Manager
    "ContainerManagerPort",
    "ContainerManagerStats",
    "ContainerError",
    # Scheduler
    "SchedulerPort",
    "SchedulerStats",
    # Image Manager
    "ImageManagerPort",
    "ImageManagerStats",
    "ImagePullError",
]
