"""Container and image entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class ContainerState(Enum):
    """Container lifecycle state."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    DELETED = "deleted"


class ImageFormat(Enum):
    """Container image format."""
    OCI = "oci"
    DOCKER = "docker"


@dataclass
class ResourceLimits:
    """Resource limits for a container."""
    cpu_shares: int = 1024  # Relative CPU weight
    cpu_quota: int | None = None  # Microseconds per period
    cpu_period: int = 100000  # Period in microseconds
    memory_limit: int | None = None  # Bytes
    memory_swap: int | None = None  # Bytes
    pids_limit: int | None = None  # Max processes
    gpu_ids: list[str] = field(default_factory=list)  # GPU UUIDs
    gpu_memory: int | None = None  # GPU memory limit per device
    
    def is_valid(self) -> bool:
        """Validate resource limits.
        
        Returns:
            True if valid.
        """
        if self.memory_limit is not None and self.memory_limit < 4_000_000:
            return False  # Less than 4MB
        if self.cpu_period < 1000:
            return False  # Less than 1ms
        if self.pids_limit is not None and self.pids_limit < 1:
            return False
        return True


@dataclass
class Container:
    """Container entity."""
    container_id: str
    image_id: str
    state: ContainerState = ContainerState.CREATED
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    stopped_at: float | None = None
    pid: int | None = None  # Host process ID
    exit_code: int | None = None
    error_message: str = ""
    
    def start(self) -> None:
        """Mark container as started."""
        if self.state != ContainerState.CREATED:
            raise Exception(f"Cannot start container in state {self.state}")
        self.state = ContainerState.RUNNING
        self.started_at = time.time()
    
    def stop(self, exit_code: int = 0) -> None:
        """Mark container as stopped.
        
        Args:
            exit_code: Container exit code.
        """
        if self.state not in [ContainerState.RUNNING, ContainerState.CREATED]:
            raise Exception(f"Cannot stop container in state {self.state}")
        self.state = ContainerState.STOPPED
        self.stopped_at = time.time()
        self.exit_code = exit_code
    
    def fail(self, error: str) -> None:
        """Mark container as failed.
        
        Args:
            error: Error message.
        """
        self.state = ContainerState.FAILED
        self.error_message = error
        self.stopped_at = time.time()
    
    def is_running(self) -> bool:
        """Check if container is running.
        
        Returns:
            True if running.
        """
        return self.state == ContainerState.RUNNING
    
    def get_uptime_seconds(self) -> float:
        """Get container uptime in seconds.
        
        Returns:
            Uptime seconds.
        """
        if not self.started_at:
            return 0
        end = self.stopped_at or time.time()
        return end - self.started_at


@dataclass
class ContainerImage:
    """Container image entity."""
    image_id: str
    name: str
    tag: str = "latest"
    format: ImageFormat = ImageFormat.OCI
    size_bytes: int = 0
    layer_count: int = 0
    created_at: float = field(default_factory=time.time)
    registry: str = "docker.io"  # Default Docker Hub
    digest: str = ""  # Content digest (SHA256)
    
    def full_name(self) -> str:
        """Get full image name with tag.
        
        Returns:
            Full name (name:tag).
        """
        return f"{self.name}:{self.tag}"
    
    def full_ref(self) -> str:
        """Get full image reference with registry.
        
        Returns:
            Full reference (registry/name:tag).
        """
        return f"{self.registry}/{self.name}:{self.tag}"


@dataclass
class ContainerConfig:
    """Configuration for creating a container."""
    image_id: str
    container_id: str
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str = "/"
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    privileged: bool = False
    readonly_rootfs: bool = False
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if valid.
        """
        if not self.image_id:
            return False
        if not self.container_id:
            return False
        if not self.limits.is_valid():
            return False
        return True


@dataclass
class ContainerStats:
    """Runtime statistics for a container."""
    container_id: str
    state: ContainerState
    uptime_seconds: float
    cpu_percent: float = 0.0  # 0-100
    memory_used_bytes: int = 0
    memory_limit_bytes: int | None = None
    pids_count: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    timestamp: float = field(default_factory=time.time)
