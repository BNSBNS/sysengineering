"""Outbound ports - External dependency interfaces for the container runtime.

Outbound ports define the interfaces for external dependencies that
the container runtime depends on for isolation and image management.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


# =============================================================================
# Cgroups Manager Port
# =============================================================================


class CgroupVersion(Enum):
    """Cgroup version."""
    V1 = "v1"
    V2 = "v2"


@dataclass
class CgroupLimits:
    """Resource limits to apply via cgroups."""
    cpu_weight: int = 100  # 1-10000 (cgroups v2)
    cpu_max_usec: int | None = None  # Max CPU time per period
    cpu_period_usec: int = 100000  # Period in microseconds
    memory_max_bytes: int | None = None  # Memory limit
    memory_swap_max_bytes: int | None = None  # Swap limit
    pids_max: int | None = None  # Max processes
    io_weight: int = 100  # 1-10000


@dataclass
class CgroupStats:
    """Cgroup resource usage statistics."""
    cpu_usage_usec: int = 0  # Total CPU time used
    memory_current_bytes: int = 0  # Current memory usage
    memory_peak_bytes: int = 0  # Peak memory usage
    pids_current: int = 0  # Current process count
    io_read_bytes: int = 0
    io_write_bytes: int = 0


class CgroupsManagerPort(Protocol):
    """Protocol for cgroups v2 operations.

    Manages resource limits and accounting for containers
    using Linux cgroups v2.

    Thread Safety:
        All methods must be thread-safe.

    References:
        - https://www.kernel.org/doc/Documentation/cgroup-v2.txt
    """

    @property
    @abstractmethod
    def version(self) -> CgroupVersion:
        """Return the cgroup version in use."""
        ...

    @abstractmethod
    def create_cgroup(self, cgroup_path: str) -> None:
        """Create a new cgroup.

        Args:
            cgroup_path: Path relative to cgroup root (e.g., "containers/c1").

        Raises:
            CgroupError: If creation fails.
        """
        ...

    @abstractmethod
    def delete_cgroup(self, cgroup_path: str) -> None:
        """Delete a cgroup.

        The cgroup must have no processes.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If deletion fails.
        """
        ...

    @abstractmethod
    def set_limits(self, cgroup_path: str, limits: CgroupLimits) -> None:
        """Set resource limits for a cgroup.

        Args:
            cgroup_path: Path to cgroup.
            limits: Resource limits to apply.

        Raises:
            CgroupError: If setting limits fails.
        """
        ...

    @abstractmethod
    def add_process(self, cgroup_path: str, pid: int) -> None:
        """Add a process to a cgroup.

        Args:
            cgroup_path: Path to cgroup.
            pid: Process ID to add.

        Raises:
            CgroupError: If adding process fails.
        """
        ...

    @abstractmethod
    def get_stats(self, cgroup_path: str) -> CgroupStats:
        """Get resource usage statistics for a cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Returns:
            Resource usage statistics.

        Raises:
            CgroupError: If reading stats fails.
        """
        ...

    @abstractmethod
    def freeze(self, cgroup_path: str) -> None:
        """Freeze all processes in a cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If freeze fails.
        """
        ...

    @abstractmethod
    def thaw(self, cgroup_path: str) -> None:
        """Thaw (unfreeze) all processes in a cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If thaw fails.
        """
        ...


class CgroupError(Exception):
    """Raised when cgroup operation fails."""

    pass


# =============================================================================
# Namespace Manager Port
# =============================================================================


class NamespaceType(Enum):
    """Linux namespace types."""
    PID = "pid"
    NET = "net"
    MNT = "mnt"
    UTS = "uts"
    IPC = "ipc"
    USER = "user"
    CGROUP = "cgroup"


@dataclass
class NamespaceConfig:
    """Configuration for namespace creation."""
    pid: bool = True  # Isolate process IDs
    net: bool = True  # Isolate network
    mnt: bool = True  # Isolate mount points
    uts: bool = True  # Isolate hostname
    ipc: bool = True  # Isolate IPC
    user: bool = False  # Isolate user IDs (requires mapping)
    cgroup: bool = True  # Isolate cgroup view

    # User namespace mappings
    uid_map: list[tuple[int, int, int]] = field(default_factory=list)  # (inside, outside, count)
    gid_map: list[tuple[int, int, int]] = field(default_factory=list)


class NamespaceManagerPort(Protocol):
    """Protocol for Linux namespace operations.

    Manages process isolation using Linux namespaces.

    Thread Safety:
        All methods must be thread-safe.

    References:
        - man 7 namespaces
        - man 2 clone
        - man 2 unshare
    """

    @abstractmethod
    def create_namespaces(self, config: NamespaceConfig) -> int:
        """Create new namespaces and return file descriptor.

        Uses clone() or unshare() to create new namespaces.

        Args:
            config: Namespace configuration.

        Returns:
            File descriptor for the namespace set.

        Raises:
            NamespaceError: If creation fails.
        """
        ...

    @abstractmethod
    def enter_namespace(self, ns_fd: int, ns_type: NamespaceType) -> None:
        """Enter an existing namespace.

        Uses setns() to join a namespace.

        Args:
            ns_fd: File descriptor for the namespace.
            ns_type: Type of namespace to enter.

        Raises:
            NamespaceError: If entering fails.
        """
        ...

    @abstractmethod
    def set_hostname(self, hostname: str) -> None:
        """Set hostname in current UTS namespace.

        Args:
            hostname: New hostname.

        Raises:
            NamespaceError: If setting hostname fails.
        """
        ...

    @abstractmethod
    def setup_network(self, container_id: str, config: dict) -> None:
        """Set up network for a container.

        Creates veth pair, assigns to namespace, configures IP.

        Args:
            container_id: Container identifier.
            config: Network configuration.

        Raises:
            NamespaceError: If network setup fails.
        """
        ...

    @abstractmethod
    def cleanup_network(self, container_id: str) -> None:
        """Clean up network resources for a container.

        Args:
            container_id: Container identifier.

        Raises:
            NamespaceError: If cleanup fails.
        """
        ...

    @abstractmethod
    def setup_mounts(self, rootfs: str, mounts: list[dict]) -> None:
        """Set up mount points for a container.

        Args:
            rootfs: Path to container root filesystem.
            mounts: List of mount configurations.

        Raises:
            NamespaceError: If mount setup fails.
        """
        ...


class NamespaceError(Exception):
    """Raised when namespace operation fails."""

    pass


# =============================================================================
# Image Registry Port
# =============================================================================


@dataclass
class ImageManifest:
    """OCI image manifest."""
    schema_version: int
    media_type: str
    config_digest: str
    layers: list[str]  # Layer digests
    total_size: int
    platform: str = "linux/amd64"


@dataclass
class RegistryAuth:
    """Registry authentication credentials."""
    username: str = ""
    password: str = ""
    token: str = ""
    auth_type: str = "basic"  # basic, bearer


class ImageRegistryPort(Protocol):
    """Protocol for OCI image registry operations.

    Handles pulling images from container registries
    following the OCI Distribution Specification.

    Thread Safety:
        All methods must be thread-safe.

    References:
        - https://github.com/opencontainers/distribution-spec
    """

    @abstractmethod
    def authenticate(self, registry: str, auth: RegistryAuth) -> str:
        """Authenticate with a registry.

        Args:
            registry: Registry URL.
            auth: Authentication credentials.

        Returns:
            Authentication token for subsequent requests.

        Raises:
            RegistryError: If authentication fails.
        """
        ...

    @abstractmethod
    def get_manifest(
        self, registry: str, name: str, tag: str, token: str = ""
    ) -> ImageManifest:
        """Get image manifest from registry.

        Args:
            registry: Registry URL.
            name: Image name.
            tag: Image tag.
            token: Authentication token.

        Returns:
            Image manifest.

        Raises:
            RegistryError: If manifest fetch fails.
        """
        ...

    @abstractmethod
    def pull_layer(
        self,
        registry: str,
        name: str,
        digest: str,
        dest_path: str,
        token: str = "",
    ) -> None:
        """Pull and extract an image layer.

        Args:
            registry: Registry URL.
            name: Image name.
            digest: Layer digest.
            dest_path: Destination path for extraction.
            token: Authentication token.

        Raises:
            RegistryError: If layer pull fails.
        """
        ...

    @abstractmethod
    def pull_config(
        self,
        registry: str,
        name: str,
        digest: str,
        token: str = "",
    ) -> dict:
        """Pull image configuration.

        Args:
            registry: Registry URL.
            name: Image name.
            digest: Config digest.
            token: Authentication token.

        Returns:
            Image configuration as dict.

        Raises:
            RegistryError: If config pull fails.
        """
        ...

    @abstractmethod
    def check_image_exists(
        self, registry: str, name: str, tag: str, token: str = ""
    ) -> bool:
        """Check if an image exists in the registry.

        Args:
            registry: Registry URL.
            name: Image name.
            tag: Image tag.
            token: Authentication token.

        Returns:
            True if image exists.
        """
        ...


class RegistryError(Exception):
    """Raised when registry operation fails."""

    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Cgroups
    "CgroupsManagerPort",
    "CgroupVersion",
    "CgroupLimits",
    "CgroupStats",
    "CgroupError",
    # Namespaces
    "NamespaceManagerPort",
    "NamespaceType",
    "NamespaceConfig",
    "NamespaceError",
    # Image Registry
    "ImageRegistryPort",
    "ImageManifest",
    "RegistryAuth",
    "RegistryError",
]
