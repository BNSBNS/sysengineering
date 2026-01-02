"""Mock namespace manager for testing and development.

This adapter provides a mock implementation of the NamespaceManagerPort
protocol for use in testing and on non-Linux systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from container_runtime.ports.outbound import (
    NamespaceManagerPort,
    NamespaceType,
    NamespaceConfig,
    NamespaceError,
)


logger = logging.getLogger(__name__)


@dataclass
class MockNamespaceState:
    """State for mock namespaces."""

    ns_fd: int
    config: NamespaceConfig
    hostname: str = "container"
    network_configured: bool = False
    mounts_configured: bool = False


@dataclass
class MockNetworkState:
    """State for mock network configuration."""

    container_id: str
    veth_host: str = ""
    veth_container: str = ""
    ip_address: str = ""
    gateway: str = ""


class MockNamespaceManager:
    """Mock implementation of NamespaceManagerPort for testing.

    Simulates Linux namespace operations in memory without requiring
    a real Linux kernel.

    Example:
        manager = MockNamespaceManager()
        ns_fd = manager.create_namespaces(NamespaceConfig())
        manager.set_hostname("mycontainer")
        manager.setup_network("container-1", {"ip": "10.0.0.2"})
    """

    def __init__(self):
        """Initialize mock namespace manager."""
        self._namespaces: dict[int, MockNamespaceState] = {}
        self._networks: dict[str, MockNetworkState] = {}
        self._next_fd = 1000  # Simulated file descriptor
        self._current_hostname = "host"

    def create_namespaces(self, config: NamespaceConfig) -> int:
        """Create mock namespaces.

        Args:
            config: Namespace configuration.

        Returns:
            Simulated file descriptor for the namespace set.
        """
        ns_fd = self._next_fd
        self._next_fd += 1

        self._namespaces[ns_fd] = MockNamespaceState(
            ns_fd=ns_fd,
            config=config,
        )

        logger.debug(f"Created mock namespaces with fd={ns_fd}: {config}")
        return ns_fd

    def enter_namespace(self, ns_fd: int, ns_type: NamespaceType) -> None:
        """Simulate entering a namespace.

        Args:
            ns_fd: File descriptor for the namespace.
            ns_type: Type of namespace to enter.

        Raises:
            NamespaceError: If namespace not found.
        """
        if ns_fd not in self._namespaces:
            raise NamespaceError(f"Namespace not found: fd={ns_fd}")

        logger.debug(f"Entered mock namespace fd={ns_fd} type={ns_type}")

    def set_hostname(self, hostname: str) -> None:
        """Set hostname (simulated).

        Args:
            hostname: New hostname.
        """
        self._current_hostname = hostname
        logger.debug(f"Set mock hostname: {hostname}")

    def setup_network(self, container_id: str, config: dict) -> None:
        """Set up mock network for a container.

        Args:
            container_id: Container identifier.
            config: Network configuration.

        Raises:
            NamespaceError: If network already configured.
        """
        if container_id in self._networks:
            raise NamespaceError(f"Network already configured: {container_id}")

        network = MockNetworkState(
            container_id=container_id,
            veth_host=f"veth-{container_id[:8]}-h",
            veth_container=f"veth-{container_id[:8]}-c",
            ip_address=config.get("ip", "10.0.0.2"),
            gateway=config.get("gateway", "10.0.0.1"),
        )

        self._networks[container_id] = network
        logger.debug(f"Set up mock network for {container_id}: {config}")

    def cleanup_network(self, container_id: str) -> None:
        """Clean up mock network resources.

        Args:
            container_id: Container identifier.
        """
        if container_id in self._networks:
            del self._networks[container_id]
            logger.debug(f"Cleaned up mock network for {container_id}")

    def setup_mounts(self, rootfs: str, mounts: list[dict]) -> None:
        """Set up mock mount points.

        Args:
            rootfs: Path to container root filesystem.
            mounts: List of mount configurations.
        """
        logger.debug(f"Set up mock mounts for rootfs={rootfs}: {len(mounts)} mounts")

    # Test helpers
    def get_namespace(self, ns_fd: int) -> MockNamespaceState | None:
        """Get namespace state for testing.

        Args:
            ns_fd: Namespace file descriptor.

        Returns:
            Namespace state or None.
        """
        return self._namespaces.get(ns_fd)

    def get_network(self, container_id: str) -> MockNetworkState | None:
        """Get network state for testing.

        Args:
            container_id: Container identifier.

        Returns:
            Network state or None.
        """
        return self._networks.get(container_id)

    def get_hostname(self) -> str:
        """Get current hostname for testing.

        Returns:
            Current hostname.
        """
        return self._current_hostname

    def clear(self) -> None:
        """Clear all state (for testing)."""
        self._namespaces.clear()
        self._networks.clear()
        self._current_hostname = "host"
