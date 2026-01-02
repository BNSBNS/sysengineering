"""Mock cgroups manager for testing and development.

This adapter provides a mock implementation of the CgroupsManagerPort
protocol for use in testing and on non-Linux systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from container_runtime.ports.outbound import (
    CgroupsManagerPort,
    CgroupVersion,
    CgroupLimits,
    CgroupStats,
    CgroupError,
)


logger = logging.getLogger(__name__)


@dataclass
class MockCgroupState:
    """State for a mock cgroup."""

    path: str
    limits: CgroupLimits = field(default_factory=CgroupLimits)
    processes: list[int] = field(default_factory=list)
    frozen: bool = False
    # Simulated stats
    cpu_usage_usec: int = 0
    memory_current_bytes: int = 0


class MockCgroupsManager:
    """Mock implementation of CgroupsManagerPort for testing.

    Simulates cgroups v2 behavior in memory without requiring
    a real Linux kernel.

    Example:
        manager = MockCgroupsManager()
        manager.create_cgroup("containers/test")
        manager.set_limits("containers/test", CgroupLimits(memory_max_bytes=1024*1024*256))
        manager.add_process("containers/test", 12345)
    """

    def __init__(self):
        """Initialize mock cgroups manager."""
        self._cgroups: dict[str, MockCgroupState] = {}
        self._version = CgroupVersion.V2

    @property
    def version(self) -> CgroupVersion:
        """Return the cgroup version (always v2 for mock)."""
        return self._version

    def create_cgroup(self, cgroup_path: str) -> None:
        """Create a mock cgroup.

        Args:
            cgroup_path: Path relative to cgroup root.

        Raises:
            CgroupError: If cgroup already exists.
        """
        if cgroup_path in self._cgroups:
            raise CgroupError(f"Cgroup already exists: {cgroup_path}")

        self._cgroups[cgroup_path] = MockCgroupState(path=cgroup_path)
        logger.debug(f"Created mock cgroup: {cgroup_path}")

    def delete_cgroup(self, cgroup_path: str) -> None:
        """Delete a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If cgroup not found or has processes.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        cgroup = self._cgroups[cgroup_path]
        if cgroup.processes:
            raise CgroupError(f"Cgroup has processes: {cgroup_path}")

        del self._cgroups[cgroup_path]
        logger.debug(f"Deleted mock cgroup: {cgroup_path}")

    def set_limits(self, cgroup_path: str, limits: CgroupLimits) -> None:
        """Set resource limits on a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.
            limits: Resource limits to apply.

        Raises:
            CgroupError: If cgroup not found.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        self._cgroups[cgroup_path].limits = limits
        logger.debug(f"Set limits on {cgroup_path}: {limits}")

    def add_process(self, cgroup_path: str, pid: int) -> None:
        """Add a process to a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.
            pid: Process ID.

        Raises:
            CgroupError: If cgroup not found.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        cgroup = self._cgroups[cgroup_path]
        if pid not in cgroup.processes:
            cgroup.processes.append(pid)
        logger.debug(f"Added process {pid} to {cgroup_path}")

    def get_stats(self, cgroup_path: str) -> CgroupStats:
        """Get simulated stats for a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Returns:
            Simulated statistics.

        Raises:
            CgroupError: If cgroup not found.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        cgroup = self._cgroups[cgroup_path]

        # Simulate some usage
        return CgroupStats(
            cpu_usage_usec=cgroup.cpu_usage_usec + 1000,
            memory_current_bytes=cgroup.memory_current_bytes + 1024 * 1024,
            memory_peak_bytes=cgroup.memory_current_bytes + 2 * 1024 * 1024,
            pids_current=len(cgroup.processes),
            io_read_bytes=0,
            io_write_bytes=0,
        )

    def freeze(self, cgroup_path: str) -> None:
        """Freeze a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If cgroup not found.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        self._cgroups[cgroup_path].frozen = True
        logger.debug(f"Froze cgroup: {cgroup_path}")

    def thaw(self, cgroup_path: str) -> None:
        """Thaw a mock cgroup.

        Args:
            cgroup_path: Path to cgroup.

        Raises:
            CgroupError: If cgroup not found.
        """
        if cgroup_path not in self._cgroups:
            raise CgroupError(f"Cgroup not found: {cgroup_path}")

        self._cgroups[cgroup_path].frozen = False
        logger.debug(f"Thawed cgroup: {cgroup_path}")

    # Test helpers
    def get_cgroup(self, cgroup_path: str) -> MockCgroupState | None:
        """Get cgroup state for testing.

        Args:
            cgroup_path: Path to cgroup.

        Returns:
            Cgroup state or None.
        """
        return self._cgroups.get(cgroup_path)

    def clear(self) -> None:
        """Clear all cgroups (for testing)."""
        self._cgroups.clear()
