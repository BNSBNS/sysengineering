"""Outbound adapters - Implementations of outbound port interfaces.

Provides mock implementations for testing and development on non-Linux systems.
"""

from container_runtime.adapters.outbound.mock_cgroups_manager import (
    MockCgroupsManager,
    MockCgroupState,
)
from container_runtime.adapters.outbound.mock_namespace_manager import (
    MockNamespaceManager,
    MockNamespaceState,
    MockNetworkState,
)

__all__ = [
    # Cgroups
    "MockCgroupsManager",
    "MockCgroupState",
    # Namespaces
    "MockNamespaceManager",
    "MockNamespaceState",
    "MockNetworkState",
]
