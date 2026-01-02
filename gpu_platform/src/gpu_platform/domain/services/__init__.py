"""Domain services for GPU platform business logic.

Services implement core workflows:
- GPUDiscoveryService: NVML-based GPU discovery and health monitoring
- GPUScheduler: NUMA-aware scheduling with gang scheduling
"""

from gpu_platform.domain.services.gpu_discovery import (
    GPUDiscoveryError,
    GPUDiscoveryService,
)
from gpu_platform.domain.services.scheduler import (
    GPUScheduler,
    ScheduleDecision,
    SchedulerError,
)

__all__ = [
    "GPUDiscoveryService",
    "GPUDiscoveryError",
    "GPUScheduler",
    "ScheduleDecision",
    "SchedulerError",
]
