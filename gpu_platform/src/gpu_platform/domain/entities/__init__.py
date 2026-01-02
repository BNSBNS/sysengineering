"""Domain entities for the GPU platform.

Entities represent core business objects with identity and lifecycle:
- GPUDevice: Physical GPU with state and health
- Job: ML workload request
- Topology: Cluster NUMA and GPU layout
"""

from gpu_platform.domain.entities.gpu_device import (
    GPUDevice,
    GPUHealth,
    GPUSpecs,
    GPUState,
)
from gpu_platform.domain.entities.job import (
    GPURequest,
    Job,
    JobPriority,
    JobState,
    Placement,
)
from gpu_platform.domain.entities.topology import (
    NUMANode,
    NVLink,
    Topology,
)

__all__ = [
    # GPU Device
    "GPUDevice",
    "GPUHealth",
    "GPUSpecs",
    "GPUState",
    # Job
    "Job",
    "JobState",
    "JobPriority",
    "GPURequest",
    "Placement",
    # Topology
    "Topology",
    "NUMANode",
    "NVLink",
]
