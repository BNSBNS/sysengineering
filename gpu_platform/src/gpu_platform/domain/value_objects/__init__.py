"""Domain value objects for the GPU platform.

Value objects are immutable objects without identity that represent
core concepts like GPU IDs, Job IDs, and NUMA topology addresses.
"""

from gpu_platform.domain.value_objects.gpu_identifiers import (
    GPUId,
    JobId,
    NUMANodeId,
    PCIeBusId,
    create_gpu_id,
    create_job_id,
)

__all__ = [
    "GPUId",
    "JobId",
    "NUMANodeId",
    "PCIeBusId",
    "create_gpu_id",
    "create_job_id",
]
