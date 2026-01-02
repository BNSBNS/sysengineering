"""GPU-related type-safe identifiers.

These value objects provide type safety for GPU-related identifiers
using Python's NewType for zero-runtime overhead.

References:
    - design.md Section 2 (Architecture)
"""

from __future__ import annotations

from typing import NewType

# GPU unique identifier (UUID from NVML)
GPUId = NewType("GPUId", str)

# Job identifier in the scheduler
JobId = NewType("JobId", str)

# NUMA node identifier
NUMANodeId = NewType("NUMANodeId", int)

# PCIe bus identifier (e.g., "0000:01:00.0")
PCIeBusId = NewType("PCIeBusId", str)


def create_gpu_id(index: int) -> GPUId:
    """Create a GPU identifier from an index."""
    return GPUId(f"GPU-{index}")


def create_job_id(prefix: str, counter: int) -> JobId:
    """Create a job identifier from prefix and counter."""
    return JobId(f"{prefix}-{counter:08d}")
