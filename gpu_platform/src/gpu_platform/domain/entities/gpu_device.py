"""GPU device entities representing physical GPU hardware.

Entities model the state and configuration of physical GPU devices,
including their topology, health status, and allocation status.

References:
    - design.md Section 3 (GPU Discovery)
    - design.md Section 4 (Data Models)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from gpu_platform.domain.value_objects.gpu_identifiers import GPUId, NUMANodeId, PCIeBusId


class GPUState(Enum):
    """GPU device lifecycle state."""
    AVAILABLE = "available"       # Ready to be allocated
    ALLOCATED = "allocated"       # Allocated to a job
    IN_USE = "in_use"             # Currently running workload
    UNHEALTHY = "unhealthy"       # ECC error or hardware issue
    RESET = "reset"               # Recovering from error
    OFFLINE = "offline"           # Permanently disabled


@dataclass
class GPUSpecs:
    """Hardware specifications of a GPU."""
    gpu_id: GPUId
    model: str                    # e.g., "A100", "H100"
    compute_capability: str       # e.g., "8.0", "9.0"
    memory_mb: int               # Total GPU memory in MB
    pcie_bus_id: PCIeBusId       # PCIe address
    nvlink_ports: int = 0        # Number of NVLink connections
    supports_mig: bool = False   # Can partition with MIG


@dataclass
class GPUHealth:
    """Health metrics for a GPU."""
    gpu_id: GPUId
    temperature_c: float           # GPU die temperature
    power_w: float                # Current power draw
    utilization_percent: float     # Compute utilization %
    memory_used_mb: int           # Used GPU memory
    ecc_errors_correctable: int   # Single-bit errors (corrected)
    ecc_errors_uncorrectable: int # Multi-bit errors (fatal)
    throttled: bool               # Thermal or power throttling
    last_update_timestamp: float  # When these metrics were collected


@dataclass
class GPUDevice:
    """Physical GPU device with state and health."""
    specs: GPUSpecs
    numa_node: NUMANodeId         # NUMA affinity for memory transfer
    state: GPUState = GPUState.AVAILABLE
    health: Optional[GPUHealth] = None
    allocated_jobs: list[str] = field(default_factory=list)  # Job IDs using this GPU
    mig_instances: int = 0        # Number of active MIG slices
    
    @property
    def is_healthy(self) -> bool:
        """Check if GPU is healthy for allocation."""
        if not self.health:
            return False
        # Uncorrectable errors mean GPU is dead
        if self.health.ecc_errors_uncorrectable > 0:
            return False
        # Thermal throttling means GPU is too hot
        if self.health.throttled:
            return False
        return True
    
    @property
    def is_available(self) -> bool:
        """Check if GPU can accept new allocations."""
        return self.state == GPUState.AVAILABLE and self.is_healthy
    
    @property
    def available_memory_mb(self) -> int:
        """Calculate available GPU memory."""
        if not self.health:
            return self.specs.memory_mb
        return self.specs.memory_mb - self.health.memory_used_mb
