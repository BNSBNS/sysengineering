"""Cluster topology entities representing GPU-to-CPU-to-NUMA layout.

Topology models the physical hardware layout including NUMA nodes,
CPU cores, GPUs, and their interconnects (PCIe, NVLink).

References:
    - design.md Section 3 (GPU Discovery)
    - design.md Section 5 (NUMA-Aware Placement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpu_platform.domain.value_objects.gpu_identifiers import GPUId, NUMANodeId


@dataclass
class NUMANode:
    """A NUMA node containing CPUs and memory."""
    node_id: NUMANodeId
    cpu_cores: list[int]         # List of CPU core IDs
    memory_gb: int               # Local memory capacity
    gpus: list[GPUId] = field(default_factory=list)  # Attached GPUs (PCIe)
    
    @property
    def cpu_count(self) -> int:
        """Number of CPUs in this node."""
        return len(self.cpu_cores)
    
    @property
    def gpu_count(self) -> int:
        """Number of GPUs in this node."""
        return len(self.gpus)


@dataclass
class NVLink:
    """NVLink connection between two GPUs."""
    gpu_a: GPUId
    gpu_b: GPUId
    bandwidth_gbps: float = 600.0  # NVLink 3.0 = 600 GB/s per direction


@dataclass
class Topology:
    """Cluster-wide GPU and NUMA topology."""
    numa_nodes: dict[int, NUMANode]  # Map node_id -> NUMANode
    nvlinks: list[NVLink] = field(default_factory=list)
    
    @property
    def node_count(self) -> int:
        """Number of NUMA nodes."""
        return len(self.numa_nodes)
    
    @property
    def total_gpus(self) -> int:
        """Total GPUs in cluster."""
        return sum(node.gpu_count for node in self.numa_nodes.values())
    
    @property
    def total_memory_gb(self) -> int:
        """Total NUMA memory in cluster."""
        return sum(node.memory_gb for node in self.numa_nodes.values())
    
    def get_node_for_gpu(self, gpu_id: GPUId) -> Optional[NUMANodeId]:
        """Find which NUMA node contains a GPU."""
        for node_id, node in self.numa_nodes.items():
            if gpu_id in node.gpus:
                return NUMANodeId(node_id)
        return None
    
    def get_gpus_in_node(self, node_id: int) -> list[GPUId]:
        """Get all GPUs in a NUMA node."""
        if node_id not in self.numa_nodes:
            return []
        return self.numa_nodes[node_id].gpus.copy()
    
    def are_gpus_connected(self, gpu_a: GPUId, gpu_b: GPUId) -> bool:
        """Check if two GPUs are connected via NVLink."""
        for link in self.nvlinks:
            if (link.gpu_a == gpu_a and link.gpu_b == gpu_b) or \
               (link.gpu_a == gpu_b and link.gpu_b == gpu_a):
                return True
        return False
    
    def find_nvlink_path(self, gpu_a: GPUId, gpu_b: GPUId) -> Optional[list[GPUId]]:
        """Find shortest NVLink path between two GPUs (BFS)."""
        if gpu_a == gpu_b:
            return [gpu_a]
        
        from collections import deque
        
        # Build adjacency list
        adj: dict[GPUId, list[GPUId]] = {}
        for link in self.nvlinks:
            if link.gpu_a not in adj:
                adj[link.gpu_a] = []
            if link.gpu_b not in adj:
                adj[link.gpu_b] = []
            adj[link.gpu_a].append(link.gpu_b)
            adj[link.gpu_b].append(link.gpu_a)
        
        # BFS
        queue = deque([(gpu_a, [gpu_a])])
        visited = {gpu_a}
        
        while queue:
            current, path = queue.popleft()
            if current == gpu_b:
                return path
            
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
