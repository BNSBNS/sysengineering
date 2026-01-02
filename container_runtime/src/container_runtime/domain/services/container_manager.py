"""Container manager service."""

from __future__ import annotations

from typing import Optional
from container_runtime.domain.entities.container import (
    Container,
    ContainerConfig,
    ContainerState,
    ContainerStats,
)


class ContainerManager:
    """Manages container lifecycle.
    
    Handles:
    - Container creation
    - Container startup/shutdown
    - Container execution
    - Container cleanup
    """
    
    def __init__(self):
        """Initialize container manager."""
        self._containers: dict[str, Container] = {}
        self._next_pid = 1000  # Simulate PIDs
    
    def create(self, config: ContainerConfig) -> Container:
        """Create a container.
        
        Args:
            config: Container configuration.
            
        Returns:
            Created container.
        """
        if not config.validate():
            raise ValueError("Invalid container config")
        
        if config.container_id in self._containers:
            raise ValueError(f"Container {config.container_id} already exists")
        
        container = Container(
            container_id=config.container_id,
            image_id=config.image_id,
            limits=config.limits,
        )
        
        self._containers[config.container_id] = container
        return container
    
    def start(self, container_id: str) -> None:
        """Start a container.
        
        Args:
            container_id: Container ID.
        """
        container = self._get_container(container_id)
        
        if container.state != ContainerState.CREATED:
            raise Exception(f"Cannot start container in state {container.state}")
        
        # Simulate process startup
        container.pid = self._next_pid
        self._next_pid += 1
        
        container.start()
    
    def stop(self, container_id: str, timeout_seconds: int = 10) -> None:
        """Stop a container.
        
        Args:
            container_id: Container ID.
            timeout_seconds: Kill timeout.
        """
        container = self._get_container(container_id)
        
        if container.state not in [ContainerState.RUNNING, ContainerState.CREATED]:
            raise Exception(f"Cannot stop container in state {container.state}")
        
        container.stop(exit_code=0)
        container.pid = None
    
    def delete(self, container_id: str) -> None:
        """Delete a container.
        
        Args:
            container_id: Container ID.
        """
        container = self._get_container(container_id)
        
        if container.state == ContainerState.RUNNING:
            raise Exception("Cannot delete running container")
        
        container.state = ContainerState.DELETED
        self._containers.pop(container_id, None)
    
    def get(self, container_id: str) -> Optional[Container]:
        """Get container by ID.
        
        Args:
            container_id: Container ID.
            
        Returns:
            Container or None.
        """
        return self._containers.get(container_id)
    
    def list(self) -> list[Container]:
        """List all containers.
        
        Returns:
            List of containers.
        """
        return list(self._containers.values())
    
    def get_stats(self, container_id: str) -> ContainerStats:
        """Get container statistics.
        
        Args:
            container_id: Container ID.
            
        Returns:
            Container stats.
        """
        container = self._get_container(container_id)
        
        uptime = container.get_uptime_seconds()
        
        # Simulate stats
        stats = ContainerStats(
            container_id=container_id,
            state=container.state,
            uptime_seconds=uptime,
            cpu_percent=25.0 if container.is_running() else 0.0,
            memory_used_bytes=100_000_000 if container.is_running() else 0,
            memory_limit_bytes=container.limits.memory_limit,
            pids_count=5 if container.is_running() else 0,
        )
        
        return stats
    
    def _get_container(self, container_id: str) -> Container:
        """Get container or raise error.
        
        Args:
            container_id: Container ID.
            
        Returns:
            Container.
            
        Raises:
            Exception if not found.
        """
        container = self._containers.get(container_id)
        if not container:
            raise ValueError(f"Container {container_id} not found")
        return container
