"""Consumer group coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ConsumerState(Enum):
    """Consumer state in group."""
    JOINING = "joining"
    STABLE = "stable"
    REBALANCING = "rebalancing"
    LEAVING = "leaving"


@dataclass
class ConsumerMember:
    """Consumer member in a group."""
    consumer_id: str
    group_id: str
    assigned_partitions: list[int] = field(default_factory=list)
    state: ConsumerState = ConsumerState.JOINING
    session_timeout_ms: int = 10_000  # 10 seconds
    last_heartbeat: float = field(default_factory=time.time)
    
    def is_alive(self) -> bool:
        """Check if consumer is alive (heartbeat still valid).
        
        Returns:
            True if consumer is responding.
        """
        elapsed_ms = (time.time() - self.last_heartbeat) * 1000
        return elapsed_ms < self.session_timeout_ms
    
    def update_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()


@dataclass
class ConsumerGroup:
    """Consumer group with members and assignments."""
    group_id: str
    members: dict[str, ConsumerMember] = field(default_factory=dict)
    generation: int = 0  # Incremented on rebalance
    leader: Optional[str] = None  # Group leader
    state: ConsumerState = ConsumerState.STABLE
    protocol_type: str = "range"  # "range", "roundrobin", "sticky"
    
    def add_member(self, member: ConsumerMember) -> None:
        """Add member to group.
        
        Args:
            member: Consumer member to add.
        """
        self.members[member.consumer_id] = member
    
    def remove_member(self, consumer_id: str) -> None:
        """Remove member from group.
        
        Args:
            consumer_id: Consumer to remove.
        """
        self.members.pop(consumer_id, None)
    
    def get_active_members(self) -> list[ConsumerMember]:
        """Get active members (alive and stable).
        
        Returns:
            List of active members.
        """
        return [m for m in self.members.values() if m.is_alive()]
    
    def start_rebalance(self) -> None:
        """Begin rebalancing process."""
        self.state = ConsumerState.REBALANCING
        self.generation += 1
        if self.members:
            self.leader = list(self.members.keys())[0]
    
    def complete_rebalance(self) -> None:
        """Complete rebalancing."""
        self.state = ConsumerState.STABLE
    
    def size(self) -> int:
        """Get group size.
        
        Returns:
            Number of members.
        """
        return len(self.members)


@dataclass
class ConsumerOffset:
    """Committed offset for a consumer group partition."""
    group_id: str
    partition_id: int
    offset: int  # Committed offset
    timestamp: float = field(default_factory=time.time)
    metadata: str = ""  # Arbitrary metadata
    
    def advance(self, new_offset: int) -> None:
        """Advance offset.
        
        Args:
            new_offset: New committed offset.
        """
        if new_offset > self.offset:
            self.offset = new_offset
            self.timestamp = time.time()


@dataclass
class PartitionAssignment:
    """Partition assignment for rebalancing."""
    partition_id: int
    consumer_id: str  # Consumer assigned to partition
    generation: int  # Generation when assigned
    lag_records: int = 0  # How far consumer is behind
    
    def update_lag(self, current_offset: int, hw: int) -> None:
        """Update consumer lag.
        
        Args:
            current_offset: Consumer's current offset.
            hw: High water mark (latest offset).
        """
        self.lag_records = max(0, hw - current_offset)
