"""Consumer group coordinator service."""

from __future__ import annotations

from typing import Optional
from streaming_system.domain.entities.consumer import (
    ConsumerGroup,
    ConsumerMember,
    ConsumerOffset,
    PartitionAssignment,
)


class ConsumerCoordinator:
    """Coordinates consumer groups and partition assignments.
    
    Handles:
    - Consumer group membership
    - Partition rebalancing
    - Offset management
    - Consumer lag tracking
    """
    
    def __init__(self):
        """Initialize consumer coordinator."""
        self._groups: dict[str, ConsumerGroup] = {}
        self._offsets: dict[tuple[str, int], ConsumerOffset] = {}  # (group, partition) -> offset
        self._assignments: dict[tuple[str, int], PartitionAssignment] = {}  # (group, partition) -> assignment
    
    def create_group(self, group_id: str, protocol_type: str = "range") -> ConsumerGroup:
        """Create new consumer group.
        
        Args:
            group_id: Group ID.
            protocol_type: Assignment strategy ("range", "roundrobin", "sticky").
            
        Returns:
            Created group.
        """
        group = ConsumerGroup(
            group_id=group_id,
            protocol_type=protocol_type,
        )
        self._groups[group_id] = group
        return group
    
    def get_group(self, group_id: str) -> Optional[ConsumerGroup]:
        """Get consumer group.
        
        Args:
            group_id: Group ID.
            
        Returns:
            Consumer group or None.
        """
        return self._groups.get(group_id)
    
    def join_group(
        self,
        group_id: str,
        consumer_id: str,
        session_timeout_ms: int = 10000,
    ) -> ConsumerGroup:
        """Join consumer to group (triggers rebalance).
        
        Args:
            group_id: Group ID.
            consumer_id: Consumer ID.
            session_timeout_ms: Session timeout in ms.
            
        Returns:
            Updated group.
        """
        group = self._groups.get(group_id)
        if not group:
            group = self.create_group(group_id)
        
        member = ConsumerMember(
            consumer_id=consumer_id,
            group_id=group_id,
            session_timeout_ms=session_timeout_ms,
        )
        
        group.add_member(member)
        group.start_rebalance()
        
        return group
    
    def leave_group(self, group_id: str, consumer_id: str) -> Optional[ConsumerGroup]:
        """Remove consumer from group.
        
        Args:
            group_id: Group ID.
            consumer_id: Consumer ID.
            
        Returns:
            Updated group or None.
        """
        group = self._groups.get(group_id)
        if not group:
            return None
        
        group.remove_member(consumer_id)
        
        # Trigger rebalance if group still has members
        if group.size() > 0:
            group.start_rebalance()
        else:
            self._groups.pop(group_id, None)
        
        return group
    
    def heartbeat(self, group_id: str, consumer_id: str) -> bool:
        """Consumer heartbeat (keep alive).
        
        Args:
            group_id: Group ID.
            consumer_id: Consumer ID.
            
        Returns:
            True if heartbeat accepted.
        """
        group = self._groups.get(group_id)
        if not group:
            return False
        
        member = group.members.get(consumer_id)
        if not member:
            return False
        
        member.update_heartbeat()
        return True
    
    def commit_offset(
        self,
        group_id: str,
        partition_id: int,
        offset: int,
    ) -> None:
        """Commit consumer offset.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition ID.
            offset: Committed offset.
        """
        key = (group_id, partition_id)
        
        if key not in self._offsets:
            self._offsets[key] = ConsumerOffset(
                group_id=group_id,
                partition_id=partition_id,
                offset=offset,
            )
        else:
            self._offsets[key].advance(offset)
    
    def get_offset(self, group_id: str, partition_id: int) -> int:
        """Get committed offset for partition.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition ID.
            
        Returns:
            Committed offset (0 if not found).
        """
        key = (group_id, partition_id)
        offset_obj = self._offsets.get(key)
        return offset_obj.offset if offset_obj else 0
    
    def assign_partitions(
        self,
        group_id: str,
        partition_count: int,
    ) -> dict[str, list[int]]:
        """Assign partitions to consumers (range strategy).
        
        Args:
            group_id: Group ID.
            partition_count: Total partitions in topic.
            
        Returns:
            Dict mapping consumer_id -> list of partition IDs.
        """
        group = self._groups.get(group_id)
        if not group:
            return {}
        
        assignment = {}
        consumers = list(group.members.keys())
        
        if not consumers:
            return assignment
        
        # Range strategy: consecutive partitions per consumer
        partitions_per_consumer = partition_count // len(consumers)
        remainder = partition_count % len(consumers)
        
        for i, consumer_id in enumerate(consumers):
            start = i * partitions_per_consumer + min(i, remainder)
            end = start + partitions_per_consumer + (1 if i < remainder else 0)
            
            assigned = list(range(start, end))
            assignment[consumer_id] = assigned
            group.members[consumer_id].assigned_partitions = assigned
        
        return assignment
    
    def update_consumer_lag(
        self,
        group_id: str,
        partition_id: int,
        current_offset: int,
        hwm: int,
    ) -> None:
        """Update consumer lag for partition.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition ID.
            current_offset: Consumer's current offset.
            hwm: Partition high water mark.
        """
        key = (group_id, partition_id)
        
        if key not in self._assignments:
            # Find which consumer has this partition
            for consumer in self._groups.get(group_id, ConsumerGroup("")).members.values():
                if partition_id in consumer.assigned_partitions:
                    self._assignments[key] = PartitionAssignment(
                        partition_id=partition_id,
                        consumer_id=consumer.consumer_id,
                        generation=self._groups.get(group_id, ConsumerGroup("")).generation,
                    )
                    break
        
        assignment = self._assignments.get(key)
        if assignment:
            assignment.update_lag(current_offset, hwm)
