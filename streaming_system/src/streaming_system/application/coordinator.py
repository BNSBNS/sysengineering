"""Application coordinator for streaming system.

Orchestrates broker operations: partition leadership, replication,
consumer coordination, and log management.

References:
    - design.md Section 2 (Architecture)
    - Kafka broker design
"""

from __future__ import annotations

import logging
from typing import Optional
from streaming_system.domain.entities.record import Record
from streaming_system.domain.services.partition_log import PartitionLog
from streaming_system.domain.services.raft_consensus import RaftConsensus
from streaming_system.domain.services.consumer_coordinator import ConsumerCoordinator

logger = logging.getLogger(__name__)


class BrokerCoordinator:
    """Coordinates broker operations for a partition.
    
    Manages:
    - Partition log (append-only log)
    - Raft consensus (leader election, replication)
    - Consumer coordination (groups, offsets)
    """
    
    def __init__(self, node_id: str, partition_id: int, cluster_size: int):
        """Initialize broker coordinator.
        
        Args:
            node_id: This node's ID.
            partition_id: Partition number.
            cluster_size: Total nodes in cluster.
        """
        self.node_id = node_id
        self.partition_id = partition_id
        self.cluster_size = cluster_size
        
        # Services
        self._log = PartitionLog(partition_id)
        self._raft = RaftConsensus(node_id, cluster_size)
        self._coordinator = ConsumerCoordinator()
        
        # Metrics
        self._metrics = {
            "messages_produced": 0,
            "messages_consumed": 0,
            "leader_elections": 0,
            "rebalances": 0,
        }
    
    def initialize(self):
        """Initialize broker."""
        logger.info(f"Initializing broker for partition {self.partition_id}")
    
    def produce(self, records: list[Record], acks: int = 1) -> list[int]:
        """Produce records to partition.
        
        Args:
            records: Records to produce.
            acks: Acknowledgment level:
                  0 = fire-and-forget
                  1 = leader ack only
                  -1 = all ISR replicas ack
                  
        Returns:
            List of assigned offsets.
        """
        if self._raft.get_state() != "leader":
            raise Exception("Not leader")
        
        # Append to log
        offsets = self._log.append(records)
        
        # For acks=-1, wait for replication (simplified)
        if acks == -1:
            # In real impl, would wait for ISR to replicate
            for offset in offsets:
                self._log.commit(offset)
        elif acks == 1:
            # Leader ack: commit immediately
            for offset in offsets:
                self._log.commit(offset)
        # acks=0: don't commit
        
        self._metrics["messages_produced"] += len(records)
        
        return offsets
    
    def consume(
        self,
        group_id: str,
        partition_id: int,
        offset: int,
        max_records: int = 100,
    ) -> list:
        """Consume records from partition.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition to consume from.
            offset: Starting offset.
            max_records: Max records to return.
            
        Returns:
            List of records.
        """
        # Get records from log
        entries = self._log.read(offset, max_records)
        records = [entry.record for entry in entries]
        
        self._metrics["messages_consumed"] += len(records)
        
        return records
    
    def join_consumer_group(
        self,
        group_id: str,
        consumer_id: str,
        session_timeout_ms: int = 10000,
    ):
        """Consumer joins group (triggers rebalance).
        
        Args:
            group_id: Consumer group ID.
            consumer_id: Consumer ID.
            session_timeout_ms: Session timeout in ms.
        """
        self._coordinator.join_group(group_id, consumer_id, session_timeout_ms)
        self._metrics["rebalances"] += 1
    
    def commit_offset(self, group_id: str, partition_id: int, offset: int) -> None:
        """Commit consumer offset.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition ID.
            offset: Offset to commit.
        """
        self._coordinator.commit_offset(group_id, partition_id, offset)
    
    def get_offset(self, group_id: str, partition_id: int) -> int:
        """Get committed offset for consumer group partition.
        
        Args:
            group_id: Consumer group ID.
            partition_id: Partition ID.
            
        Returns:
            Committed offset.
        """
        return self._coordinator.get_offset(group_id, partition_id)
    
    def check_election(self) -> bool:
        """Check if election timeout and trigger if needed.
        
        Returns:
            True if election started.
        """
        if self._raft.check_election_timeout():
            self._raft.start_election()
            self._metrics["leader_elections"] += 1
            return True
        return False
    
    def replicate_entries(self, term: int, prev_log_index: int, entries: list = None):
        """Handle replication from leader.
        
        Args:
            term: Leader's term.
            prev_log_index: Index of entry before new entries.
            entries: Entries to replicate.
        """
        if entries is None:
            entries = []
        
        result = self._raft.append_entries(term, "", prev_log_index, entries)
        return result
    
    def get_leader(self) -> Optional[str]:
        """Get current leader.
        
        Returns:
            Leader node ID or None.
        """
        return self._raft.get_leader()
    
    def get_state(self) -> str:
        """Get broker state.
        
        Returns:
            State: "leader", "candidate", or "follower".
        """
        return self._raft.get_state()
    
    def get_hwm(self) -> int:
        """Get high water mark (committed offset).
        
        Returns:
            HWM offset.
        """
        return self._log.get_hwm()
    
    def get_next_offset(self) -> int:
        """Get next available offset.
        
        Returns:
            Next offset.
        """
        return self._log.get_next_offset()
    
    def get_metrics(self) -> dict:
        """Get broker metrics.
        
        Returns:
            Dictionary with metrics.
        """
        return {
            **self._metrics,
            "partition_id": self.partition_id,
            "state": self.get_state(),
            "hwm": self.get_hwm(),
            "next_offset": self.get_next_offset(),
            "leader": self.get_leader(),
        }
