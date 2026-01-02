"""Streaming system domain layer."""

from streaming_system.domain.entities.record import (
    Record,
    LogEntry,
    LogSegment,
    HighWaterMark,
    RecordCompressionType,
)
from streaming_system.domain.entities.raft import (
    RaftStateEnum,
    RaftTerm,
    InSyncReplicas,
    RaftNode,
)
from streaming_system.domain.entities.consumer import (
    ConsumerMember,
    ConsumerGroup,
    ConsumerOffset,
    PartitionAssignment,
    ConsumerState,
)
from streaming_system.domain.services.partition_log import PartitionLog
from streaming_system.domain.services.raft_consensus import RaftConsensus
from streaming_system.domain.services.consumer_coordinator import ConsumerCoordinator
from streaming_system.domain.value_objects.identifiers import (
    NodeId,
    TopicName,
    PartitionId,
    ConsumerId,
    ConsumerGroupId,
    Offset,
    Term,
    create_node_id,
    create_consumer_id,
    create_consumer_group_id,
)

__all__ = [
    # Value objects
    "NodeId",
    "TopicName",
    "PartitionId",
    "ConsumerId",
    "ConsumerGroupId",
    "Offset",
    "Term",
    "create_node_id",
    "create_consumer_id",
    "create_consumer_group_id",
    # Entities
    "Record",
    "LogEntry",
    "LogSegment",
    "HighWaterMark",
    "RecordCompressionType",
    "RaftStateEnum",
    "RaftTerm",
    "InSyncReplicas",
    "RaftNode",
    "ConsumerMember",
    "ConsumerGroup",
    "ConsumerOffset",
    "PartitionAssignment",
    "ConsumerState",
    # Services
    "PartitionLog",
    "RaftConsensus",
    "ConsumerCoordinator",
]
