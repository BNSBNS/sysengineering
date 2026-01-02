"""Inbound ports - API contracts for the streaming system.

Inbound ports define the interfaces that clients and upper layers
use to interact with partition logs, Raft consensus, and consumer coordination.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from streaming_system.domain.entities.record import Record, LogEntry


# =============================================================================
# Partition Log Port
# =============================================================================


@dataclass
class PartitionLogStats:
    """Statistics for partition log monitoring."""

    partition_id: int
    total_records: int
    committed_offset: int
    segments: int
    current_term: int


class PartitionLogPort(Protocol):
    """Protocol for partition log operations.

    Implements an append-only segmented log for message storage.

    Thread Safety:
        All methods must be thread-safe.

    Durability:
        Records are durable after append returns.

    Example:
        offsets = log.append([record1, record2])
        entries = log.read(offset=0, max_records=100)
        log.commit(offsets[-1])  # Advance HWM
    """

    @property
    @abstractmethod
    def partition_id(self) -> int:
        """Return the partition ID."""
        ...

    @abstractmethod
    def append(self, records: list[Record]) -> list[int]:
        """Append records to the log.

        Args:
            records: Records to append.

        Returns:
            List of assigned offsets.
        """
        ...

    @abstractmethod
    def read(self, offset: int, max_records: int = 100) -> list[LogEntry]:
        """Read records from the log.

        Args:
            offset: Starting offset.
            max_records: Maximum records to return.

        Returns:
            List of log entries.
        """
        ...

    @abstractmethod
    def commit(self, offset: int) -> None:
        """Commit entries up to offset (advance HWM).

        Args:
            offset: Offset to commit to.
        """
        ...

    @abstractmethod
    def truncate(self, offset: int) -> None:
        """Truncate log at offset (for Raft recovery).

        Args:
            offset: Truncate at this offset (exclusive).
        """
        ...

    @abstractmethod
    def get_hwm(self) -> int:
        """Get high water mark (latest committed offset).

        Returns:
            HWM offset.
        """
        ...

    @abstractmethod
    def get_next_offset(self) -> int:
        """Get next available offset.

        Returns:
            Next offset to be assigned.
        """
        ...

    @abstractmethod
    def set_term(self, term: int) -> None:
        """Set current Raft term.

        Args:
            term: Raft term.
        """
        ...

    @abstractmethod
    def get_stats(self) -> PartitionLogStats:
        """Get log statistics.

        Returns:
            Partition log statistics.
        """
        ...


# =============================================================================
# Raft Consensus Port
# =============================================================================


@dataclass
class RaftStats:
    """Statistics for Raft consensus monitoring."""

    node_id: str
    state: str  # "follower", "candidate", "leader"
    current_term: int
    commit_index: int
    last_applied: int
    cluster_size: int
    isr_count: int


class RaftConsensusPort(Protocol):
    """Protocol for Raft consensus operations.

    Implements Raft consensus algorithm for leader election and log replication.

    Thread Safety:
        All methods must be thread-safe.

    References:
        - Raft Paper (Ongaro & Ousterhout, 2014)

    Example:
        # Handle vote request from candidate
        granted = raft.request_vote(candidate_id, term, last_log_index, last_log_term)

        # As leader, append and replicate
        index = raft.append_log_entry(current_term, data)
        raft.append_entries(follower_id, ...)  # Send to followers
    """

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Return this node's ID."""
        ...

    @abstractmethod
    def request_vote(
        self,
        candidate_id: str,
        term: int,
        last_log_index: int,
        last_log_term: int,
    ) -> bool:
        """Handle vote request from a candidate.

        Args:
            candidate_id: Candidate requesting vote.
            term: Candidate's term.
            last_log_index: Candidate's last log index.
            last_log_term: Candidate's last log term.

        Returns:
            True if vote granted.
        """
        ...

    @abstractmethod
    def append_entries(
        self,
        leader_id: str,
        term: int,
        prev_log_index: int,
        prev_log_term: int,
        entries: list[tuple[int, bytes]],
        leader_commit: int,
    ) -> tuple[bool, int]:
        """Handle append entries from leader.

        Args:
            leader_id: Leader sending entries.
            term: Leader's term.
            prev_log_index: Index before new entries.
            prev_log_term: Term at prev_log_index.
            entries: List of (term, data) tuples.
            leader_commit: Leader's commit index.

        Returns:
            Tuple of (success, match_index).
        """
        ...

    @abstractmethod
    def check_election_timeout(self) -> bool:
        """Check if election timeout has elapsed.

        Returns:
            True if timeout elapsed and election should start.
        """
        ...

    @abstractmethod
    def start_election(self) -> None:
        """Start a new election (become candidate)."""
        ...

    @abstractmethod
    def win_election(self) -> None:
        """Win election and become leader."""
        ...

    @abstractmethod
    def get_last_log_index(self) -> int:
        """Get index of last log entry.

        Returns:
            Last log index (0 if log is empty).
        """
        ...

    @abstractmethod
    def get_stats(self) -> RaftStats:
        """Get Raft consensus statistics.

        Returns:
            Raft statistics.
        """
        ...


# =============================================================================
# Consumer Coordinator Port
# =============================================================================


@dataclass
class ConsumerGroupStats:
    """Statistics for consumer group."""

    group_id: str
    member_count: int
    partition_count: int
    total_lag: int


@dataclass
class CoordinatorStats:
    """Statistics for consumer coordinator."""

    total_groups: int
    total_members: int
    total_partitions: int


class ConsumerCoordinatorPort(Protocol):
    """Protocol for consumer group coordination.

    Manages consumer groups, partition assignment, and offset tracking.

    Thread Safety:
        All methods must be thread-safe.

    Example:
        group_id = coordinator.create_group("my-group")
        coordinator.join_group(group_id, consumer_id)
        coordinator.assign_partitions(group_id, [0, 1, 2])
        coordinator.commit_offset(group_id, partition=0, offset=100)
    """

    @abstractmethod
    def create_group(self, group_id: str) -> str:
        """Create a new consumer group.

        Args:
            group_id: Group identifier.

        Returns:
            Group ID.
        """
        ...

    @abstractmethod
    def join_group(self, group_id: str, consumer_id: str) -> bool:
        """Join a consumer group.

        Args:
            group_id: Group to join.
            consumer_id: Consumer identifier.

        Returns:
            True if joined successfully.
        """
        ...

    @abstractmethod
    def leave_group(self, group_id: str, consumer_id: str) -> bool:
        """Leave a consumer group.

        Args:
            group_id: Group to leave.
            consumer_id: Consumer identifier.

        Returns:
            True if left successfully.
        """
        ...

    @abstractmethod
    def heartbeat(self, group_id: str, consumer_id: str) -> bool:
        """Send heartbeat to coordinator.

        Args:
            group_id: Consumer's group.
            consumer_id: Consumer identifier.

        Returns:
            True if heartbeat accepted.
        """
        ...

    @abstractmethod
    def commit_offset(self, group_id: str, partition: int, offset: int) -> None:
        """Commit consumer offset.

        Args:
            group_id: Consumer's group.
            partition: Partition number.
            offset: Offset to commit.
        """
        ...

    @abstractmethod
    def get_committed_offset(self, group_id: str, partition: int) -> int:
        """Get committed offset for partition.

        Args:
            group_id: Consumer's group.
            partition: Partition number.

        Returns:
            Committed offset.
        """
        ...

    @abstractmethod
    def assign_partitions(self, group_id: str, partitions: list[int]) -> dict[str, list[int]]:
        """Assign partitions to group members.

        Args:
            group_id: Group to assign partitions to.
            partitions: Partitions to assign.

        Returns:
            Dict mapping consumer_id -> list of partitions.
        """
        ...

    @abstractmethod
    def get_stats(self) -> CoordinatorStats:
        """Get coordinator statistics.

        Returns:
            Coordinator statistics.
        """
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Partition Log
    "PartitionLogPort",
    "PartitionLogStats",
    # Raft Consensus
    "RaftConsensusPort",
    "RaftStats",
    # Consumer Coordinator
    "ConsumerCoordinatorPort",
    "ConsumerGroupStats",
    "CoordinatorStats",
]
