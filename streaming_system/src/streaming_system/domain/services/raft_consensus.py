"""Raft consensus service for leader election and replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random
import time
from streaming_system.domain.entities.raft import (
    RaftStateEnum,
    RaftNode,
    RaftTerm,
    InSyncReplicas,
)


@dataclass
class LogEntry:
    """Single entry in the Raft log."""
    index: int
    term: int
    data: bytes = b""


class RaftConsensus:
    """Raft consensus implementation for a partition.
    
    Handles:
    - Leader election
    - Log replication
    - In-sync replica tracking
    - Commit index advancement
    """
    
    def __init__(self, node_id: str, cluster_size: int):
        """Initialize Raft node.

        Args:
            node_id: This node's ID.
            cluster_size: Total nodes in cluster.
        """
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.node = RaftNode(node_id=node_id)
        self.isr = InSyncReplicas(members={node_id})
        self.commit_index = 0
        self.last_applied = 0
        self.match_index: dict[str, int] = {node_id: 0}  # Replication progress
        self.next_index: dict[str, int] = {node_id: 1}   # Next entry to send

        # Log storage for consistency checks
        self._log: list[LogEntry] = []

        # Election randomization
        self._election_timeout = random.randint(150, 300) / 1000.0  # seconds
        self._last_heartbeat = time.time()

    def get_last_log_index(self) -> int:
        """Get index of last log entry.

        Returns:
            Last log index (0 if log is empty).
        """
        return len(self._log)

    def get_last_log_term(self) -> int:
        """Get term of last log entry.

        Returns:
            Term of last entry (0 if log is empty).
        """
        if not self._log:
            return 0
        return self._log[-1].term

    def get_log_term(self, index: int) -> int:
        """Get term of log entry at given index.

        Args:
            index: Log index (1-based).

        Returns:
            Term at index, or 0 if index is 0 or invalid.
        """
        if index <= 0 or index > len(self._log):
            return 0
        return self._log[index - 1].term

    def append_log_entry(self, term: int, data: bytes = b"") -> int:
        """Append new entry to log (leader only).

        Args:
            term: Current term.
            data: Entry data.

        Returns:
            Index of new entry.
        """
        index = len(self._log) + 1
        self._log.append(LogEntry(index=index, term=term, data=data))
        self.match_index[self.node_id] = index
        return index
    
    def request_vote(
        self,
        term: int,
        candidate_id: str,
        last_log_index: int,
        last_log_term: int,
    ) -> bool:
        """Handle vote request from candidate.

        Per Raft paper Section 5.4.1: Vote is granted only if candidate's log
        is at least as up-to-date as voter's log.

        Args:
            term: Candidate's term.
            candidate_id: Candidate node ID.
            last_log_index: Candidate's last log index.
            last_log_term: Candidate's last log term.

        Returns:
            True if we grant vote.
        """
        # If candidate's term is older, reject
        if term < self.node.current_term.get():
            return False

        # If we see newer term, update and step down
        if term > self.node.current_term.get():
            self.node.become_follower(term)

        # Check if candidate's log is at least as up-to-date as ours
        # Raft paper: "If the logs have last entries with different terms,
        # the log with the later term is more up-to-date. If the logs end
        # with the same term, the longer log is more up-to-date."
        my_last_term = self.get_last_log_term()
        my_last_index = self.get_last_log_index()

        candidate_log_ok = (
            last_log_term > my_last_term or
            (last_log_term == my_last_term and last_log_index >= my_last_index)
        )

        if not candidate_log_ok:
            return False

        # Grant vote if: haven't voted yet OR already voted for this candidate
        if self.node.voted_for is None or self.node.voted_for == candidate_id:
            self.node.voted_for = candidate_id
            self._last_heartbeat = time.time()
            return True

        return False
    
    def append_entries(
        self,
        term: int,
        leader_id: str,
        prev_log_index: int,
        prev_log_term: int,
        entries: Optional[list[LogEntry]] = None,
        leader_commit: int = 0,
    ) -> bool:
        """Handle append entries from leader (heartbeat or replication).

        Per Raft paper Section 5.3: Follower must verify log consistency
        before accepting new entries.

        Args:
            term: Leader's term.
            leader_id: Leader node ID.
            prev_log_index: Index of entry before new entries.
            prev_log_term: Term of entry at prev_log_index.
            entries: Entries to replicate.
            leader_commit: Leader's commit index.

        Returns:
            True if append succeeded.
        """
        if entries is None:
            entries = []

        # If leader's term is older, reject
        if term < self.node.current_term.get():
            return False

        # If we see newer term, update and step down
        if term > self.node.current_term.get():
            self.node.become_follower(term)

        # Update leader and heartbeat
        self.node.leader_id = leader_id
        self._last_heartbeat = time.time()
        self.node.reset_election_timer()

        # Log consistency check (Raft paper Section 5.3)
        # If prev_log_index > 0, we must have an entry at that index with matching term
        if prev_log_index > 0:
            if prev_log_index > len(self._log):
                # We don't have enough entries - reject
                return False
            if self.get_log_term(prev_log_index) != prev_log_term:
                # Term mismatch at prev_log_index - reject and truncate
                # Delete conflicting entries and all that follow
                self._log = self._log[:prev_log_index - 1]
                return False

        # Append new entries (if any)
        if entries:
            # Start appending from prev_log_index + 1
            for i, entry in enumerate(entries):
                new_index = prev_log_index + 1 + i
                if new_index <= len(self._log):
                    # Entry exists - check for conflict
                    if self._log[new_index - 1].term != entry.term:
                        # Conflict: delete this and all following entries
                        self._log = self._log[:new_index - 1]
                        self._log.append(entry)
                    # else: entry matches, skip
                else:
                    # Append new entry
                    self._log.append(entry)

        # Update commit index if leader's is higher
        if leader_commit > self.commit_index:
            # Commit index = min(leader_commit, index of last new entry)
            last_new_entry = prev_log_index + len(entries) if entries else prev_log_index
            self.commit_index = min(leader_commit, max(last_new_entry, len(self._log)))

        return True
    
    def check_election_timeout(self) -> bool:
        """Check if election timeout has elapsed.
        
        Returns:
            True if timeout exceeded.
        """
        if self.node.state == RaftStateEnum.LEADER:
            return False  # Leaders don't elect
        
        elapsed = time.time() - self._last_heartbeat
        return elapsed > self._election_timeout
    
    def start_election(self) -> None:
        """Start leader election process."""
        self.node.become_candidate()
    
    def win_election(self) -> None:
        """Become leader (after winning election).

        Per Raft paper: Initialize next_index to leader's last log index + 1.
        """
        self.node.become_leader()

        # Initialize next_index for all followers to last log index + 1
        # (leader will decrement if AppendEntries fails)
        last_log_index = self.get_last_log_index()
        for peer in range(self.cluster_size):
            peer_id = f"node-{peer}"
            if peer_id != self.node_id:
                self.next_index[peer_id] = last_log_index + 1
                self.match_index[peer_id] = 0
    
    def get_state(self) -> str:
        """Get current node state.
        
        Returns:
            State name: "follower", "candidate", or "leader".
        """
        return self.node.state.value
    
    def get_current_term(self) -> int:
        """Get current term.
        
        Returns:
            Current term value.
        """
        return self.node.current_term.get()
    
    def get_leader(self) -> Optional[str]:
        """Get current leader ID.
        
        Returns:
            Leader ID or None if no leader.
        """
        return self.node.leader_id
    
    def advance_commit_index(self, quorum_match_index: int) -> None:
        """Advance commit index when quorum has replicated.

        Per Raft paper Section 5.4.2: Leader can only commit entries from
        its current term. Entries from previous terms are committed indirectly.

        Args:
            quorum_match_index: Highest index replicated on quorum.
        """
        if quorum_match_index <= self.commit_index:
            return

        # Only leaders can advance commit index
        if self.node.state != RaftStateEnum.LEADER:
            return

        # Verify the entry at quorum_match_index is from current term
        entry_term = self.get_log_term(quorum_match_index)
        if entry_term == self.node.current_term.get():
            self.commit_index = quorum_match_index
    
    def get_isr(self) -> list[str]:
        """Get in-sync replicas.
        
        Returns:
            List of ISR node IDs.
        """
        return list(self.isr.members)
    
    def add_to_isr(self, node_id: str) -> None:
        """Add node to ISR.
        
        Args:
            node_id: Node to add.
        """
        self.isr.add(node_id)
    
    def remove_from_isr(self, node_id: str) -> None:
        """Remove node from ISR (lagging).
        
        Args:
            node_id: Node to remove.
        """
        self.isr.remove(node_id)
