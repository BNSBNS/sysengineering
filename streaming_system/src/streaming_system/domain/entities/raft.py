"""Raft consensus implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional


class RaftStateEnum(Enum):
    """Raft node state."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class VoteResponse(Enum):
    """Response to vote request."""
    GRANTED = "granted"
    DENIED = "denied"


class AppendResponse(Enum):
    """Response to append entries request."""
    SUCCESS = "success"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class RaftTerm:
    """Logical clock for Raft consensus.
    
    Terms increase with each election and prevent stale leaders
    from making decisions.
    """
    value: int = 0
    
    def increment(self) -> None:
        """Increment to next term."""
        self.value += 1
    
    def update(self, new_term: int) -> bool:
        """Update term to higher value.
        
        Args:
            new_term: Proposed new term.
            
        Returns:
            True if term was updated.
        """
        if new_term > self.value:
            self.value = new_term
            return True
        return False
    
    def get(self) -> int:
        """Get current term.
        
        Returns:
            Current term value.
        """
        return self.value


@dataclass
class RaftVoteData:
    """Vote record for leader election."""
    term: int
    voted_for: Optional[str] = None  # Node ID we voted for
    timestamp: float = field(default_factory=time.time)


@dataclass
class RaftNodeState:
    """Persistent Raft state on every node."""
    current_term: RaftTerm = field(default_factory=RaftTerm)
    voted_for: Optional[str] = None  # Voted for in current term
    vote_history: dict[int, str] = field(default_factory=dict)  # term → node_id
    
    def vote_for(self, term: int, candidate: str) -> bool:
        """Record a vote for a candidate.
        
        Args:
            term: Election term.
            candidate: Candidate node ID.
            
        Returns:
            True if vote was recorded.
        """
        # Can only vote once per term
        if term in self.vote_history:
            return self.vote_history[term] == candidate
        
        self.vote_history[term] = candidate
        return True


@dataclass
class InSyncReplicas:
    """In-sync replicas for durability guarantees.
    
    Only replicas in ISR can become leader, ensuring no data loss.
    """
    members: set[str] = field(default_factory=set)  # Node IDs in ISR
    
    def add(self, node_id: str) -> None:
        """Add node to ISR.
        
        Args:
            node_id: Node to add.
        """
        self.members.add(node_id)
    
    def remove(self, node_id: str) -> None:
        """Remove node from ISR (e.g., lagging replica).
        
        Args:
            node_id: Node to remove.
        """
        self.members.discard(node_id)
    
    def contains(self, node_id: str) -> bool:
        """Check if node is in ISR.
        
        Args:
            node_id: Node to check.
            
        Returns:
            True if node is in ISR.
        """
        return node_id in self.members
    
    def is_quorum(self, ack_count: int) -> bool:
        """Check if acks form a quorum.
        
        Quorum = majority of ISR members.
        
        Args:
            ack_count: Number of acks received.
            
        Returns:
            True if quorum is achieved.
        """
        return ack_count >= (len(self.members) // 2) + 1
    
    def size(self) -> int:
        """Get ISR size.
        
        Returns:
            Number of members in ISR.
        """
        return len(self.members)


@dataclass
class RaftNode:
    """Raft consensus node state machine."""
    node_id: str
    state: RaftStateEnum = field(default=RaftStateEnum.FOLLOWER)
    current_term: RaftTerm = field(default_factory=RaftTerm)
    voted_for: Optional[str] = None
    leader_id: Optional[str] = None
    election_timeout_ms: int = 150  # 150-300ms election window
    heartbeat_interval_ms: int = 50  # Send heartbeats every 50ms
    last_heartbeat: float = field(default_factory=time.time)
    
    def become_follower(self, term: int) -> None:
        """Transition to follower state.
        
        Args:
            term: Current term.
        """
        self.state = RaftStateEnum.FOLLOWER
        self.current_term.update(term)
        self.voted_for = None
        self.last_heartbeat = time.time()
    
    def become_candidate(self) -> None:
        """Transition to candidate state (election)."""
        self.current_term.increment()
        self.state = RaftStateEnum.CANDIDATE
        self.voted_for = self.node_id  # Vote for self
    
    def become_leader(self) -> None:
        """Transition to leader state."""
        self.state = RaftStateEnum.LEADER
        self.leader_id = self.node_id
        self.last_heartbeat = time.time()
    
    def is_election_timeout(self) -> bool:
        """Check if election timeout has elapsed.
        
        Returns:
            True if timeout exceeded.
        """
        elapsed_ms = (time.time() - self.last_heartbeat) * 1000
        return elapsed_ms > self.election_timeout_ms
    
    def reset_election_timer(self) -> None:
        """Reset election timeout timer."""
        self.last_heartbeat = time.time()
