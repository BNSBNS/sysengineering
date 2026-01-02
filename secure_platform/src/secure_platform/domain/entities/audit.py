"""Audit log entities with Merkle tree tamper-evidence.

Implements append-only audit log with cryptographic proof of integrity,
following Certificate Transparency (RFC 6962) patterns.

References:
    - design.md Section 3 (Audit Logger)
    - RFC 6962 (Certificate Transparency)
    - Merkle, R. "A Certified Digital Signature" CRYPTO 1989
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import json


class AuditEventType(Enum):
    """Type of audit event."""
    CERT_ISSUED = "cert_issued"
    CERT_REVOKED = "cert_revoked"
    CERT_ROTATED = "cert_rotated"
    AUTH_ALLOWED = "auth_allowed"
    AUTH_DENIED = "auth_denied"
    POLICY_CREATED = "policy_created"
    POLICY_UPDATED = "policy_updated"
    POLICY_DELETED = "policy_deleted"
    PRINCIPAL_CREATED = "principal_created"
    PRINCIPAL_MODIFIED = "principal_modified"


@dataclass
class AuditEvent:
    """Single audit log entry."""
    event_id: str
    event_type: AuditEventType
    principal_id: str  # Who performed the action
    action: str  # What they did
    resource_id: str  # What it affected
    result: str  # "success" or "failure"
    details: dict = field(default_factory=dict)  # Additional context
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_bytes(self) -> bytes:
        """Serialize event to bytes for hashing.
        
        Returns:
            JSON bytes representation.
        """
        event_dict = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'principal_id': self.principal_id,
            'action': self.action,
            'resource_id': self.resource_id,
            'result': self.result,
            'details': self.details,
            'timestamp': self.timestamp,
        }
        return json.dumps(event_dict, sort_keys=True).encode('utf-8')
    
    def hash(self) -> str:
        """Compute SHA-256 hash of event.
        
        Returns:
            Hex-encoded hash.
        """
        return hashlib.sha256(self.to_bytes()).hexdigest()


@dataclass
class MerkleNode:
    """Single node in Merkle tree.
    
    A Merkle tree node is either:
    - Leaf: hash(event)
    - Internal: hash(left_hash + right_hash)
    """
    hash_value: str
    level: int  # 0 = leaf, >0 = internal node
    left_child: Optional[MerkleNode] = None
    right_child: Optional[MerkleNode] = None
    event_id: Optional[str] = None  # Only set for leaf nodes


@dataclass
class MerkleProof:
    """Proof that an event is in the Merkle tree.
    
    To verify an event is in the tree:
    1. Hash the event
    2. Apply proof hashes up the tree
    3. Verify final hash matches tree root
    """
    event_id: str
    leaf_hash: str
    sibling_hashes: list[str]  # Hashes needed to reconstruct root
    leaf_index: int  # Position in leaf array
    tree_size: int  # Total leaves in tree


class MerkleTree:
    """Append-only Merkle tree for audit log integrity."""
    
    def __init__(self):
        """Initialize empty Merkle tree."""
        self._leaves: list[MerkleNode] = []  # Leaf nodes
        self._root: Optional[MerkleNode] = None
        self._size = 0
    
    def append(self, event: AuditEvent) -> None:
        """Append event to tree.
        
        Args:
            event: Audit event to append.
        """
        # Create leaf node
        leaf = MerkleNode(
            hash_value=event.hash(),
            level=0,
            event_id=event.event_id,
        )
        self._leaves.append(leaf)
        self._size += 1
        
        # Rebuild tree
        self._rebuild_tree()
    
    def _rebuild_tree(self) -> None:
        """Rebuild tree from leaves (after append).
        
        Merkle tree structure:
                       Root
                      /    \
                   H01      H23
                  /  \      /  \
                 H0  H1   H2   H3
                 │   │    │    │
                L0  L1   L2   L3  (Leaves)
        """
        if not self._leaves:
            self._root = None
            return
        
        # Start with leaves
        current_level = self._leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            # Pair up nodes and hash
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Parent hash = hash(left || right)
                parent_hash = self._hash_pair(left.hash_value, right.hash_value)
                parent = MerkleNode(
                    hash_value=parent_hash,
                    level=left.level + 1,
                    left_child=left,
                    right_child=right,
                )
                next_level.append(parent)
            
            current_level = next_level
        
        self._root = current_level[0] if current_level else None
    
    def _hash_pair(self, left_hash: str, right_hash: str) -> str:
        """Hash two child hashes.
        
        Args:
            left_hash: Left child hash.
            right_hash: Right child hash.
            
        Returns:
            Parent hash.
        """
        combined = (left_hash + right_hash).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def get_root_hash(self) -> Optional[str]:
        """Get root hash of tree.
        
        Returns:
            Root hash or None if empty.
        """
        return self._root.hash_value if self._root else None
    
    def get_proof(self, event_id: str) -> Optional[MerkleProof]:
        """Generate Merkle proof for an event.
        
        Args:
            event_id: Event to generate proof for.
            
        Returns:
            MerkleProof or None if not found.
        """
        # Find leaf index
        leaf_index = None
        for i, leaf in enumerate(self._leaves):
            if leaf.event_id == event_id:
                leaf_index = i
                break
        
        if leaf_index is None:
            return None
        
        # Collect sibling hashes on path to root
        sibling_hashes = []
        current_index = leaf_index
        current_level = self._leaves
        
        while len(current_level) > 1:
            # Find sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1
            
            if sibling_index < len(current_level):
                sibling_hashes.append(current_level[sibling_index].hash_value)
            
            # Move to parent level
            current_index = current_index // 2
            # Build next level for iteration
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent_hash = self._hash_pair(left.hash_value, right.hash_value)
                next_level.append(MerkleNode(hash_value=parent_hash, level=0))
            current_level = next_level
        
        leaf = self._leaves[leaf_index]
        return MerkleProof(
            event_id=event_id,
            leaf_hash=leaf.hash_value,
            sibling_hashes=sibling_hashes,
            leaf_index=leaf_index,
            tree_size=self._size,
        )
    
    def verify_proof(self, proof: MerkleProof, expected_root: str) -> bool:
        """Verify a Merkle proof.
        
        Args:
            proof: Proof to verify.
            expected_root: Expected root hash.
            
        Returns:
            True if proof is valid.
        """
        # Reconstruct root from leaf and proof
        hash_value = proof.leaf_hash
        current_index = proof.leaf_index
        
        for sibling_hash in proof.sibling_hashes:
            if current_index % 2 == 0:
                hash_value = self._hash_pair(hash_value, sibling_hash)
            else:
                hash_value = self._hash_pair(sibling_hash, hash_value)
            current_index //= 2
        
        return hash_value == expected_root
