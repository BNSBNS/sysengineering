"""Chunk entity for object storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Chunk:
    """A chunk of object data."""
    
    chunk_id: str
    object_id: str
    sequence: int
    data: bytes
    size: int
    checksum: str  # SHA256
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of data.
        
        Returns:
            Hex string of checksum.
        """
        import hashlib
        return hashlib.sha256(self.data).hexdigest()
    
    def verify_checksum(self) -> bool:
        """Verify chunk data integrity.
        
        Returns:
            True if checksum matches.
        """
        return self.checksum == self.calculate_checksum()


@dataclass
class ChunkRef:
    """Reference to a chunk."""
    
    chunk_id: str
    checksum: str
    sequence: int
    size: int
    shard_indices: list[int] = field(default_factory=list)
