"""Object entity for object storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_store.domain.entities.chunk import ChunkRef


@dataclass
class Object:
    """An object in the store."""

    object_id: str
    bucket: str
    key: str
    version: int = 1
    size: int = 0
    checksum: str = ""  # Merkle root
    content_type: str = "application/octet-stream"
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    chunk_refs: "list[ChunkRef]" = field(default_factory=list)
    replication_factor: int = 3
    erasure_parity_shards: int = 2
    
    def get_full_path(self) -> str:
        """Get full object path.
        
        Returns:
            Bucket/key path.
        """
        return f"{self.bucket}/{self.key}"
    
    def is_latest_version(self) -> bool:
        """Check if this is latest version.
        
        Returns:
            True if latest.
        """
        return self.version > 0


@dataclass
class ObjectVersion:
    """Object version tracking."""
    
    object_id: str
    version: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    size: int = 0
    checksum: str = ""
    metadata: dict = field(default_factory=dict)
    is_deleted: bool = False
