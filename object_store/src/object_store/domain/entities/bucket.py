"""Bucket entity for object storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Bucket:
    """A bucket (namespace) for objects."""
    
    bucket_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    versioning_enabled: bool = True
    deduplication_enabled: bool = True
    compression_algorithm: str = "zstd"  # zstd, gzip, none
    replication_factor: int = 3
    erasure_parity_shards: int = 2
    metadata: dict = field(default_factory=dict)
    object_count: int = 0
    total_size: int = 0
    
    def should_compress(self, content_type: str) -> bool:
        """Check if content should be compressed.
        
        Args:
            content_type: MIME type.
            
        Returns:
            True if should compress.
        """
        if self.compression_algorithm == "none":
            return False
        
        # Compress text and json
        compressible = ["text/", "application/json", "application/xml"]
        return any(content_type.startswith(c) for c in compressible)
    
    def get_chunk_size(self) -> int:
        """Get recommended chunk size.
        
        Returns:
            Chunk size in bytes.
        """
        return 64 * 1024 * 1024  # 64MB
