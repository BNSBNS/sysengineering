"""Record and log entry entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib


class RecordCompressionType(Enum):
    """Compression algorithm for records."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"


@dataclass
class Record:
    """Single message record in a partition.
    
    Records are immutable and include key, value, and metadata.
    """
    key: bytes
    value: bytes
    timestamp: float = field(default_factory=time.time)
    compression_type: RecordCompressionType = RecordCompressionType.NONE
    headers: dict = field(default_factory=dict)
    
    def compute_crc(self) -> str:
        """Compute CRC32 checksum for integrity verification.
        
        Returns:
            CRC32 hex string.
        """
        data = self.key + self.value + str(self.timestamp).encode()
        return hashlib.sha256(data).hexdigest()[:8]


@dataclass
class LogEntry:
    """Entry in a partition log (Raft log entry)."""
    offset: int  # Position in log
    term: int  # Raft term when entry was received
    record: Record
    is_committed: bool = False
    
    def __hash__(self) -> int:
        """Hash for log entry comparison."""
        return hash((self.offset, self.term))


@dataclass
class LogSegment:
    """Segment of log file with metadata.
    
    Logs are split into segments for efficient retention:
    - Old segments can be deleted without rewriting
    - Each segment has an index for fast lookup
    - Typical segment size: 1GB
    """
    segment_id: int  # Base offset of segment
    base_offset: int  # First offset in segment
    next_offset: int  # Next available offset
    is_sealed: bool = False  # Sealed = no more appends
    size_bytes: int = 0  # Current size
    max_segment_bytes: int = 1_000_000_000  # 1GB default
    
    def is_full(self) -> bool:
        """Check if segment is at capacity.
        
        Returns:
            True if segment should be rolled.
        """
        return self.size_bytes >= self.max_segment_bytes
    
    def records_count(self) -> int:
        """Get number of records in segment.
        
        Returns:
            Record count.
        """
        return self.next_offset - self.base_offset


@dataclass
class HighWaterMark:
    """High water mark for durability guarantees.
    
    Records up to HWM are committed and durable.
    Records above HWM are not yet replicated.
    """
    value: int = 0  # Current HWM offset
    
    def advance(self, offset: int) -> None:
        """Advance HWM to new offset.
        
        Args:
            offset: New HWM offset.
        """
        if offset > self.value:
            self.value = offset
    
    def get(self) -> int:
        """Get current HWM.
        
        Returns:
            Current HWM offset.
        """
        return self.value
