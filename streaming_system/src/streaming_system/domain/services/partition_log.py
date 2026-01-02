"""Partition log service for append-only message storage."""

from __future__ import annotations

from typing import Optional
import time
from streaming_system.domain.entities.record import (
    Record,
    LogEntry,
    LogSegment,
    HighWaterMark,
)


class PartitionLog:
    """Append-only log for a partition.
    
    Implements segmented log structure with sequential I/O for high throughput.
    """
    
    def __init__(self, partition_id: int):
        """Initialize partition log.
        
        Args:
            partition_id: Partition number.
        """
        self.partition_id = partition_id
        self._entries: list[LogEntry] = []
        self._segments: list[LogSegment] = []
        self._hwm = HighWaterMark()
        self._next_offset = 0
        self._term = 0
        
        # Create initial segment
        self._create_segment()
    
    def append(self, records: list[Record]) -> list[int]:
        """Append records to log.
        
        Args:
            records: Records to append.
            
        Returns:
            List of assigned offsets.
        """
        offsets = []
        
        for record in records:
            entry = LogEntry(
                offset=self._next_offset,
                term=self._term,
                record=record,
                is_committed=False,
            )
            
            self._entries.append(entry)
            offsets.append(self._next_offset)
            self._next_offset += 1
            
            # Update current segment
            if self._segments:
                self._segments[-1].next_offset = self._next_offset
                self._segments[-1].size_bytes += len(record.key) + len(record.value)
                
                # Roll to new segment if needed
                if self._segments[-1].is_full():
                    self._create_segment()
        
        return offsets
    
    def read(self, offset: int, max_records: int = 100) -> list[LogEntry]:
        """Read records from log.
        
        Args:
            offset: Starting offset.
            max_records: Maximum records to return.
            
        Returns:
            List of log entries.
        """
        if offset < 0 or offset >= self._next_offset:
            return []
        
        entries = []
        for i in range(offset, min(offset + max_records, self._next_offset)):
            entries.append(self._entries[i])
        
        return entries
    
    def commit(self, offset: int) -> None:
        """Commit entries up to offset (set HWM).
        
        Args:
            offset: Offset to commit to.
        """
        if offset >= self._next_offset:
            return
        
        # Mark entries as committed
        for i in range(offset + 1):
            if i < len(self._entries):
                self._entries[i].is_committed = True
        
        # Advance HWM
        self._hwm.advance(offset)
    
    def truncate(self, offset: int) -> None:
        """Truncate log at offset (for Raft recovery).
        
        Removes all entries from offset onwards.
        
        Args:
            offset: Truncate at this offset (exclusive).
        """
        if offset < 0 or offset >= self._next_offset:
            return
        
        # Remove entries after truncation point
        self._entries = self._entries[:offset]
        self._next_offset = offset
    
    def get_hwm(self) -> int:
        """Get high water mark (latest committed offset).
        
        Returns:
            HWM offset.
        """
        return self._hwm.get()
    
    def get_next_offset(self) -> int:
        """Get next available offset.
        
        Returns:
            Next offset to be assigned.
        """
        return self._next_offset
    
    def set_term(self, term: int) -> None:
        """Set current Raft term.
        
        Args:
            term: Raft term.
        """
        self._term = term
    
    def get_term(self) -> int:
        """Get current Raft term.
        
        Returns:
            Current term.
        """
        return self._term
    
    def _create_segment(self) -> None:
        """Create new log segment."""
        segment = LogSegment(
            segment_id=len(self._segments),
            base_offset=self._next_offset,
            next_offset=self._next_offset,
        )
        self._segments.append(segment)
    
    def get_stats(self) -> dict:
        """Get log statistics.
        
        Returns:
            Dictionary with stats.
        """
        return {
            "partition_id": self.partition_id,
            "total_records": self._next_offset,
            "committed_offset": self._hwm.get(),
            "segments": len(self._segments),
            "current_term": self._term,
        }
