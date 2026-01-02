"""File-based WAL Writer implementation.

This adapter implements the WALWriter protocol using segment files.
Each segment is a fixed-size file containing sequential log records.

Segment File Format:
    - Segment Header (32 bytes): magic, version, segment_id, first_lsn, record_count
    - Log Records: [length(4) + record_bytes + CRC32(4)] ...

Thread Safety:
    Single-writer assumed. All writes are serialized internally.

References:
    - design.md Section 3 (WAL Manager)
    - ARIES paper (Mohan et al., 1992)
"""

from __future__ import annotations

import os
import struct
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from db_engine.domain.entities import LogRecord
from db_engine.domain.value_objects import LSN, INVALID_LSN
from db_engine.infrastructure.config import get_config
from db_engine.ports.outbound.wal_writer import SyncMode


# Segment header format
SEGMENT_MAGIC = b"WALSGMT\x00"
SEGMENT_VERSION = 1
SEGMENT_HEADER_FORMAT = ">8sIQQI"  # magic, version, segment_id, first_lsn, record_count
SEGMENT_HEADER_SIZE = struct.calcsize(SEGMENT_HEADER_FORMAT)

# Record wrapper format: length(4) + data + crc32(4)
RECORD_LENGTH_FORMAT = ">I"
RECORD_CRC_FORMAT = ">I"
RECORD_OVERHEAD = 8  # 4 bytes length + 4 bytes CRC


@dataclass
class SegmentInfo:
    """Metadata for a WAL segment."""

    segment_id: int
    first_lsn: LSN
    record_count: int
    file_path: Path
    file_size: int


class FileWALWriter:
    """File-based implementation of the WALWriter protocol.

    Manages multiple segment files in a WAL directory. Each segment
    has a fixed maximum size. When a segment is full, a new one is
    created.

    Attributes:
        wal_dir: Directory containing WAL segment files.
        segment_size: Maximum size of each segment file.
        sync_mode: How to sync writes to disk.
    """

    def __init__(
        self,
        wal_dir: str | Path,
        segment_size: int | None = None,
        sync_mode: SyncMode | None = None,
    ) -> None:
        """Initialize the WAL writer.

        Args:
            wal_dir: Directory for WAL segment files.
            segment_size: Maximum segment size in bytes (default from config).
            sync_mode: Sync mode for durability (default from config).
        """
        config = get_config()
        self._wal_dir = Path(wal_dir)
        self._segment_size = segment_size or config.wal.segment_size
        self._sync_mode = sync_mode or SyncMode(config.wal.sync_mode)

        self._lock = threading.Lock()
        self._closed = False

        # Current state
        self._current_lsn = LSN(1)  # LSN 0 is invalid
        self._flushed_lsn = LSN(0)
        self._current_segment_id = 0
        self._current_file: BinaryIO | None = None
        self._current_segment_first_lsn = LSN(1)
        self._current_record_count = 0
        self._buffer: list[tuple[LSN, bytes]] = []  # (lsn, serialized_record)

        # Segment tracking
        self._segments: list[SegmentInfo] = []

        # Initialize
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        self._recover_state()

    def _segment_path(self, segment_id: int) -> Path:
        """Get the file path for a segment."""
        return self._wal_dir / f"wal_{segment_id:08d}.log"

    def _recover_state(self) -> None:
        """Recover state from existing WAL segments."""
        # Find all existing segments
        segment_files = sorted(self._wal_dir.glob("wal_*.log"))

        if not segment_files:
            # No existing segments - start fresh
            self._open_new_segment()
            return

        # Load segment metadata
        for segment_path in segment_files:
            try:
                info = self._read_segment_header(segment_path)
                self._segments.append(info)
            except (ValueError, IOError):
                # Corrupted segment - stop here
                break

        if self._segments:
            # Continue from last segment
            last_segment = self._segments[-1]
            self._current_segment_id = last_segment.segment_id
            self._current_segment_first_lsn = last_segment.first_lsn

            # Scan to find the last LSN
            max_lsn = last_segment.first_lsn
            for record in self._scan_segment(last_segment.file_path):
                max_lsn = record.lsn

            self._current_lsn = LSN(max_lsn + 1)
            self._flushed_lsn = max_lsn
            self._current_record_count = last_segment.record_count

            # Open the current segment for appending
            self._current_file = open(last_segment.file_path, "r+b")
            self._current_file.seek(0, os.SEEK_END)

            # Check if we need a new segment
            if self._current_file.tell() >= self._segment_size:
                self._current_file.close()
                self._open_new_segment()
        else:
            self._open_new_segment()

    def _read_segment_header(self, path: Path) -> SegmentInfo:
        """Read segment header from a file."""
        with open(path, "rb") as f:
            header_data = f.read(SEGMENT_HEADER_SIZE)

        if len(header_data) < SEGMENT_HEADER_SIZE:
            raise ValueError("Segment header too short")

        magic, version, segment_id, first_lsn, record_count = struct.unpack(
            SEGMENT_HEADER_FORMAT, header_data
        )

        if magic != SEGMENT_MAGIC:
            raise ValueError(f"Invalid segment magic: {magic!r}")

        if version != SEGMENT_VERSION:
            raise ValueError(f"Unsupported segment version: {version}")

        return SegmentInfo(
            segment_id=segment_id,
            first_lsn=LSN(first_lsn),
            record_count=record_count,
            file_path=path,
            file_size=path.stat().st_size,
        )

    def _open_new_segment(self) -> None:
        """Create and open a new segment file."""
        if self._current_file is not None:
            self._update_segment_header()
            self._current_file.close()

        self._current_segment_id += 1
        segment_path = self._segment_path(self._current_segment_id)
        self._current_segment_first_lsn = self._current_lsn
        self._current_record_count = 0

        self._current_file = open(segment_path, "w+b")

        # Write segment header
        header = struct.pack(
            SEGMENT_HEADER_FORMAT,
            SEGMENT_MAGIC,
            SEGMENT_VERSION,
            self._current_segment_id,
            self._current_segment_first_lsn,
            0,  # record_count - updated later
        )
        self._current_file.write(header)

        # Track segment
        self._segments.append(
            SegmentInfo(
                segment_id=self._current_segment_id,
                first_lsn=self._current_segment_first_lsn,
                record_count=0,
                file_path=segment_path,
                file_size=SEGMENT_HEADER_SIZE,
            )
        )

    def _update_segment_header(self) -> None:
        """Update the current segment's header with record count."""
        if self._current_file is None:
            return

        pos = self._current_file.tell()
        self._current_file.seek(0)

        header = struct.pack(
            SEGMENT_HEADER_FORMAT,
            SEGMENT_MAGIC,
            SEGMENT_VERSION,
            self._current_segment_id,
            self._current_segment_first_lsn,
            self._current_record_count,
        )
        self._current_file.write(header)
        self._current_file.seek(pos)

        # Update segment info
        if self._segments:
            self._segments[-1].record_count = self._current_record_count

    def _scan_segment(self, path: Path) -> Iterator[LogRecord]:
        """Scan a segment file and yield log records."""
        with open(path, "rb") as f:
            # Skip header
            f.seek(SEGMENT_HEADER_SIZE)

            while True:
                # Read record length
                length_data = f.read(4)
                if len(length_data) < 4:
                    break

                (length,) = struct.unpack(RECORD_LENGTH_FORMAT, length_data)
                if length == 0:
                    break

                # Read record data
                record_data = f.read(length)
                if len(record_data) < length:
                    break

                # Read and verify CRC
                crc_data = f.read(4)
                if len(crc_data) < 4:
                    break

                (stored_crc,) = struct.unpack(RECORD_CRC_FORMAT, crc_data)
                computed_crc = zlib.crc32(record_data) & 0xFFFFFFFF

                if stored_crc != computed_crc:
                    raise ValueError(
                        f"CRC mismatch: stored={stored_crc}, computed={computed_crc}"
                    )

                # Deserialize record
                yield LogRecord.from_bytes(record_data)

    @property
    def segment_size(self) -> int:
        """Return the maximum segment size in bytes."""
        return self._segment_size

    @property
    def sync_mode(self) -> SyncMode:
        """Return the current sync mode."""
        return self._sync_mode

    def append(self, record: LogRecord) -> LSN:
        """Append a log record to the WAL.

        Args:
            record: The log record to append.

        Returns:
            The LSN assigned to this record.

        Raises:
            IOError: If the write fails.
        """
        if self._closed or self._current_file is None:
            raise IOError("WAL writer is closed")

        with self._lock:
            # Assign LSN
            lsn = self._current_lsn
            self._current_lsn = LSN(self._current_lsn + 1)

            # Update record's LSN before serialization
            record.lsn = lsn

            # Serialize record
            record_bytes = record.to_bytes()
            crc = zlib.crc32(record_bytes) & 0xFFFFFFFF

            # Format: length + data + crc
            wrapped = (
                struct.pack(RECORD_LENGTH_FORMAT, len(record_bytes))
                + record_bytes
                + struct.pack(RECORD_CRC_FORMAT, crc)
            )

            # Buffer the record
            self._buffer.append((lsn, wrapped))

            # Check if we need a new segment
            current_size = self._current_file.tell() + sum(
                len(w) for _, w in self._buffer
            )
            if current_size >= self._segment_size:
                # Flush buffer to current segment, then open new one
                self._flush_buffer()
                self._open_new_segment()

            return lsn

    def _flush_buffer(self) -> None:
        """Flush buffered records to the current segment."""
        if not self._buffer or self._current_file is None:
            return

        for lsn, wrapped in self._buffer:
            self._current_file.write(wrapped)
            self._current_record_count += 1

        self._buffer.clear()

        # Sync based on mode
        if self._sync_mode == SyncMode.FSYNC:
            self._current_file.flush()
            os.fsync(self._current_file.fileno())
        elif self._sync_mode == SyncMode.FDATASYNC:
            self._current_file.flush()
            # fdatasync not available on Windows, fall back to fsync
            os.fsync(self._current_file.fileno())
        # SyncMode.NONE - no sync

    def flush(self, lsn: LSN) -> None:
        """Flush all records up to and including the given LSN.

        Args:
            lsn: The LSN to flush up to.

        Raises:
            ValueError: If LSN is invalid.
            IOError: If the flush fails.
        """
        if self._closed:
            raise IOError("WAL writer is closed")

        if lsn > self._current_lsn:
            raise ValueError(f"LSN {lsn} has not been written yet")

        with self._lock:
            if lsn <= self._flushed_lsn:
                return  # Already flushed

            self._flush_buffer()
            self._update_segment_header()

            if self._current_file is not None:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())

            self._flushed_lsn = lsn

    def get_flushed_lsn(self) -> LSN:
        """Return the highest LSN that has been flushed to disk."""
        return self._flushed_lsn

    def get_current_lsn(self) -> LSN:
        """Return the next LSN that will be assigned."""
        return self._current_lsn

    def read_from(self, start_lsn: LSN) -> Iterator[LogRecord]:
        """Read log records starting from the given LSN.

        Args:
            start_lsn: The LSN to start reading from.

        Yields:
            LogRecord objects in LSN order.

        Raises:
            ValueError: If start_lsn is invalid or not found.
            IOError: If reading fails.
        """
        if self._closed:
            raise IOError("WAL writer is closed")

        # Find the segment containing start_lsn
        start_segment_idx = None
        for i, segment in enumerate(self._segments):
            if segment.first_lsn <= start_lsn:
                start_segment_idx = i

        if start_segment_idx is None:
            raise ValueError(f"LSN {start_lsn} not found in WAL")

        # Flush any buffered records first
        with self._lock:
            self._flush_buffer()
            if self._current_file is not None:
                self._current_file.flush()

        # Scan segments from start_segment_idx
        for segment in self._segments[start_segment_idx:]:
            for record in self._scan_segment(segment.file_path):
                if record.lsn >= start_lsn:
                    yield record

    def truncate_before(self, lsn: LSN) -> None:
        """Remove log segments that only contain records before LSN.

        Args:
            lsn: Records before this LSN may be removed.

        Raises:
            ValueError: If LSN is invalid.
        """
        if self._closed:
            raise IOError("WAL writer is closed")

        with self._lock:
            # Find segments that can be removed
            segments_to_remove = []
            for segment in self._segments:
                # Keep the segment if any record might be >= lsn
                # This is conservative - we keep if first_lsn is close to lsn
                if segment.first_lsn < lsn:
                    # Check if this is not the current segment
                    if segment.segment_id != self._current_segment_id:
                        segments_to_remove.append(segment)

            # Remove old segments
            for segment in segments_to_remove:
                try:
                    segment.file_path.unlink()
                    self._segments.remove(segment)
                except OSError:
                    pass  # Ignore errors during cleanup

    def close(self) -> None:
        """Close the WAL writer and release resources."""
        if self._closed:
            return

        with self._lock:
            self._closed = True
            self._flush_buffer()
            self._update_segment_header()

            if self._current_file is not None:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
                self._current_file.close()
                self._current_file = None

    def __enter__(self) -> FileWALWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure file is closed."""
        self.close()
