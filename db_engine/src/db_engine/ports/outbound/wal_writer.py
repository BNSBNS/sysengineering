"""WAL Writer port for Write-Ahead Log persistence.

This outbound port defines the contract for WAL segment storage.
The WAL writer handles the low-level details of writing log records
to disk with appropriate durability guarantees.

References:
    - design.md Section 3 (WAL Manager)
    - ARIES paper (Mohan et al., 1992)
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Iterator, Protocol

from db_engine.domain.entities import LogRecord
from db_engine.domain.value_objects import LSN


class SyncMode(Enum):
    """WAL sync modes with different durability/performance tradeoffs.

    FSYNC: Full durability - sync file and metadata (safest)
    FDATASYNC: Data durability - sync file data only (faster on Linux)
    NONE: No sync - rely on OS buffering (fastest, but unsafe)
    """

    FSYNC = "fsync"
    FDATASYNC = "fdatasync"
    NONE = "none"


class WALWriter(Protocol):
    """Protocol for WAL segment persistence.

    The WAL writer manages log segments on disk. Each segment is
    a fixed-size file containing sequential log records.

    Key guarantees:
    - Records are written in LSN order
    - flush() ensures durability up to specified LSN
    - Segments are self-contained for recovery

    Thread Safety:
        Single-writer assumed. The WAL manager serializes all writes.
    """

    @property
    @abstractmethod
    def segment_size(self) -> int:
        """Return the maximum segment size in bytes.

        Default is typically 64MB.
        """
        ...

    @property
    @abstractmethod
    def sync_mode(self) -> SyncMode:
        """Return the current sync mode."""
        ...

    @abstractmethod
    def append(self, record: LogRecord) -> LSN:
        """Append a log record to the WAL.

        The record is written to the current segment. If the segment
        is full, a new segment is created automatically.

        Args:
            record: The log record to append.

        Returns:
            The LSN assigned to this record.

        Raises:
            IOError: If the write fails.
        """
        ...

    @abstractmethod
    def flush(self, lsn: LSN) -> None:
        """Flush all records up to and including the given LSN.

        This ensures durability - after flush returns, the records
        are guaranteed to be on stable storage.

        Args:
            lsn: The LSN to flush up to.

        Raises:
            ValueError: If LSN is invalid.
            IOError: If the flush fails.
        """
        ...

    @abstractmethod
    def get_flushed_lsn(self) -> LSN:
        """Return the highest LSN that has been flushed to disk.

        Returns:
            The flushed LSN. Records up to this LSN are durable.
        """
        ...

    @abstractmethod
    def get_current_lsn(self) -> LSN:
        """Return the next LSN that will be assigned.

        Returns:
            The current (next) LSN.
        """
        ...

    @abstractmethod
    def read_from(self, start_lsn: LSN) -> Iterator[LogRecord]:
        """Read log records starting from the given LSN.

        Used during recovery to replay the WAL.

        Args:
            start_lsn: The LSN to start reading from.

        Yields:
            LogRecord objects in LSN order.

        Raises:
            ValueError: If start_lsn is invalid or not found.
            IOError: If reading fails.
        """
        ...

    @abstractmethod
    def truncate_before(self, lsn: LSN) -> None:
        """Remove log segments that only contain records before LSN.

        Used to reclaim disk space after a checkpoint.

        Args:
            lsn: Records before this LSN may be removed.

        Raises:
            ValueError: If LSN is invalid.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the WAL writer and release resources.

        Ensures all buffered data is flushed before closing.
        """
        ...
