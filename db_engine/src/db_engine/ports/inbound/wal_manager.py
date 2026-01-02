"""WAL Manager port for transaction logging and recovery.

This inbound port defines the high-level WAL interface used by
the transaction manager and recovery service.

Key responsibilities:
- Append log records for transaction operations
- Ensure durability via flush
- Perform ARIES-style crash recovery
- Create checkpoints for faster recovery

References:
    - design.md Section 3 (WAL Manager)
    - design.md Section 6 (ARIES Recovery)
    - Mohan et al., "ARIES" (1992)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from db_engine.domain.entities import LogRecord
from db_engine.domain.value_objects import LSN, PageId, TransactionId


@dataclass
class RecoveryStats:
    """Statistics from crash recovery."""

    checkpoint_lsn: LSN  # LSN of last checkpoint
    redo_start_lsn: LSN  # Where redo phase started
    records_analyzed: int = 0  # Records read in analysis
    records_redone: int = 0  # Records replayed in redo
    records_undone: int = 0  # Records undone
    transactions_committed: int = 0  # Winners
    transactions_aborted: int = 0  # Losers
    duration_ms: float = 0.0  # Total recovery time


@dataclass
class CheckpointData:
    """Data captured during a checkpoint.

    Active Transaction Table (ATT): Maps active transaction IDs
    to their last LSN (for undo chain).

    Dirty Page Table (DPT): Maps dirty page IDs to their
    recovery LSN (first LSN that made them dirty).
    """

    active_txns: dict[TransactionId, LSN] = field(default_factory=dict)
    dirty_pages: dict[PageId, LSN] = field(default_factory=dict)


class WALManager(Protocol):
    """Protocol for WAL operations.

    The WAL manager is the high-level interface for transaction
    logging. It coordinates with the buffer pool and disk manager
    to implement the WAL protocol.

    WAL Protocol:
    1. Log record written to WAL before data modification
    2. WAL flushed to disk before COMMIT returns
    3. Data pages flushed lazily (background or checkpoint)

    Thread Safety:
        The WAL manager serializes log appends internally.
        Multiple transactions can append concurrently.
    """

    @abstractmethod
    def append_log(self, record: LogRecord) -> LSN:
        """Append a log record to the WAL.

        The record is buffered until flush is called.

        Args:
            record: The log record to append.

        Returns:
            The LSN assigned to this record.
        """
        ...

    @abstractmethod
    def flush(self, lsn: LSN) -> None:
        """Ensure all records up to LSN are durable.

        This is called during COMMIT to guarantee durability.
        Blocks until the flush completes.

        Args:
            lsn: The LSN to flush up to.
        """
        ...

    @abstractmethod
    def flush_all(self) -> None:
        """Flush all buffered log records.

        Convenience method equivalent to flush(current_lsn).
        """
        ...

    @abstractmethod
    def recover(self) -> RecoveryStats:
        """Perform ARIES crash recovery.

        This should be called at startup before any transactions
        are processed. It restores the database to a consistent
        state by:

        1. Analysis: Scan WAL to find active transactions and dirty pages
        2. Redo: Replay all logged operations from redo start point
        3. Undo: Rollback uncommitted transactions

        Returns:
            Statistics about the recovery process.

        Raises:
            RecoveryError: If recovery fails.
        """
        ...

    @abstractmethod
    def checkpoint(self) -> LSN:
        """Create a checkpoint to speed up future recovery.

        A checkpoint captures:
        - Active Transaction Table (ATT)
        - Dirty Page Table (DPT)

        During checkpoint:
        1. Acquire checkpoint lock
        2. Write CHECKPOINT_BEGIN record
        3. Flush all dirty pages (fuzzy checkpoint)
        4. Write CHECKPOINT_END record with ATT/DPT
        5. Release checkpoint lock

        Returns:
            The LSN of the checkpoint record.
        """
        ...

    @abstractmethod
    def get_flushed_lsn(self) -> LSN:
        """Return the highest LSN that has been flushed."""
        ...

    @abstractmethod
    def get_current_lsn(self) -> LSN:
        """Return the next LSN that will be assigned."""
        ...

    @abstractmethod
    def get_checkpoint_data(self) -> CheckpointData:
        """Get current ATT and DPT for checkpoint."""
        ...


class RecoveryError(Exception):
    """Raised when crash recovery fails.

    Recovery failures are critical - the database cannot start
    until recovery completes successfully.
    """

    pass
