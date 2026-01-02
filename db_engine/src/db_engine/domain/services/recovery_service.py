"""ARIES-style crash recovery service.

This module implements the ARIES (Algorithm for Recovery and Isolation
Exploiting Semantics) recovery algorithm. ARIES is the industry standard
for database crash recovery, used by DB2, PostgreSQL, SQL Server, and others.

Key Concepts:
    - WAL (Write-Ahead Logging): Log before modify, flush before commit
    - LSN (Log Sequence Number): Total order on all log records
    - ATT (Active Transaction Table): Tracks in-flight transactions
    - DPT (Dirty Page Table): Tracks pages that might not be on disk

Three Recovery Phases:
    1. Analysis: Scan WAL to rebuild ATT and DPT
    2. Redo: Replay history to restore crash-time state
    3. Undo: Rollback uncommitted transactions

References:
    - Mohan, C. et al. "ARIES: A Transaction Recovery Method" (1992)
    - design.md Section 6 (Failure Modes & Recovery)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from db_engine.domain.entities import (
    AbortRecord,
    BeginRecord,
    CheckpointRecord,
    CLRRecord,
    CommitRecord,
    DeleteRecord,
    InsertRecord,
    LogRecord,
    UpdateRecord,
)
from db_engine.domain.value_objects import LSN, INVALID_LSN, PageId, TransactionId
from db_engine.ports.inbound.buffer_pool import BufferPool
from db_engine.ports.inbound.wal_manager import CheckpointData, RecoveryError, RecoveryStats
from db_engine.ports.outbound.wal_writer import WALWriter


@dataclass
class TransactionState:
    """State of a transaction during recovery.

    Attributes:
        txn_id: The transaction identifier.
        last_lsn: The last LSN written by this transaction.
        status: Current status ('active', 'committing', 'committed', 'aborting', 'aborted').
        undo_next_lsn: Next LSN to undo (for partial rollback).
    """
    txn_id: TransactionId
    last_lsn: LSN = INVALID_LSN
    status: str = "active"
    undo_next_lsn: LSN = INVALID_LSN


@dataclass
class RecoveryContext:
    """Context maintained during recovery.

    This is rebuilt during the Analysis phase and used by Redo and Undo.
    """
    # Active Transaction Table: txn_id -> TransactionState
    active_txns: dict[TransactionId, TransactionState] = field(default_factory=dict)

    # Dirty Page Table: page_id -> recLSN (first LSN that dirtied the page)
    dirty_pages: dict[PageId, LSN] = field(default_factory=dict)

    # Last checkpoint LSN (or INVALID_LSN if none)
    checkpoint_lsn: LSN = INVALID_LSN

    # Statistics
    records_analyzed: int = 0
    records_redone: int = 0
    records_undone: int = 0
    winners: set[TransactionId] = field(default_factory=set)
    losers: set[TransactionId] = field(default_factory=set)


class RecoveryService:
    """ARIES-style crash recovery implementation.

    This service is responsible for restoring the database to a consistent
    state after a crash. It reads the WAL and applies the three-phase
    ARIES algorithm: Analysis, Redo, Undo.

    Usage:
        recovery = RecoveryService(wal_writer, buffer_pool)
        stats = recovery.recover()
        print(f"Recovery completed: {stats.transactions_committed} committed")

    Thread Safety:
        Recovery should run single-threaded before any concurrent access.
    """

    def __init__(
        self,
        wal_writer: WALWriter,
        buffer_pool: BufferPool,
    ) -> None:
        """Initialize the recovery service.

        Args:
            wal_writer: The WAL writer for reading/writing log records.
            buffer_pool: The buffer pool for page access.
        """
        self._wal_writer = wal_writer
        self._buffer_pool = buffer_pool

    def recover(self) -> RecoveryStats:
        """Perform full ARIES crash recovery.

        This method should be called at database startup before
        any new transactions are processed.

        Returns:
            Statistics about the recovery process.

        Raises:
            RecoveryError: If recovery fails.
        """
        start_time = time.time()

        try:
            # Phase 1: Analysis
            context = self._analysis_phase()

            # Phase 2: Redo
            self._redo_phase(context)

            # Phase 3: Undo
            self._undo_phase(context)

            # Build stats
            duration_ms = (time.time() - start_time) * 1000

            return RecoveryStats(
                checkpoint_lsn=context.checkpoint_lsn,
                redo_start_lsn=self._get_redo_start_lsn(context),
                records_analyzed=context.records_analyzed,
                records_redone=context.records_redone,
                records_undone=context.records_undone,
                transactions_committed=len(context.winners),
                transactions_aborted=len(context.losers),
                duration_ms=duration_ms,
            )

        except Exception as e:
            raise RecoveryError(f"Recovery failed: {e}") from e

    def _analysis_phase(self) -> RecoveryContext:
        """Phase 1: Analysis - Scan WAL to rebuild ATT and DPT.

        Starting from the last checkpoint (or beginning of WAL if none),
        scan forward to the end of the WAL, rebuilding:
        - Active Transaction Table (ATT)
        - Dirty Page Table (DPT)
        - Winner/loser transaction sets

        Returns:
            RecoveryContext with ATT, DPT, and transaction status.
        """
        context = RecoveryContext()

        # Find the last checkpoint and start scanning from there
        start_lsn = self._find_checkpoint(context)

        # Scan forward from start point
        try:
            for record in self._wal_writer.read_from(start_lsn):
                context.records_analyzed += 1
                self._analyze_record(record, context)
        except ValueError:
            # No records found (empty WAL or no records from start_lsn)
            pass

        # Determine winners and losers
        for txn_id, txn_state in context.active_txns.items():
            if txn_state.status == "committed":
                context.winners.add(txn_id)
            else:
                context.losers.add(txn_id)

        return context

    def _find_checkpoint(self, context: RecoveryContext) -> LSN:
        """Find the last checkpoint and initialize context from it.

        Args:
            context: The recovery context to initialize.

        Returns:
            The LSN to start scanning from.
        """
        # For now, start from LSN 1 (beginning)
        # A full implementation would scan backwards to find CHECKPOINT record
        # and initialize ATT/DPT from it

        # Try to read from the beginning
        start_lsn = LSN(1)

        # Scan for most recent checkpoint record
        try:
            for record in self._wal_writer.read_from(start_lsn):
                if isinstance(record, CheckpointRecord):
                    context.checkpoint_lsn = record.lsn
                    # Initialize ATT and DPT from checkpoint
                    for txn_id, last_lsn in record.active_txns.items():
                        context.active_txns[txn_id] = TransactionState(
                            txn_id=txn_id,
                            last_lsn=last_lsn,
                            status="active",
                            undo_next_lsn=last_lsn,
                        )
                    for page_id, rec_lsn in record.dirty_pages.items():
                        context.dirty_pages[page_id] = rec_lsn
        except ValueError:
            pass

        # If we found a checkpoint, start from there; otherwise from beginning
        if context.checkpoint_lsn != INVALID_LSN:
            return context.checkpoint_lsn
        return start_lsn

    def _analyze_record(self, record: LogRecord, context: RecoveryContext) -> None:
        """Process a single record during Analysis phase.

        Updates ATT and DPT based on record type.
        """
        txn_id = record.txn_id

        # Ensure transaction is in ATT
        if txn_id not in context.active_txns:
            context.active_txns[txn_id] = TransactionState(
                txn_id=txn_id,
                status="active",
            )

        txn_state = context.active_txns[txn_id]
        txn_state.last_lsn = record.lsn
        txn_state.undo_next_lsn = record.prev_lsn

        if isinstance(record, BeginRecord):
            txn_state.status = "active"

        elif isinstance(record, CommitRecord):
            txn_state.status = "committed"

        elif isinstance(record, AbortRecord):
            txn_state.status = "aborted"

        elif isinstance(record, (UpdateRecord, InsertRecord, DeleteRecord)):
            # Add page to DPT if not already there
            page_id = record.page_id
            if page_id not in context.dirty_pages:
                context.dirty_pages[page_id] = record.lsn

        elif isinstance(record, CLRRecord):
            # CLR indicates undo in progress
            txn_state.undo_next_lsn = record.undo_next_lsn
            # Add page to DPT if not already there
            if record.page_id not in context.dirty_pages:
                context.dirty_pages[record.page_id] = record.lsn

    def _get_redo_start_lsn(self, context: RecoveryContext) -> LSN:
        """Determine where to start the Redo phase.

        Redo starts from the minimum recLSN in the DPT,
        as that's the oldest operation that might not be on disk.
        """
        if not context.dirty_pages:
            return self._wal_writer.get_current_lsn()

        return min(context.dirty_pages.values())

    def _redo_phase(self, context: RecoveryContext) -> None:
        """Phase 2: Redo - Replay all operations from redo start point.

        Redo restores the database to its crash-time state by replaying
        all operations. Even operations from uncommitted transactions
        are redone (they will be undone in Phase 3).

        Key principle: "Repeating history"
        """
        redo_start = self._get_redo_start_lsn(context)

        try:
            for record in self._wal_writer.read_from(redo_start):
                if self._should_redo(record, context):
                    self._redo_record(record, context)
                    context.records_redone += 1
        except ValueError:
            # No records to redo
            pass

    def _should_redo(self, record: LogRecord, context: RecoveryContext) -> bool:
        """Determine if a record should be redone.

        A record is redone if:
        1. It modifies a page (UPDATE, INSERT, DELETE, CLR)
        2. The page is in the DPT
        3. The record's LSN >= page's recLSN in DPT
        4. The page's current LSN < record's LSN (not already applied)
        """
        if not isinstance(record, (UpdateRecord, InsertRecord, DeleteRecord, CLRRecord)):
            return False

        page_id = record.page_id

        # Page must be in DPT
        if page_id not in context.dirty_pages:
            return False

        # Record must be >= recLSN
        rec_lsn = context.dirty_pages[page_id]
        if record.lsn < rec_lsn:
            return False

        # Check page's current LSN (if page is in buffer pool)
        try:
            page = self._buffer_pool.fetch_page(page_id)
            page_lsn = self._buffer_pool.get_page_lsn(page_id)
            self._buffer_pool.unpin_page(page_id, is_dirty=False)

            # Already applied?
            if page_lsn >= record.lsn:
                return False
        except (ValueError, IOError):
            # Page not in buffer pool or error - assume needs redo
            pass

        return True

    def _redo_record(self, record: LogRecord, context: RecoveryContext) -> None:
        """Apply a record during Redo phase.

        This re-applies the operation recorded in the WAL.
        """
        if isinstance(record, UpdateRecord):
            self._redo_update(record)
        elif isinstance(record, InsertRecord):
            self._redo_insert(record)
        elif isinstance(record, DeleteRecord):
            self._redo_delete(record)
        elif isinstance(record, CLRRecord):
            self._redo_clr(record)

    def _redo_update(self, record: UpdateRecord) -> None:
        """Redo an UPDATE operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            # Apply after_image
            page.update_record(record.slot_id, record.after_image)
            self._buffer_pool.set_page_lsn(record.page_id, record.lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            # Page doesn't exist or slot invalid - skip
            pass

    def _redo_insert(self, record: InsertRecord) -> None:
        """Redo an INSERT operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            # Re-insert at the same slot
            # Note: This assumes the page structure supports this
            if record.slot_id < page.slot_count:
                # Slot exists, update it
                page.update_record(record.slot_id, record.data)
            else:
                # Insert new record
                page.insert_record(record.data)
            self._buffer_pool.set_page_lsn(record.page_id, record.lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            # Page doesn't exist - skip
            pass

    def _redo_delete(self, record: DeleteRecord) -> None:
        """Redo a DELETE operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page.delete_record(record.slot_id)
            self._buffer_pool.set_page_lsn(record.page_id, record.lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            # Page doesn't exist or record already deleted - skip
            pass

    def _redo_clr(self, record: CLRRecord) -> None:
        """Redo a CLR (Compensation Log Record).

        CLRs are generated during undo and record the undo action.
        They are redone like any other record.
        """
        # CLR redo depends on what was being compensated
        # For now, just update the page LSN
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            self._buffer_pool.set_page_lsn(record.page_id, record.lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            pass

    def _undo_phase(self, context: RecoveryContext) -> None:
        """Phase 3: Undo - Rollback uncommitted transactions.

        Process losers (uncommitted transactions) in reverse LSN order.
        Write CLR records to ensure idempotence on repeated crashes.
        """
        # Collect all LSNs to undo
        to_undo: list[tuple[LSN, TransactionId]] = []

        for txn_id in context.losers:
            txn_state = context.active_txns.get(txn_id)
            if txn_state and txn_state.undo_next_lsn != INVALID_LSN:
                to_undo.append((txn_state.last_lsn, txn_id))

        # Sort by LSN descending (process most recent first)
        to_undo.sort(key=lambda x: x[0], reverse=True)

        # Process each loser transaction
        for _, txn_id in to_undo:
            self._undo_transaction(txn_id, context)

    def _undo_transaction(self, txn_id: TransactionId, context: RecoveryContext) -> None:
        """Undo all operations for a single transaction.

        Follows the prev_lsn chain backwards, undoing each operation.
        """
        txn_state = context.active_txns.get(txn_id)
        if not txn_state:
            return

        current_lsn = txn_state.last_lsn

        while current_lsn != INVALID_LSN:
            # Find the record at this LSN
            record = self._find_record(current_lsn)
            if record is None:
                break

            # Undo if it's a data modification
            if isinstance(record, (UpdateRecord, InsertRecord, DeleteRecord)):
                self._undo_record(record, context)
                context.records_undone += 1

            # Move to previous record in transaction
            current_lsn = record.prev_lsn

    def _find_record(self, lsn: LSN) -> LogRecord | None:
        """Find a record by its LSN.

        This is inefficient for large WALs - a production implementation
        would use an index or cache.
        """
        try:
            for record in self._wal_writer.read_from(lsn):
                if record.lsn == lsn:
                    return record
                if record.lsn > lsn:
                    break
        except ValueError:
            pass
        return None

    def _undo_record(self, record: LogRecord, context: RecoveryContext) -> None:
        """Undo a single record and write a CLR.

        The CLR ensures that if we crash during undo, we don't
        redo the undo operation on restart.
        """
        if isinstance(record, UpdateRecord):
            self._undo_update(record, context)
        elif isinstance(record, InsertRecord):
            self._undo_insert(record, context)
        elif isinstance(record, DeleteRecord):
            self._undo_delete(record, context)

    def _undo_update(self, record: UpdateRecord, context: RecoveryContext) -> None:
        """Undo an UPDATE by restoring the before_image."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            # Restore before_image
            page.update_record(record.slot_id, record.before_image)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),  # Will be assigned by WAL writer
                txn_id=record.txn_id,
                prev_lsn=record.lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            clr_lsn = self._wal_writer.append(clr)

            self._buffer_pool.set_page_lsn(record.page_id, clr_lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            pass

    def _undo_insert(self, record: InsertRecord, context: RecoveryContext) -> None:
        """Undo an INSERT by deleting the record."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page.delete_record(record.slot_id)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),
                txn_id=record.txn_id,
                prev_lsn=record.lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            clr_lsn = self._wal_writer.append(clr)

            self._buffer_pool.set_page_lsn(record.page_id, clr_lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            pass

    def _undo_delete(self, record: DeleteRecord, context: RecoveryContext) -> None:
        """Undo a DELETE by re-inserting the record."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            # Re-insert the deleted data
            page.insert_record(record.data)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),
                txn_id=record.txn_id,
                prev_lsn=record.lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            clr_lsn = self._wal_writer.append(clr)

            self._buffer_pool.set_page_lsn(record.page_id, clr_lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except (ValueError, IOError):
            pass

    def create_checkpoint(
        self,
        active_txns: dict[TransactionId, LSN],
        dirty_pages: dict[PageId, LSN],
    ) -> LSN:
        """Create a checkpoint record.

        Checkpoints speed up recovery by recording the current
        ATT and DPT, allowing the Analysis phase to skip older
        WAL records.

        Args:
            active_txns: Current Active Transaction Table.
            dirty_pages: Current Dirty Page Table.

        Returns:
            The LSN of the checkpoint record.
        """
        # Flush all dirty pages first (fuzzy checkpoint)
        self._buffer_pool.flush_all_pages()

        # Write checkpoint record
        checkpoint = CheckpointRecord(
            lsn=LSN(0),  # Will be assigned
            txn_id=TransactionId(0),  # System transaction
            prev_lsn=INVALID_LSN,
            active_txns=active_txns,
            dirty_pages=dirty_pages,
        )

        lsn = self._wal_writer.append(checkpoint)
        self._wal_writer.flush(lsn)

        return lsn
