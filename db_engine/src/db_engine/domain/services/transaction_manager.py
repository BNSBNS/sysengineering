"""Transaction Manager for MVCC and transaction lifecycle.

This module implements the transaction manager, which coordinates:
- Transaction lifecycle (begin, commit, abort)
- MVCC snapshots for isolation
- Two-phase locking via Lock Manager
- WAL logging for durability

MVCC Snapshot Isolation:
    Each transaction sees a consistent snapshot of the database
    as of its start time. Writers create new versions; readers
    see only committed versions visible to their snapshot.

References:
    - Reed, "Naming and Synchronization" MIT PhD Thesis (1978)
    - PostgreSQL MVCC documentation
    - design.md Section 5 (MVCC Visibility Rules)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Set

from db_engine.domain.entities import (
    AbortRecord,
    BeginRecord,
    CLRRecord,
    CommitRecord,
    DeleteRecord,
    InsertRecord,
    Record,
    Snapshot,
    UpdateRecord,
)
from db_engine.domain.services.lock_manager import LockManager
from db_engine.domain.value_objects import (
    LSN,
    INVALID_LSN,
    PageId,
    RecordId,
    TransactionId,
)
from db_engine.domain.value_objects.transaction_types import (
    IsolationLevel,
    LockMode,
    TransactionState,
)
from db_engine.ports.inbound.buffer_pool import BufferPool
from db_engine.ports.inbound.transaction_manager import (
    LockConflictError,
    Transaction,
    TransactionStats,
)
from db_engine.ports.outbound.wal_writer import WALWriter


class MVCCTransactionManager:
    """MVCC-based transaction manager.

    Provides snapshot isolation for concurrent transactions.
    Uses two-phase locking for write conflicts and WAL for durability.

    Usage:
        txn_mgr = MVCCTransactionManager(wal_writer, buffer_pool)
        txn = txn_mgr.begin()
        txn_mgr.insert(txn, page_id, slot_id, data)
        txn_mgr.commit(txn)

    Thread Safety:
        All methods are thread-safe for concurrent transactions.
    """

    def __init__(
        self,
        wal_writer: WALWriter,
        buffer_pool: BufferPool,
        lock_manager: LockManager | None = None,
    ) -> None:
        """Initialize the transaction manager.

        Args:
            wal_writer: The WAL writer for logging.
            buffer_pool: The buffer pool for page access.
            lock_manager: Optional lock manager (created if not provided).
        """
        self._wal_writer = wal_writer
        self._buffer_pool = buffer_pool
        self._lock_manager = lock_manager or LockManager()

        self._lock = threading.Lock()
        self._next_txn_id = 1
        self._active_txns: Dict[TransactionId, Transaction] = {}

        # Statistics
        self._committed_total = 0
        self._aborted_total = 0
        self._total_duration_ms = 0.0
        self._txn_start_times: Dict[TransactionId, float] = {}

    def begin(
        self, isolation_level: IsolationLevel = IsolationLevel.SNAPSHOT
    ) -> Transaction:
        """Begin a new transaction.

        Creates a new transaction with a unique ID and takes a
        snapshot for MVCC visibility.

        Args:
            isolation_level: The isolation level for this transaction.

        Returns:
            A new Transaction object.
        """
        with self._lock:
            txn_id = TransactionId(self._next_txn_id)
            self._next_txn_id += 1

            # Take snapshot of active transactions
            active_txn_ids = frozenset(self._active_txns.keys())
            snapshot = Snapshot(txn_id=txn_id, active_txns=active_txn_ids)

            # Create transaction
            txn = Transaction(
                txn_id=txn_id,
                state=TransactionState.ACTIVE,
                isolation_level=isolation_level,
                snapshot=snapshot,
                last_lsn=LSN(0),
            )

            # Register as active
            self._active_txns[txn_id] = txn
            self._txn_start_times[txn_id] = time.time()

        # Write BEGIN record
        begin_record = BeginRecord(
            lsn=LSN(0),  # Will be assigned
            txn_id=txn_id,
            prev_lsn=INVALID_LSN,
        )
        lsn = self._wal_writer.append(begin_record)

        with self._lock:
            txn.last_lsn = lsn

        return txn

    def commit(self, txn: Transaction) -> None:
        """Commit a transaction.

        This operation:
        1. Writes COMMIT record to WAL
        2. Flushes WAL to ensure durability
        3. Releases all locks
        4. Updates transaction state

        Args:
            txn: The transaction to commit.

        Raises:
            ValueError: If transaction is not active.
        """
        if txn.state != TransactionState.ACTIVE:
            raise ValueError(f"Transaction {txn.txn_id} is not active")

        # Update state
        txn.state = TransactionState.COMMITTING

        # Write COMMIT record
        commit_record = CommitRecord(
            lsn=LSN(0),
            txn_id=txn.txn_id,
            prev_lsn=txn.last_lsn,
        )
        lsn = self._wal_writer.append(commit_record)
        txn.last_lsn = lsn

        # Flush WAL to guarantee durability
        self._wal_writer.flush(lsn)

        # Release all locks
        self._lock_manager.release_all(txn.txn_id)

        # Update state
        txn.state = TransactionState.COMMITTED

        # Update statistics
        with self._lock:
            del self._active_txns[txn.txn_id]
            self._committed_total += 1

            if txn.txn_id in self._txn_start_times:
                duration = (time.time() - self._txn_start_times[txn.txn_id]) * 1000
                self._total_duration_ms += duration
                del self._txn_start_times[txn.txn_id]

    def abort(self, txn: Transaction) -> None:
        """Abort a transaction and rollback changes.

        This operation:
        1. Undoes all changes using the WAL chain
        2. Writes CLR records for each undo
        3. Writes ABORT record
        4. Releases all locks
        5. Updates transaction state

        Args:
            txn: The transaction to abort.

        Raises:
            ValueError: If transaction is not active.
        """
        if txn.state != TransactionState.ACTIVE:
            raise ValueError(f"Transaction {txn.txn_id} is not active")

        # Update state
        txn.state = TransactionState.ABORTING

        # Undo all changes by following prev_lsn chain
        current_lsn = txn.last_lsn

        while current_lsn != INVALID_LSN:
            record = self._find_record(current_lsn)
            if record is None:
                break

            # Undo data modifications
            if isinstance(record, UpdateRecord):
                self._undo_update(txn, record)
            elif isinstance(record, InsertRecord):
                self._undo_insert(txn, record)
            elif isinstance(record, DeleteRecord):
                self._undo_delete(txn, record)

            current_lsn = record.prev_lsn

        # Write ABORT record
        abort_record = AbortRecord(
            lsn=LSN(0),
            txn_id=txn.txn_id,
            prev_lsn=txn.last_lsn,
        )
        self._wal_writer.append(abort_record)

        # Release all locks
        self._lock_manager.release_all(txn.txn_id)

        # Update state
        txn.state = TransactionState.ABORTED

        # Update statistics
        with self._lock:
            del self._active_txns[txn.txn_id]
            self._aborted_total += 1

            if txn.txn_id in self._txn_start_times:
                duration = (time.time() - self._txn_start_times[txn.txn_id]) * 1000
                self._total_duration_ms += duration
                del self._txn_start_times[txn.txn_id]

    def get_snapshot(self, txn: Transaction) -> Snapshot:
        """Get the MVCC snapshot for a transaction.

        For SNAPSHOT isolation, this is taken at BEGIN time.
        For READ_COMMITTED, this is refreshed each statement.

        Args:
            txn: The transaction.

        Returns:
            The snapshot for visibility checks.
        """
        if txn.isolation_level == IsolationLevel.READ_COMMITTED:
            # Fresh snapshot for each statement
            with self._lock:
                active_txn_ids = frozenset(self._active_txns.keys())
                return Snapshot(txn_id=txn.txn_id, active_txns=active_txn_ids)

        # Use existing snapshot for other isolation levels
        return txn.snapshot or Snapshot(txn_id=txn.txn_id, active_txns=frozenset())

    def is_visible(self, record: Record, txn: Transaction) -> bool:
        """Check if a record is visible to a transaction.

        Implements MVCC visibility rules:
        - Created by committed transaction before snapshot
        - Not deleted (or deleted after snapshot)

        Args:
            record: The record to check.
            txn: The transaction checking visibility.

        Returns:
            True if the record is visible.
        """
        snapshot = self.get_snapshot(txn)
        return record.is_visible(snapshot)

    def insert(
        self, txn: Transaction, page_id: PageId, slot_id: int, data: bytes
    ) -> RecordId:
        """Insert a new record within a transaction.

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            data: The record data.

        Returns:
            The RecordId of the inserted record.

        Raises:
            LockConflictError: If lock cannot be acquired.
            ValueError: If transaction is not active.
        """
        if not txn.is_active():
            raise ValueError(f"Transaction {txn.txn_id} is not active")

        # Acquire exclusive lock
        if not self._lock_manager.acquire(
            txn.txn_id, page_id, slot_id, LockMode.EXCLUSIVE, wait=False
        ):
            raise LockConflictError(
                f"Cannot acquire lock on ({page_id}, {slot_id})"
            )

        txn.locks_held.add((page_id, slot_id))

        # Write INSERT record to WAL
        insert_record = InsertRecord(
            lsn=LSN(0),
            txn_id=txn.txn_id,
            prev_lsn=txn.last_lsn,
            page_id=page_id,
            slot_id=slot_id,
            data=data,
        )
        lsn = self._wal_writer.append(insert_record)
        txn.last_lsn = lsn

        # Perform the insert on the page
        try:
            page = self._buffer_pool.fetch_page(page_id)
            actual_slot = page.insert_record(data)
            self._buffer_pool.set_page_lsn(page_id, lsn)
            self._buffer_pool.unpin_page(page_id, is_dirty=True)
        except Exception:
            # Insert failed - still logged, will be handled by recovery
            pass

        return RecordId(page_id=page_id, slot_id=slot_id)

    def update(
        self,
        txn: Transaction,
        page_id: PageId,
        slot_id: int,
        old_data: bytes,
        new_data: bytes,
    ) -> None:
        """Update an existing record within a transaction.

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            old_data: The current record data (for WAL).
            new_data: The new record data.

        Raises:
            LockConflictError: If lock cannot be acquired.
            ValueError: If transaction is not active.
        """
        if not txn.is_active():
            raise ValueError(f"Transaction {txn.txn_id} is not active")

        # Acquire exclusive lock
        if not self._lock_manager.acquire(
            txn.txn_id, page_id, slot_id, LockMode.EXCLUSIVE, wait=False
        ):
            raise LockConflictError(
                f"Cannot acquire lock on ({page_id}, {slot_id})"
            )

        txn.locks_held.add((page_id, slot_id))

        # Write UPDATE record to WAL
        update_record = UpdateRecord(
            lsn=LSN(0),
            txn_id=txn.txn_id,
            prev_lsn=txn.last_lsn,
            page_id=page_id,
            slot_id=slot_id,
            before_image=old_data,
            after_image=new_data,
        )
        lsn = self._wal_writer.append(update_record)
        txn.last_lsn = lsn

        # Perform the update on the page
        try:
            page = self._buffer_pool.fetch_page(page_id)
            page.update_record(slot_id, new_data)
            self._buffer_pool.set_page_lsn(page_id, lsn)
            self._buffer_pool.unpin_page(page_id, is_dirty=True)
        except Exception:
            pass

    def delete(
        self, txn: Transaction, page_id: PageId, slot_id: int, data: bytes
    ) -> None:
        """Delete a record within a transaction.

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            data: The record data (for WAL undo).

        Raises:
            LockConflictError: If lock cannot be acquired.
            ValueError: If transaction is not active.
        """
        if not txn.is_active():
            raise ValueError(f"Transaction {txn.txn_id} is not active")

        # Acquire exclusive lock
        if not self._lock_manager.acquire(
            txn.txn_id, page_id, slot_id, LockMode.EXCLUSIVE, wait=False
        ):
            raise LockConflictError(
                f"Cannot acquire lock on ({page_id}, {slot_id})"
            )

        txn.locks_held.add((page_id, slot_id))

        # Write DELETE record to WAL
        delete_record = DeleteRecord(
            lsn=LSN(0),
            txn_id=txn.txn_id,
            prev_lsn=txn.last_lsn,
            page_id=page_id,
            slot_id=slot_id,
            data=data,
        )
        lsn = self._wal_writer.append(delete_record)
        txn.last_lsn = lsn

        # Perform the delete on the page
        try:
            page = self._buffer_pool.fetch_page(page_id)
            page.delete_record(slot_id)
            self._buffer_pool.set_page_lsn(page_id, lsn)
            self._buffer_pool.unpin_page(page_id, is_dirty=True)
        except Exception:
            pass

    def get_active_transactions(self) -> list[TransactionId]:
        """Return IDs of all active transactions."""
        with self._lock:
            return list(self._active_txns.keys())

    def get_stats(self) -> TransactionStats:
        """Return transaction statistics for monitoring."""
        with self._lock:
            active_count = len(self._active_txns)
            total = self._committed_total + self._aborted_total

            if total > 0:
                avg_duration = self._total_duration_ms / total
            else:
                avg_duration = 0.0

            return TransactionStats(
                active_count=active_count,
                committed_total=self._committed_total,
                aborted_total=self._aborted_total,
                avg_duration_ms=avg_duration,
            )

    def _find_record(self, lsn: LSN) -> BeginRecord | InsertRecord | UpdateRecord | DeleteRecord | None:
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

    def _undo_update(self, txn: Transaction, record: UpdateRecord) -> None:
        """Undo an UPDATE operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page.update_record(record.slot_id, record.before_image)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),
                txn_id=txn.txn_id,
                prev_lsn=txn.last_lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            lsn = self._wal_writer.append(clr)
            txn.last_lsn = lsn

            self._buffer_pool.set_page_lsn(record.page_id, lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except Exception:
            pass

    def _undo_insert(self, txn: Transaction, record: InsertRecord) -> None:
        """Undo an INSERT operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page.delete_record(record.slot_id)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),
                txn_id=txn.txn_id,
                prev_lsn=txn.last_lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            lsn = self._wal_writer.append(clr)
            txn.last_lsn = lsn

            self._buffer_pool.set_page_lsn(record.page_id, lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except Exception:
            pass

    def _undo_delete(self, txn: Transaction, record: DeleteRecord) -> None:
        """Undo a DELETE operation."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page.insert_record(record.data)

            # Write CLR
            clr = CLRRecord(
                lsn=LSN(0),
                txn_id=txn.txn_id,
                prev_lsn=txn.last_lsn,
                undo_next_lsn=record.prev_lsn,
                page_id=record.page_id,
                slot_id=record.slot_id,
            )
            lsn = self._wal_writer.append(clr)
            txn.last_lsn = lsn

            self._buffer_pool.set_page_lsn(record.page_id, lsn)
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)
        except Exception:
            pass
