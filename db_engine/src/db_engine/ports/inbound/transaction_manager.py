"""Transaction Manager port for MVCC and transaction lifecycle.

This inbound port defines the contract for transaction management,
including begin/commit/abort operations and snapshot isolation.

Key responsibilities:
- Manage transaction lifecycle
- Provide MVCC snapshots for isolation
- Coordinate with lock manager for 2PL
- Write transaction log records

References:
    - design.md Section 3 (Transaction Manager)
    - design.md Section 5 (MVCC Visibility Rules)
    - Reed, "Naming and Synchronization" (1978)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from db_engine.domain.entities import Record, Snapshot
from db_engine.domain.value_objects import (
    LSN,
    PageId,
    RecordId,
    TransactionId,
)
from db_engine.domain.value_objects.transaction_types import (
    IsolationLevel,
    TransactionState,
)


@dataclass
class Transaction:
    """A database transaction.

    Transactions provide ACID guarantees:
    - Atomicity: All-or-nothing execution
    - Consistency: Only valid states
    - Isolation: Concurrent transactions don't interfere
    - Durability: Committed changes survive crashes
    """

    txn_id: TransactionId
    state: TransactionState = TransactionState.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.SNAPSHOT
    snapshot: Snapshot | None = None
    last_lsn: LSN = LSN(0)
    locks_held: set[tuple[PageId, int]] = field(default_factory=set)  # (page_id, slot_id)

    def is_active(self) -> bool:
        """Return True if transaction can still perform operations."""
        return self.state == TransactionState.ACTIVE

    def is_terminal(self) -> bool:
        """Return True if transaction has ended."""
        return self.state in (TransactionState.COMMITTED, TransactionState.ABORTED)


@dataclass
class TransactionStats:
    """Statistics for transaction monitoring."""

    active_count: int
    committed_total: int
    aborted_total: int
    avg_duration_ms: float


class TransactionManager(Protocol):
    """Protocol for transaction management.

    The transaction manager coordinates MVCC, locking, and logging
    to provide isolation between concurrent transactions.

    Isolation levels supported:
    - READ_UNCOMMITTED: No MVCC checks (dirty reads allowed)
    - READ_COMMITTED: Fresh snapshot per statement
    - REPEATABLE_READ: Same snapshot for entire transaction
    - SNAPSHOT: Same as REPEATABLE_READ in our implementation
    - SERIALIZABLE: 2PL with predicate locks (strictest)

    Thread Safety:
        All methods must be thread-safe for concurrent transactions.
    """

    @abstractmethod
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
        ...

    @abstractmethod
    def commit(self, txn: Transaction) -> None:
        """Commit a transaction.

        This operation:
        1. Writes COMMIT record to WAL
        2. Flushes WAL to ensure durability
        3. Releases all locks
        4. Updates transaction state

        After commit returns, the transaction is guaranteed durable.

        Args:
            txn: The transaction to commit.

        Raises:
            ValueError: If transaction is not active.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def get_snapshot(self, txn: Transaction) -> Snapshot:
        """Get the MVCC snapshot for a transaction.

        For SNAPSHOT isolation, this is taken at BEGIN time.
        For READ_COMMITTED, this is refreshed each statement.

        Args:
            txn: The transaction.

        Returns:
            The snapshot for visibility checks.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def insert(
        self, txn: Transaction, page_id: PageId, slot_id: int, data: bytes
    ) -> RecordId:
        """Insert a new record within a transaction.

        This operation:
        1. Acquires exclusive lock on the slot
        2. Writes INSERT record to WAL
        3. Creates the record with MVCC metadata
        4. Updates transaction's last_lsn

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            data: The record data.

        Returns:
            The RecordId of the inserted record.

        Raises:
            LockConflictError: If lock cannot be acquired.
        """
        ...

    @abstractmethod
    def update(
        self,
        txn: Transaction,
        page_id: PageId,
        slot_id: int,
        old_data: bytes,
        new_data: bytes,
    ) -> None:
        """Update an existing record within a transaction.

        This operation:
        1. Acquires exclusive lock on the slot
        2. Writes UPDATE record to WAL with before/after images
        3. Modifies the record
        4. Updates transaction's last_lsn

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            old_data: The current record data (for WAL).
            new_data: The new record data.

        Raises:
            LockConflictError: If lock cannot be acquired.
            ValueError: If record not found or not visible.
        """
        ...

    @abstractmethod
    def delete(
        self, txn: Transaction, page_id: PageId, slot_id: int, data: bytes
    ) -> None:
        """Delete a record within a transaction.

        This operation:
        1. Acquires exclusive lock on the slot
        2. Writes DELETE record to WAL
        3. Marks the record as deleted (sets deleted_by)
        4. Updates transaction's last_lsn

        Args:
            txn: The transaction.
            page_id: The target page.
            slot_id: The target slot.
            data: The record data (for WAL undo).

        Raises:
            LockConflictError: If lock cannot be acquired.
            ValueError: If record not found or not visible.
        """
        ...

    @abstractmethod
    def get_active_transactions(self) -> list[TransactionId]:
        """Return IDs of all active transactions.

        Used for snapshot creation and checkpoint.
        """
        ...

    @abstractmethod
    def get_stats(self) -> TransactionStats:
        """Return transaction statistics for monitoring."""
        ...


class LockConflictError(Exception):
    """Raised when a lock cannot be acquired.

    This may occur due to:
    - Conflicting lock held by another transaction
    - Deadlock detected
    - Lock wait timeout
    """

    def __init__(self, message: str, blocking_txn: TransactionId | None = None):
        super().__init__(message)
        self.blocking_txn = blocking_txn
