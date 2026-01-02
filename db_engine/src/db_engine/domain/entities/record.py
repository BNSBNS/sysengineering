"""Record entity with MVCC (Multi-Version Concurrency Control) metadata.

Each record in the database contains both the actual data and MVCC metadata
that enables snapshot isolation. The metadata tracks which transaction created
the record and which transaction (if any) deleted it.

References:
    - design.md Section 5 (MVCC Visibility Rules)
    - Reed, D. "Naming and Synchronization in a Decentralized Computer System" (1978)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import ClassVar

from db_engine.domain.value_objects import (
    LSN,
    INVALID_LSN,
    INVALID_TXN_ID,
    TransactionId,
)


@dataclass
class RecordHeader:
    """MVCC metadata for each record.

    This header is stored as a prefix to every record and contains the
    transaction IDs that created and (optionally) deleted the record.
    These are used to determine visibility during query execution.

    Terminology (PostgreSQL equivalent):
        - created_by: xmin - transaction that created this tuple
        - deleted_by: xmax - transaction that deleted this tuple (or 0)
        - created_at_lsn: LSN when record was created (for recovery)
        - deleted_at_lsn: LSN when record was deleted (for recovery)

    Size: 32 bytes
        - created_by: 8 bytes (TransactionId)
        - deleted_by: 8 bytes (TransactionId, 0 if not deleted)
        - created_at_lsn: 8 bytes (LSN)
        - deleted_at_lsn: 8 bytes (LSN, 0 if not deleted)
    """

    created_by: TransactionId
    deleted_by: TransactionId
    created_at_lsn: LSN
    deleted_at_lsn: LSN

    HEADER_SIZE: ClassVar[int] = 32
    HEADER_FORMAT: ClassVar[str] = ">QQQQ"  # 4 x uint64

    def is_deleted(self) -> bool:
        """Check if this record has been marked as deleted."""
        return self.deleted_by != INVALID_TXN_ID

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self.HEADER_FORMAT,
            self.created_by,
            self.deleted_by,
            self.created_at_lsn,
            self.deleted_at_lsn,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> RecordHeader:
        """Deserialize header from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"RecordHeader requires {cls.HEADER_SIZE} bytes, got {len(data)}")

        created_by, deleted_by, created_at_lsn, deleted_at_lsn = struct.unpack(
            cls.HEADER_FORMAT, data[: cls.HEADER_SIZE]
        )

        return cls(
            created_by=TransactionId(created_by),
            deleted_by=TransactionId(deleted_by),
            created_at_lsn=LSN(created_at_lsn),
            deleted_at_lsn=LSN(deleted_at_lsn),
        )

    @classmethod
    def new(cls, created_by: TransactionId, created_at_lsn: LSN) -> RecordHeader:
        """Create a new record header for a freshly inserted record."""
        return cls(
            created_by=created_by,
            deleted_by=INVALID_TXN_ID,
            created_at_lsn=created_at_lsn,
            deleted_at_lsn=INVALID_LSN,
        )

    def mark_deleted(self, deleted_by: TransactionId, deleted_at_lsn: LSN) -> RecordHeader:
        """Create a new header with deletion metadata.

        Returns a new RecordHeader (immutable pattern) with the deletion
        transaction and LSN set.
        """
        return RecordHeader(
            created_by=self.created_by,
            deleted_by=deleted_by,
            created_at_lsn=self.created_at_lsn,
            deleted_at_lsn=deleted_at_lsn,
        )


@dataclass
class Record:
    """A database record with MVCC metadata and payload.

    Records are stored in slotted pages. The format is:
        [RecordHeader (32 bytes)] [Data (variable length)]

    The header contains MVCC metadata for visibility determination.
    The data is the actual record content (tuple data).

    Example:
        >>> txn_id = TransactionId(1)
        >>> lsn = LSN(100)
        >>> record = Record.new(txn_id, lsn, b"Hello, World!")
        >>> record.header.created_by
        TransactionId(1)
        >>> record.data
        b'Hello, World!'
    """

    header: RecordHeader
    data: bytes

    def __len__(self) -> int:
        """Total size of record including header."""
        return RecordHeader.HEADER_SIZE + len(self.data)

    def to_bytes(self) -> bytes:
        """Serialize record to bytes for storage."""
        return self.header.to_bytes() + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> Record:
        """Deserialize record from bytes."""
        if len(data) < RecordHeader.HEADER_SIZE:
            raise ValueError(
                f"Record requires at least {RecordHeader.HEADER_SIZE} bytes, got {len(data)}"
            )

        header = RecordHeader.from_bytes(data[: RecordHeader.HEADER_SIZE])
        payload = data[RecordHeader.HEADER_SIZE:]

        return cls(header=header, data=payload)

    @classmethod
    def new(cls, created_by: TransactionId, created_at_lsn: LSN, data: bytes) -> Record:
        """Create a new record with fresh MVCC metadata.

        Args:
            created_by: The transaction creating this record
            created_at_lsn: The LSN of the insert operation
            data: The actual record data

        Returns:
            A new Record instance
        """
        header = RecordHeader.new(created_by, created_at_lsn)
        return cls(header=header, data=data)

    def mark_deleted(self, deleted_by: TransactionId, deleted_at_lsn: LSN) -> Record:
        """Create a new record with deletion metadata set.

        This doesn't actually remove the record - it marks it as deleted
        so that future transactions won't see it, while still allowing
        transactions that started before the deletion to see it.

        Args:
            deleted_by: The transaction deleting this record
            deleted_at_lsn: The LSN of the delete operation

        Returns:
            A new Record with updated header
        """
        new_header = self.header.mark_deleted(deleted_by, deleted_at_lsn)
        return Record(header=new_header, data=self.data)

    def is_visible_to(self, snapshot: Snapshot) -> bool:
        """Check if this record is visible to the given snapshot.

        Implements the MVCC visibility rules from design.md Section 5:

        1. Record was created by a committed transaction that committed
           before the snapshot was taken
        2. Record was not deleted, OR was deleted by a transaction that
           started after the snapshot was taken

        Args:
            snapshot: The transaction snapshot to check visibility against

        Returns:
            True if the record is visible to the snapshot
        """
        # Created after snapshot? Not visible
        if self.header.created_by > snapshot.txn_id:
            return False

        # Created by concurrent transaction? Not visible
        if self.header.created_by in snapshot.active_txns:
            return False

        # Not deleted? Visible
        if not self.header.is_deleted():
            return True

        # Deleted after snapshot? Still visible
        if self.header.deleted_by > snapshot.txn_id:
            return True

        # Deleted by concurrent transaction? Still visible
        if self.header.deleted_by in snapshot.active_txns:
            return True

        # Deleted before snapshot by committed txn - not visible
        return False


@dataclass(frozen=True)
class Snapshot:
    """A point-in-time view of the database for MVCC.

    A snapshot captures:
    - The transaction ID of the transaction taking the snapshot
    - The set of transaction IDs that were active (uncommitted) at snapshot time

    Records created by transactions in active_txns are not visible,
    even if those transactions have IDs lower than txn_id.

    This enables snapshot isolation without blocking.
    """

    txn_id: TransactionId
    active_txns: frozenset[TransactionId] = field(default_factory=frozenset)

    @classmethod
    def new(
        cls,
        txn_id: TransactionId,
        active_txns: set[TransactionId] | frozenset[TransactionId] | None = None,
    ) -> Snapshot:
        """Create a new snapshot.

        Args:
            txn_id: The transaction taking the snapshot
            active_txns: Set of currently active transaction IDs

        Returns:
            A new Snapshot instance
        """
        if active_txns is None:
            active_txns = frozenset()
        elif isinstance(active_txns, set):
            active_txns = frozenset(active_txns)

        return cls(txn_id=txn_id, active_txns=active_txns)
