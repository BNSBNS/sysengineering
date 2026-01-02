"""Write-Ahead Log (WAL) record types for crash recovery.

This module defines all WAL record types used by the ARIES-style recovery
algorithm. Each record type maps to a specific recovery action:

    Record Type | Purpose                | Recovery Action
    ------------|------------------------|------------------
    BEGIN       | Start transaction      | Add to active set
    COMMIT      | End successfully       | Remove from active, no undo needed
    ABORT       | End unsuccessfully     | Trigger undo
    UPDATE      | Data modification      | Redo if committed, undo if not
    INSERT      | New record creation    | Redo if committed, undo if not
    DELETE      | Record deletion        | Redo if committed, undo if not
    CLR         | Compensation record    | Never undo (marks undo progress)
    CHECKPOINT  | Recovery optimization  | Analysis starts here

References:
    - design.md Section 4 (WAL Record Types)
    - Mohan, C. et al. "ARIES: A Transaction Recovery Method" (1992)
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from db_engine.domain.value_objects import LSN, INVALID_LSN, PageId, TransactionId


class LogRecordType(IntEnum):
    """Enumeration of WAL record types.

    Using IntEnum for efficient serialization (single byte).
    """

    BEGIN = 1
    COMMIT = 2
    ABORT = 3
    UPDATE = 4
    INSERT = 5
    DELETE = 6
    CLR = 7
    CHECKPOINT = 8


@dataclass
class LogRecord(ABC):
    """Base class for all WAL records.

    Every log record has:
    - lsn: The Log Sequence Number (assigned when appended to WAL)
    - txn_id: The transaction that generated this record
    - prev_lsn: The previous LSN for this transaction (for undo chain)

    The LSN forms a total order on all WAL records. The prev_lsn
    chains together all records from the same transaction, enabling
    efficient rollback.
    """

    lsn: LSN
    txn_id: TransactionId
    prev_lsn: LSN

    # Common header: type(1) + lsn(8) + txn_id(8) + prev_lsn(8) = 25 bytes
    COMMON_HEADER_SIZE: ClassVar[int] = 25
    COMMON_HEADER_FORMAT: ClassVar[str] = ">BQQQ"

    @property
    @abstractmethod
    def record_type(self) -> LogRecordType:
        """Return the type of this log record."""
        ...

    @abstractmethod
    def payload_to_bytes(self) -> bytes:
        """Serialize the record-specific payload to bytes."""
        ...

    @classmethod
    @abstractmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> LogRecord:
        """Deserialize the record-specific payload from bytes."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the complete record to bytes.

        Format: [common_header][payload]
        Common header: type(1) + lsn(8) + txn_id(8) + prev_lsn(8)
        """
        header = struct.pack(
            self.COMMON_HEADER_FORMAT,
            self.record_type,
            self.lsn,
            self.txn_id,
            self.prev_lsn,
        )
        return header + self.payload_to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> LogRecord:
        """Deserialize a log record from bytes.

        This is a factory method that determines the record type and
        delegates to the appropriate subclass.
        """
        if len(data) < cls.COMMON_HEADER_SIZE:
            raise ValueError(
                f"LogRecord requires at least {cls.COMMON_HEADER_SIZE} bytes, got {len(data)}"
            )

        record_type_int, lsn, txn_id, prev_lsn = struct.unpack(
            cls.COMMON_HEADER_FORMAT, data[: cls.COMMON_HEADER_SIZE]
        )

        try:
            record_type = LogRecordType(record_type_int)
        except ValueError:
            raise ValueError(f"Unknown record type: {record_type_int}")
        payload = data[cls.COMMON_HEADER_SIZE:]

        # Dispatch to appropriate subclass
        record_classes: dict[LogRecordType, type[LogRecord]] = {
            LogRecordType.BEGIN: BeginRecord,
            LogRecordType.COMMIT: CommitRecord,
            LogRecordType.ABORT: AbortRecord,
            LogRecordType.UPDATE: UpdateRecord,
            LogRecordType.INSERT: InsertRecord,
            LogRecordType.DELETE: DeleteRecord,
            LogRecordType.CLR: CLRRecord,
            LogRecordType.CHECKPOINT: CheckpointRecord,
        }

        record_class = record_classes.get(record_type)
        if record_class is None:
            raise ValueError(f"Unknown record type: {record_type}")

        return record_class.payload_from_bytes(
            LSN(lsn), TransactionId(txn_id), LSN(prev_lsn), payload
        )


@dataclass
class BeginRecord(LogRecord):
    """Transaction begin marker.

    Written when a transaction starts. Used during recovery to track
    which transactions were active.
    """

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.BEGIN

    def payload_to_bytes(self) -> bytes:
        return b""

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> BeginRecord:
        return cls(lsn=lsn, txn_id=txn_id, prev_lsn=prev_lsn)


@dataclass
class CommitRecord(LogRecord):
    """Transaction commit marker.

    Written when a transaction commits. The commit is durable once this
    record is flushed to disk. During recovery, transactions with a
    COMMIT record are winners (no undo needed).
    """

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.COMMIT

    def payload_to_bytes(self) -> bytes:
        return b""

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> CommitRecord:
        return cls(lsn=lsn, txn_id=txn_id, prev_lsn=prev_lsn)


@dataclass
class AbortRecord(LogRecord):
    """Transaction abort marker.

    Written when a transaction aborts. During recovery, transactions
    with an ABORT record are losers that have already been rolled back.
    """

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.ABORT

    def payload_to_bytes(self) -> bytes:
        return b""

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> AbortRecord:
        return cls(lsn=lsn, txn_id=txn_id, prev_lsn=prev_lsn)


@dataclass
class UpdateRecord(LogRecord):
    """Data modification record with before/after images.

    Contains both the old value (before_image) for undo and the new
    value (after_image) for redo. This is a physiological logging
    approach - we log the physical changes but at a logical level.

    Attributes:
        page_id: The page being modified
        slot_id: The slot within the page
        before_image: Old value (for undo)
        after_image: New value (for redo)
    """

    page_id: PageId
    slot_id: int
    before_image: bytes
    after_image: bytes

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.UPDATE

    def payload_to_bytes(self) -> bytes:
        # page_id(4) + slot_id(2) + before_len(4) + after_len(4) + before + after
        header = struct.pack(
            ">iHII",
            self.page_id,
            self.slot_id,
            len(self.before_image),
            len(self.after_image),
        )
        return header + self.before_image + self.after_image

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> UpdateRecord:
        page_id, slot_id, before_len, after_len = struct.unpack(">iHII", data[:14])
        before_image = data[14 : 14 + before_len]
        after_image = data[14 + before_len : 14 + before_len + after_len]

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            page_id=PageId(page_id),
            slot_id=slot_id,
            before_image=before_image,
            after_image=after_image,
        )


@dataclass
class InsertRecord(LogRecord):
    """New record insertion.

    Contains the inserted data. For undo, we delete the record.
    For redo, we re-insert it.

    Attributes:
        page_id: The page where record was inserted
        slot_id: The slot assigned to the record
        data: The record data that was inserted
    """

    page_id: PageId
    slot_id: int
    data: bytes

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.INSERT

    def payload_to_bytes(self) -> bytes:
        # page_id(4) + slot_id(2) + data_len(4) + data
        header = struct.pack(">iHI", self.page_id, self.slot_id, len(self.data))
        return header + self.data

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> InsertRecord:
        page_id, slot_id, data_len = struct.unpack(">iHI", data[:10])
        record_data = data[10 : 10 + data_len]

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            page_id=PageId(page_id),
            slot_id=slot_id,
            data=record_data,
        )


@dataclass
class DeleteRecord(LogRecord):
    """Record deletion.

    Contains the deleted data so we can undo the deletion if needed.

    Attributes:
        page_id: The page where record was deleted
        slot_id: The slot that was deleted
        data: The record data that was deleted (for undo)
    """

    page_id: PageId
    slot_id: int
    data: bytes

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.DELETE

    def payload_to_bytes(self) -> bytes:
        # page_id(4) + slot_id(2) + data_len(4) + data
        header = struct.pack(">iHI", self.page_id, self.slot_id, len(self.data))
        return header + self.data

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> DeleteRecord:
        page_id, slot_id, data_len = struct.unpack(">iHI", data[:10])
        record_data = data[10 : 10 + data_len]

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            page_id=PageId(page_id),
            slot_id=slot_id,
            data=record_data,
        )


@dataclass
class CLRRecord(LogRecord):
    """Compensation Log Record - marks undo progress.

    CLRs are written during undo to ensure idempotence. If we crash
    during recovery and restart, we don't re-undo operations that
    have already been undone.

    The undo_next_lsn points to the next record to undo for this
    transaction (skipping the record that was just compensated).

    Key property: CLRs are NEVER undone. They only have redo actions.

    Attributes:
        undo_next_lsn: LSN of the next record to undo (or INVALID_LSN if done)
        page_id: The page that was modified by this compensation
        slot_id: The slot that was modified
    """

    undo_next_lsn: LSN
    page_id: PageId
    slot_id: int

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.CLR

    def payload_to_bytes(self) -> bytes:
        # undo_next_lsn(8) + page_id(4) + slot_id(2)
        return struct.pack(">QiH", self.undo_next_lsn, self.page_id, self.slot_id)

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> CLRRecord:
        undo_next_lsn, page_id, slot_id = struct.unpack(">QiH", data[:14])

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            undo_next_lsn=LSN(undo_next_lsn),
            page_id=PageId(page_id),
            slot_id=slot_id,
        )


@dataclass
class CheckpointRecord(LogRecord):
    """Checkpoint for recovery optimization.

    Checkpoints speed up recovery by recording the current state:
    - Active Transaction Table (ATT): txn_id -> lastLSN
    - Dirty Page Table (DPT): page_id -> recLSN

    During recovery, we can start analysis from the checkpoint instead
    of scanning the entire WAL.

    We use fuzzy checkpointing: dirty pages are NOT flushed at
    checkpoint time. The DPT records which pages might be dirty.

    Attributes:
        active_txns: Map of active transaction IDs to their last LSN
        dirty_pages: Map of dirty page IDs to their recovery LSN
    """

    active_txns: dict[TransactionId, LSN] = field(default_factory=dict)
    dirty_pages: dict[PageId, LSN] = field(default_factory=dict)

    @property
    def record_type(self) -> LogRecordType:
        return LogRecordType.CHECKPOINT

    def payload_to_bytes(self) -> bytes:
        # Format: att_count(4) + dpt_count(4) + [att entries] + [dpt entries]
        # ATT entry: txn_id(8) + lsn(8) = 16 bytes
        # DPT entry: page_id(4) + lsn(8) = 12 bytes

        parts = [struct.pack(">II", len(self.active_txns), len(self.dirty_pages))]

        for txn_id, last_lsn in self.active_txns.items():
            parts.append(struct.pack(">QQ", txn_id, last_lsn))

        for page_id, rec_lsn in self.dirty_pages.items():
            parts.append(struct.pack(">iQ", page_id, rec_lsn))

        return b"".join(parts)

    @classmethod
    def payload_from_bytes(
        cls, lsn: LSN, txn_id: TransactionId, prev_lsn: LSN, data: bytes
    ) -> CheckpointRecord:
        att_count, dpt_count = struct.unpack(">II", data[:8])
        offset = 8

        active_txns: dict[TransactionId, LSN] = {}
        for _ in range(att_count):
            tid, last_lsn = struct.unpack(">QQ", data[offset : offset + 16])
            active_txns[TransactionId(tid)] = LSN(last_lsn)
            offset += 16

        dirty_pages: dict[PageId, LSN] = {}
        for _ in range(dpt_count):
            pid, rec_lsn = struct.unpack(">iQ", data[offset : offset + 12])
            dirty_pages[PageId(pid)] = LSN(rec_lsn)
            offset += 12

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            active_txns=active_txns,
            dirty_pages=dirty_pages,
        )
