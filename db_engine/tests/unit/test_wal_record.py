"""Unit tests for WAL record types."""

from __future__ import annotations

import pytest

from db_engine.domain.entities import (
    AbortRecord,
    BeginRecord,
    CheckpointRecord,
    CLRRecord,
    CommitRecord,
    DeleteRecord,
    InsertRecord,
    LogRecord,
    LogRecordType,
    UpdateRecord,
)
from db_engine.domain.value_objects import LSN, PageId, TransactionId


class TestLogRecordType:
    """Tests for LogRecordType enum."""

    def test_all_types_exist(self) -> None:
        """All expected record types are defined."""
        types = [
            LogRecordType.BEGIN,
            LogRecordType.COMMIT,
            LogRecordType.ABORT,
            LogRecordType.UPDATE,
            LogRecordType.INSERT,
            LogRecordType.DELETE,
            LogRecordType.CLR,
            LogRecordType.CHECKPOINT,
        ]
        assert len(types) == 8

    def test_types_are_integers(self) -> None:
        """Record types are small integers for efficient storage."""
        for record_type in LogRecordType:
            assert 1 <= record_type <= 255  # Fits in one byte


class TestBeginRecord:
    """Tests for transaction begin records."""

    def test_creation(self) -> None:
        """BeginRecord can be created."""
        record = BeginRecord(
            lsn=LSN(100),
            txn_id=TransactionId(42),
            prev_lsn=LSN(0),
        )

        assert record.lsn == LSN(100)
        assert record.txn_id == TransactionId(42)
        assert record.prev_lsn == LSN(0)
        assert record.record_type == LogRecordType.BEGIN

    def test_serialization_roundtrip(self) -> None:
        """BeginRecord serializes and deserializes correctly."""
        record = BeginRecord(
            lsn=LSN(100),
            txn_id=TransactionId(42),
            prev_lsn=LSN(50),
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, BeginRecord)
        assert restored.lsn == record.lsn
        assert restored.txn_id == record.txn_id
        assert restored.prev_lsn == record.prev_lsn


class TestCommitRecord:
    """Tests for transaction commit records."""

    def test_creation(self) -> None:
        """CommitRecord can be created."""
        record = CommitRecord(
            lsn=LSN(200),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
        )

        assert record.record_type == LogRecordType.COMMIT

    def test_serialization_roundtrip(self) -> None:
        """CommitRecord serializes and deserializes correctly."""
        record = CommitRecord(
            lsn=LSN(200),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, CommitRecord)
        assert restored.lsn == record.lsn
        assert restored.txn_id == record.txn_id


class TestAbortRecord:
    """Tests for transaction abort records."""

    def test_creation(self) -> None:
        """AbortRecord can be created."""
        record = AbortRecord(
            lsn=LSN(200),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
        )

        assert record.record_type == LogRecordType.ABORT

    def test_serialization_roundtrip(self) -> None:
        """AbortRecord serializes and deserializes correctly."""
        record = AbortRecord(
            lsn=LSN(200),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, AbortRecord)


class TestUpdateRecord:
    """Tests for data update records."""

    def test_creation(self) -> None:
        """UpdateRecord can be created with before/after images."""
        record = UpdateRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            before_image=b"old value",
            after_image=b"new value",
        )

        assert record.record_type == LogRecordType.UPDATE
        assert record.page_id == PageId(10)
        assert record.slot_id == 5
        assert record.before_image == b"old value"
        assert record.after_image == b"new value"

    def test_serialization_roundtrip(self) -> None:
        """UpdateRecord serializes and deserializes correctly."""
        record = UpdateRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            before_image=b"old value with \x00 binary",
            after_image=b"new value with \xff binary",
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, UpdateRecord)
        assert restored.page_id == record.page_id
        assert restored.slot_id == record.slot_id
        assert restored.before_image == record.before_image
        assert restored.after_image == record.after_image


class TestInsertRecord:
    """Tests for record insert records."""

    def test_creation(self) -> None:
        """InsertRecord can be created."""
        record = InsertRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            data=b"inserted data",
        )

        assert record.record_type == LogRecordType.INSERT
        assert record.data == b"inserted data"

    def test_serialization_roundtrip(self) -> None:
        """InsertRecord serializes and deserializes correctly."""
        record = InsertRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            data=b"inserted data \x00\xff",
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, InsertRecord)
        assert restored.page_id == record.page_id
        assert restored.slot_id == record.slot_id
        assert restored.data == record.data


class TestDeleteRecord:
    """Tests for record delete records."""

    def test_creation(self) -> None:
        """DeleteRecord can be created."""
        record = DeleteRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            data=b"deleted data",  # Saved for undo
        )

        assert record.record_type == LogRecordType.DELETE
        assert record.data == b"deleted data"

    def test_serialization_roundtrip(self) -> None:
        """DeleteRecord serializes and deserializes correctly."""
        record = DeleteRecord(
            lsn=LSN(150),
            txn_id=TransactionId(42),
            prev_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
            data=b"deleted data",
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, DeleteRecord)
        assert restored.data == record.data


class TestCLRRecord:
    """Tests for Compensation Log Records."""

    def test_creation(self) -> None:
        """CLRRecord can be created."""
        record = CLRRecord(
            lsn=LSN(300),
            txn_id=TransactionId(42),
            prev_lsn=LSN(200),
            undo_next_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
        )

        assert record.record_type == LogRecordType.CLR
        assert record.undo_next_lsn == LSN(100)

    def test_serialization_roundtrip(self) -> None:
        """CLRRecord serializes and deserializes correctly."""
        record = CLRRecord(
            lsn=LSN(300),
            txn_id=TransactionId(42),
            prev_lsn=LSN(200),
            undo_next_lsn=LSN(100),
            page_id=PageId(10),
            slot_id=5,
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, CLRRecord)
        assert restored.undo_next_lsn == record.undo_next_lsn
        assert restored.page_id == record.page_id
        assert restored.slot_id == record.slot_id


class TestCheckpointRecord:
    """Tests for checkpoint records."""

    def test_creation_empty(self) -> None:
        """CheckpointRecord can be created with empty tables."""
        record = CheckpointRecord(
            lsn=LSN(1000),
            txn_id=TransactionId(0),
            prev_lsn=LSN(0),
        )

        assert record.record_type == LogRecordType.CHECKPOINT
        assert len(record.active_txns) == 0
        assert len(record.dirty_pages) == 0

    def test_creation_with_data(self) -> None:
        """CheckpointRecord can be created with ATT and DPT."""
        record = CheckpointRecord(
            lsn=LSN(1000),
            txn_id=TransactionId(0),
            prev_lsn=LSN(0),
            active_txns={
                TransactionId(10): LSN(100),
                TransactionId(20): LSN(200),
            },
            dirty_pages={
                PageId(1): LSN(50),
                PageId(5): LSN(150),
                PageId(10): LSN(300),
            },
        )

        assert len(record.active_txns) == 2
        assert len(record.dirty_pages) == 3
        assert record.active_txns[TransactionId(10)] == LSN(100)
        assert record.dirty_pages[PageId(5)] == LSN(150)

    def test_serialization_roundtrip_empty(self) -> None:
        """Empty CheckpointRecord serializes correctly."""
        record = CheckpointRecord(
            lsn=LSN(1000),
            txn_id=TransactionId(0),
            prev_lsn=LSN(0),
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, CheckpointRecord)
        assert len(restored.active_txns) == 0
        assert len(restored.dirty_pages) == 0

    def test_serialization_roundtrip_with_data(self) -> None:
        """CheckpointRecord with data serializes correctly."""
        record = CheckpointRecord(
            lsn=LSN(1000),
            txn_id=TransactionId(0),
            prev_lsn=LSN(500),
            active_txns={
                TransactionId(10): LSN(100),
                TransactionId(20): LSN(200),
                TransactionId(30): LSN(300),
            },
            dirty_pages={
                PageId(1): LSN(50),
                PageId(5): LSN(150),
            },
        )

        data = record.to_bytes()
        restored = LogRecord.from_bytes(data)

        assert isinstance(restored, CheckpointRecord)
        assert restored.active_txns == record.active_txns
        assert restored.dirty_pages == record.dirty_pages


class TestLogRecordFactory:
    """Tests for LogRecord.from_bytes factory method."""

    def test_dispatches_to_correct_subclass(self) -> None:
        """from_bytes creates the correct subclass."""
        records = [
            BeginRecord(lsn=LSN(1), txn_id=TransactionId(1), prev_lsn=LSN(0)),
            CommitRecord(lsn=LSN(2), txn_id=TransactionId(1), prev_lsn=LSN(1)),
            AbortRecord(lsn=LSN(3), txn_id=TransactionId(2), prev_lsn=LSN(0)),
            UpdateRecord(
                lsn=LSN(4), txn_id=TransactionId(1), prev_lsn=LSN(2),
                page_id=PageId(1), slot_id=0, before_image=b"a", after_image=b"b"
            ),
            InsertRecord(
                lsn=LSN(5), txn_id=TransactionId(1), prev_lsn=LSN(4),
                page_id=PageId(1), slot_id=1, data=b"data"
            ),
            DeleteRecord(
                lsn=LSN(6), txn_id=TransactionId(1), prev_lsn=LSN(5),
                page_id=PageId(1), slot_id=0, data=b"deleted"
            ),
            CLRRecord(
                lsn=LSN(7), txn_id=TransactionId(2), prev_lsn=LSN(3),
                undo_next_lsn=LSN(0), page_id=PageId(1), slot_id=0
            ),
            CheckpointRecord(
                lsn=LSN(8), txn_id=TransactionId(0), prev_lsn=LSN(0)
            ),
        ]

        for record in records:
            data = record.to_bytes()
            restored = LogRecord.from_bytes(data)
            assert type(restored) == type(record)

    def test_too_short_data_raises(self) -> None:
        """from_bytes raises on insufficient data."""
        with pytest.raises(ValueError, match="at least"):
            LogRecord.from_bytes(b"short")

    def test_unknown_type_raises(self) -> None:
        """from_bytes raises on unknown record type."""
        # Create a record with invalid type byte
        import struct
        # type(1) + lsn(8) + txn_id(8) + prev_lsn(8) = 25 bytes
        invalid_data = struct.pack(">BQQQ", 255, 1, 1, 0)  # Type 255 doesn't exist

        with pytest.raises(ValueError, match="Unknown record type"):
            LogRecord.from_bytes(invalid_data)


class TestLogRecordCommonHeader:
    """Tests for common header properties."""

    def test_common_header_size(self) -> None:
        """Common header is 25 bytes."""
        assert LogRecord.COMMON_HEADER_SIZE == 25

    def test_prev_lsn_chains_transactions(self) -> None:
        """prev_lsn creates a chain for each transaction."""
        # Simulate a transaction with multiple operations
        txn = TransactionId(42)

        begin = BeginRecord(lsn=LSN(100), txn_id=txn, prev_lsn=LSN(0))
        insert1 = InsertRecord(
            lsn=LSN(101), txn_id=txn, prev_lsn=LSN(100),
            page_id=PageId(1), slot_id=0, data=b"a"
        )
        insert2 = InsertRecord(
            lsn=LSN(102), txn_id=txn, prev_lsn=LSN(101),
            page_id=PageId(1), slot_id=1, data=b"b"
        )
        commit = CommitRecord(lsn=LSN(103), txn_id=txn, prev_lsn=LSN(102))

        # Verify chain
        assert begin.prev_lsn == LSN(0)  # First record has no prev
        assert insert1.prev_lsn == begin.lsn
        assert insert2.prev_lsn == insert1.lsn
        assert commit.prev_lsn == insert2.lsn
