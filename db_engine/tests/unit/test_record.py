"""Unit tests for Record and MVCC entities."""

from __future__ import annotations

import pytest

from db_engine.domain.entities import Record, RecordHeader, Snapshot
from db_engine.domain.value_objects import LSN, INVALID_LSN, INVALID_TXN_ID, TransactionId


class TestRecordHeader:
    """Tests for RecordHeader with MVCC metadata."""

    def test_header_size(self) -> None:
        """Header is exactly 32 bytes."""
        assert RecordHeader.HEADER_SIZE == 32

    def test_new_header(self) -> None:
        """New header has correct initial values."""
        header = RecordHeader.new(
            created_by=TransactionId(100),
            created_at_lsn=LSN(500),
        )

        assert header.created_by == TransactionId(100)
        assert header.deleted_by == INVALID_TXN_ID
        assert header.created_at_lsn == LSN(500)
        assert header.deleted_at_lsn == INVALID_LSN
        assert not header.is_deleted()

    def test_mark_deleted(self) -> None:
        """Header can be marked as deleted."""
        header = RecordHeader.new(
            created_by=TransactionId(100),
            created_at_lsn=LSN(500),
        )

        deleted_header = header.mark_deleted(
            deleted_by=TransactionId(200),
            deleted_at_lsn=LSN(600),
        )

        # Original is unchanged (immutable pattern)
        assert not header.is_deleted()

        # New header is deleted
        assert deleted_header.is_deleted()
        assert deleted_header.deleted_by == TransactionId(200)
        assert deleted_header.deleted_at_lsn == LSN(600)
        # Created info preserved
        assert deleted_header.created_by == TransactionId(100)

    def test_serialization_roundtrip(self) -> None:
        """Header serializes and deserializes correctly."""
        header = RecordHeader(
            created_by=TransactionId(100),
            deleted_by=TransactionId(200),
            created_at_lsn=LSN(500),
            deleted_at_lsn=LSN(600),
        )

        data = header.to_bytes()
        assert len(data) == RecordHeader.HEADER_SIZE

        restored = RecordHeader.from_bytes(data)
        assert restored.created_by == header.created_by
        assert restored.deleted_by == header.deleted_by
        assert restored.created_at_lsn == header.created_at_lsn
        assert restored.deleted_at_lsn == header.deleted_at_lsn


class TestRecord:
    """Tests for Record entity."""

    def test_new_record(self) -> None:
        """New record has correct values."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"Hello, World!",
        )

        assert record.header.created_by == TransactionId(10)
        assert record.header.created_at_lsn == LSN(100)
        assert not record.header.is_deleted()
        assert record.data == b"Hello, World!"

    def test_record_length(self) -> None:
        """Record length includes header."""
        record = Record.new(
            created_by=TransactionId(1),
            created_at_lsn=LSN(1),
            data=b"12345",  # 5 bytes
        )

        assert len(record) == RecordHeader.HEADER_SIZE + 5
        assert len(record) == 37

    def test_mark_deleted(self) -> None:
        """Record can be marked as deleted."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"Original data",
        )

        deleted_record = record.mark_deleted(
            deleted_by=TransactionId(20),
            deleted_at_lsn=LSN(200),
        )

        # Original unchanged
        assert not record.header.is_deleted()

        # New record is deleted
        assert deleted_record.header.is_deleted()
        assert deleted_record.data == record.data  # Data preserved

    def test_serialization_roundtrip(self) -> None:
        """Record serializes and deserializes correctly."""
        record = Record.new(
            created_by=TransactionId(42),
            created_at_lsn=LSN(999),
            data=b"Some binary \x00\xff data",
        )

        serialized = record.to_bytes()
        restored = Record.from_bytes(serialized)

        assert restored.header.created_by == record.header.created_by
        assert restored.header.created_at_lsn == record.header.created_at_lsn
        assert restored.data == record.data


class TestSnapshot:
    """Tests for MVCC Snapshot."""

    def test_new_snapshot(self) -> None:
        """Snapshot can be created."""
        snapshot = Snapshot.new(
            txn_id=TransactionId(100),
            active_txns={TransactionId(50), TransactionId(75)},
        )

        assert snapshot.txn_id == TransactionId(100)
        assert TransactionId(50) in snapshot.active_txns
        assert TransactionId(75) in snapshot.active_txns

    def test_snapshot_immutable(self) -> None:
        """Snapshot is immutable (frozen dataclass)."""
        snapshot = Snapshot.new(TransactionId(1))

        with pytest.raises(AttributeError):
            snapshot.txn_id = TransactionId(99)  # type: ignore[misc]

    def test_empty_active_txns(self) -> None:
        """Snapshot can have no active transactions."""
        snapshot = Snapshot.new(TransactionId(100))

        assert len(snapshot.active_txns) == 0


class TestMVCCVisibility:
    """Tests for MVCC visibility rules.

    These tests verify the visibility rules from design.md Section 5:
    - Record visible if created by committed txn before snapshot
    - Record not visible if created by concurrent (active) txn
    - Record visible if not deleted
    - Record visible if deleted by txn that started after snapshot
    - Record visible if deleted by concurrent txn
    """

    def test_visible_committed_before_snapshot(self) -> None:
        """Record created by committed txn before snapshot is visible."""
        # Txn 10 created the record, committed before snapshot
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"data",
        )

        # Snapshot at txn 100, no active transactions
        snapshot = Snapshot.new(TransactionId(100))

        assert record.is_visible_to(snapshot)

    def test_not_visible_created_after_snapshot(self) -> None:
        """Record created after snapshot is not visible."""
        # Txn 200 created the record after snapshot
        record = Record.new(
            created_by=TransactionId(200),
            created_at_lsn=LSN(1000),
            data=b"data",
        )

        # Snapshot at txn 100
        snapshot = Snapshot.new(TransactionId(100))

        assert not record.is_visible_to(snapshot)

    def test_not_visible_created_by_concurrent(self) -> None:
        """Record created by concurrent (active) txn is not visible."""
        # Txn 50 created the record, but was active when snapshot taken
        record = Record.new(
            created_by=TransactionId(50),
            created_at_lsn=LSN(100),
            data=b"data",
        )

        # Snapshot at txn 100, with txn 50 still active
        snapshot = Snapshot.new(
            txn_id=TransactionId(100),
            active_txns={TransactionId(50)},
        )

        assert not record.is_visible_to(snapshot)

    def test_visible_not_deleted(self) -> None:
        """Non-deleted record is visible."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"data",
        )

        snapshot = Snapshot.new(TransactionId(100))

        assert not record.header.is_deleted()
        assert record.is_visible_to(snapshot)

    def test_not_visible_deleted_before_snapshot(self) -> None:
        """Record deleted by committed txn before snapshot is not visible."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"data",
        )
        deleted_record = record.mark_deleted(
            deleted_by=TransactionId(50),
            deleted_at_lsn=LSN(200),
        )

        # Snapshot at txn 100, txn 50 already committed (deleted before snapshot)
        snapshot = Snapshot.new(TransactionId(100))

        assert not deleted_record.is_visible_to(snapshot)

    def test_visible_deleted_after_snapshot(self) -> None:
        """Record deleted after snapshot started is still visible to that snapshot."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"data",
        )
        deleted_record = record.mark_deleted(
            deleted_by=TransactionId(200),  # Deleted by txn 200, after snapshot
            deleted_at_lsn=LSN(1000),
        )

        # Snapshot at txn 100
        snapshot = Snapshot.new(TransactionId(100))

        assert deleted_record.is_visible_to(snapshot)

    def test_visible_deleted_by_concurrent(self) -> None:
        """Record deleted by concurrent txn is still visible."""
        record = Record.new(
            created_by=TransactionId(10),
            created_at_lsn=LSN(100),
            data=b"data",
        )
        deleted_record = record.mark_deleted(
            deleted_by=TransactionId(50),  # Concurrent txn
            deleted_at_lsn=LSN(200),
        )

        # Snapshot at txn 100, with txn 50 still active
        snapshot = Snapshot.new(
            txn_id=TransactionId(100),
            active_txns={TransactionId(50)},
        )

        assert deleted_record.is_visible_to(snapshot)

    def test_visibility_edge_case_same_txn(self) -> None:
        """Record created by same txn as snapshot - edge case."""
        # When txn_id equals created_by and created_by is not in active_txns,
        # this means the transaction itself sees its own uncommitted changes
        record = Record.new(
            created_by=TransactionId(100),
            created_at_lsn=LSN(500),
            data=b"data",
        )

        snapshot = Snapshot.new(TransactionId(100))

        # created_by (100) > txn_id (100) is False
        # created_by (100) in active_txns is False (empty set)
        # So this record IS visible (created by current txn)
        # Actually, per the visibility rules, created_by > snapshot.txn_id is False
        # and created_by not in active_txns, so it passes first two checks
        # and is visible since not deleted
        assert record.is_visible_to(snapshot)
