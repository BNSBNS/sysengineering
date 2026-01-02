"""Unit tests for domain value objects - identifiers."""

from __future__ import annotations

import pytest

from db_engine.domain.value_objects import (
    INVALID_LSN,
    INVALID_PAGE_ID,
    INVALID_TXN_ID,
    LSN,
    PageId,
    RECORD_ID_SIZE,
    RecordId,
    TransactionId,
)


class TestPageId:
    """Tests for PageId type."""

    def test_creation(self) -> None:
        """PageId can be created from an integer."""
        page_id = PageId(42)
        assert page_id == 42

    def test_type_safety(self) -> None:
        """PageId is distinct from plain int for type checking."""
        page_id = PageId(1)
        # At runtime, PageId is just an int
        assert isinstance(page_id, int)

    def test_invalid_sentinel(self) -> None:
        """INVALID_PAGE_ID is -1."""
        assert INVALID_PAGE_ID == PageId(-1)
        assert INVALID_PAGE_ID == -1


class TestTransactionId:
    """Tests for TransactionId type."""

    def test_creation(self) -> None:
        """TransactionId can be created from an integer."""
        txn_id = TransactionId(100)
        assert txn_id == 100

    def test_invalid_sentinel(self) -> None:
        """INVALID_TXN_ID is 0."""
        assert INVALID_TXN_ID == TransactionId(0)
        assert INVALID_TXN_ID == 0


class TestLSN:
    """Tests for Log Sequence Number type."""

    def test_creation(self) -> None:
        """LSN can be created from an integer."""
        lsn = LSN(12345)
        assert lsn == 12345

    def test_invalid_sentinel(self) -> None:
        """INVALID_LSN is 0."""
        assert INVALID_LSN == LSN(0)
        assert INVALID_LSN == 0

    def test_comparison(self) -> None:
        """LSNs can be compared for ordering."""
        lsn1 = LSN(100)
        lsn2 = LSN(200)
        assert lsn1 < lsn2
        assert lsn2 > lsn1


class TestRecordId:
    """Tests for RecordId (RID) composite identifier."""

    def test_creation(self) -> None:
        """RecordId can be created with page_id and slot_id."""
        rid = RecordId(PageId(10), 5)
        assert rid.page_id == PageId(10)
        assert rid.slot_id == 5

    def test_immutability(self) -> None:
        """RecordId is immutable (frozen dataclass)."""
        rid = RecordId(PageId(1), 2)
        with pytest.raises(AttributeError):
            rid.page_id = PageId(99)  # type: ignore[misc]

    def test_slot_id_validation(self) -> None:
        """slot_id must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            RecordId(PageId(1), -1)

    def test_equality(self) -> None:
        """RecordIds with same values are equal."""
        rid1 = RecordId(PageId(1), 2)
        rid2 = RecordId(PageId(1), 2)
        rid3 = RecordId(PageId(1), 3)

        assert rid1 == rid2
        assert rid1 != rid3

    def test_hashability(self) -> None:
        """RecordId can be used as dict key."""
        rid1 = RecordId(PageId(1), 2)
        rid2 = RecordId(PageId(1), 2)

        d = {rid1: "value"}
        assert d[rid2] == "value"

    def test_repr(self) -> None:
        """RecordId has informative repr."""
        rid = RecordId(PageId(42), 7)
        assert "42" in repr(rid)
        assert "7" in repr(rid)

    def test_str(self) -> None:
        """RecordId has readable str representation."""
        rid = RecordId(PageId(10), 5)
        assert str(rid) == "(10, 5)"

    def test_serialization_roundtrip(self) -> None:
        """RecordId can be serialized and deserialized."""
        original = RecordId(PageId(12345), 67)
        serialized = original.to_bytes()

        assert len(serialized) == RECORD_ID_SIZE
        assert len(serialized) == 6

        deserialized = RecordId.from_bytes(serialized)
        assert deserialized == original

    def test_serialization_format(self) -> None:
        """Verify serialization uses big-endian format."""
        rid = RecordId(PageId(0x01020304), 0x0506)
        data = rid.to_bytes()

        # 4 bytes page_id (big-endian)
        assert data[0:4] == bytes([0x01, 0x02, 0x03, 0x04])
        # 2 bytes slot_id (big-endian)
        assert data[4:6] == bytes([0x05, 0x06])

    def test_deserialization_wrong_size(self) -> None:
        """Deserialization fails with wrong data size."""
        with pytest.raises(ValueError, match="6 bytes"):
            RecordId.from_bytes(b"short")

    def test_negative_page_id_serialization(self) -> None:
        """Negative page IDs serialize correctly."""
        rid = RecordId(PageId(-1), 0)  # INVALID_PAGE_ID
        data = rid.to_bytes()
        restored = RecordId.from_bytes(data)

        assert restored.page_id == PageId(-1)

    def test_max_slot_id(self) -> None:
        """Maximum slot_id (65535) works correctly."""
        rid = RecordId(PageId(1), 65535)
        data = rid.to_bytes()
        restored = RecordId.from_bytes(data)

        assert restored.slot_id == 65535


class TestRecordIdSize:
    """Tests for RECORD_ID_SIZE constant."""

    def test_constant_value(self) -> None:
        """RECORD_ID_SIZE is 6 bytes."""
        assert RECORD_ID_SIZE == 6
