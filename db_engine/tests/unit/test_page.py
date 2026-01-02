"""Unit tests for SlottedPage entity."""

from __future__ import annotations

import pytest

from db_engine.domain.entities import PageHeader, Slot, SlottedPage
from db_engine.domain.value_objects import LSN, INVALID_LSN, PageId


class TestPageHeader:
    """Tests for PageHeader serialization."""

    def test_header_size(self) -> None:
        """Header is exactly 24 bytes."""
        assert PageHeader.HEADER_SIZE == 24

    def test_serialization_roundtrip(self) -> None:
        """PageHeader serializes and deserializes correctly."""
        header = PageHeader(
            page_id=PageId(42),
            lsn=LSN(12345),
            checksum=0xDEADBEEF,
            slot_count=10,
            free_space_ptr=3000,
        )

        data = header.to_bytes()
        assert len(data) == PageHeader.HEADER_SIZE

        restored = PageHeader.from_bytes(data)
        assert restored.page_id == header.page_id
        assert restored.lsn == header.lsn
        assert restored.checksum == header.checksum
        assert restored.slot_count == header.slot_count
        assert restored.free_space_ptr == header.free_space_ptr


class TestSlot:
    """Tests for Slot structure."""

    def test_slot_size(self) -> None:
        """Slot is exactly 4 bytes."""
        assert Slot.SLOT_SIZE == 4

    def test_serialization_roundtrip(self) -> None:
        """Slot serializes and deserializes correctly."""
        slot = Slot(offset=1000, length=250)

        data = slot.to_bytes()
        assert len(data) == Slot.SLOT_SIZE

        restored = Slot.from_bytes(data)
        assert restored.offset == slot.offset
        assert restored.length == slot.length

    def test_empty_slot(self) -> None:
        """Empty slots have offset 0."""
        slot = Slot.empty()
        assert slot.is_empty()
        assert slot.offset == 0
        assert slot.length == 0

    def test_non_empty_slot(self) -> None:
        """Non-empty slots have offset > 0."""
        slot = Slot(offset=100, length=50)
        assert not slot.is_empty()


class TestSlottedPage:
    """Tests for SlottedPage."""

    def test_new_page(self) -> None:
        """New page is properly initialized."""
        page = SlottedPage.new(PageId(1))

        assert page.page_id == PageId(1)
        assert page.lsn == INVALID_LSN
        assert page.slot_count == 0
        assert page.record_count == 0
        # Free space is page size minus header
        assert page.get_free_space() == 4096 - PageHeader.HEADER_SIZE

    def test_insert_record(self) -> None:
        """Records can be inserted into a page."""
        page = SlottedPage.new(PageId(1))

        slot_id = page.insert_record(b"Hello, World!")

        assert slot_id == 0
        assert page.slot_count == 1
        assert page.record_count == 1
        assert page.get_record(slot_id) == b"Hello, World!"

    def test_insert_multiple_records(self) -> None:
        """Multiple records can be inserted."""
        page = SlottedPage.new(PageId(1))

        id1 = page.insert_record(b"First")
        id2 = page.insert_record(b"Second")
        id3 = page.insert_record(b"Third")

        assert id1 == 0
        assert id2 == 1
        assert id3 == 2
        assert page.record_count == 3

        assert page.get_record(id1) == b"First"
        assert page.get_record(id2) == b"Second"
        assert page.get_record(id3) == b"Third"

    def test_get_nonexistent_record(self) -> None:
        """Getting a nonexistent slot returns None."""
        page = SlottedPage.new(PageId(1))

        assert page.get_record(0) is None
        assert page.get_record(999) is None
        assert page.get_record(-1) is None

    def test_delete_record(self) -> None:
        """Records can be deleted."""
        page = SlottedPage.new(PageId(1))

        slot_id = page.insert_record(b"To be deleted")
        assert page.get_record(slot_id) == b"To be deleted"

        result = page.delete_record(slot_id)
        assert result is True
        assert page.get_record(slot_id) is None
        assert page.record_count == 0
        assert page.slot_count == 1  # Slot still exists as tombstone

    def test_delete_nonexistent(self) -> None:
        """Deleting nonexistent slot returns False."""
        page = SlottedPage.new(PageId(1))

        assert page.delete_record(0) is False
        assert page.delete_record(-1) is False

    def test_delete_already_deleted(self) -> None:
        """Deleting already-deleted slot returns False."""
        page = SlottedPage.new(PageId(1))
        slot_id = page.insert_record(b"data")
        page.delete_record(slot_id)

        assert page.delete_record(slot_id) is False

    def test_slot_reuse_after_delete(self) -> None:
        """Deleted slots are reused for new inserts."""
        page = SlottedPage.new(PageId(1))

        id1 = page.insert_record(b"First")
        page.delete_record(id1)

        id2 = page.insert_record(b"Second")
        assert id2 == id1  # Reused the slot

    def test_update_record_in_place(self) -> None:
        """Records can be updated in place if same size or smaller."""
        page = SlottedPage.new(PageId(1))

        slot_id = page.insert_record(b"Original value")
        result = page.update_record(slot_id, b"Updated value!")  # Same length

        assert result is True
        assert page.get_record(slot_id) == b"Updated value!"

    def test_update_record_smaller(self) -> None:
        """Smaller records update in place."""
        page = SlottedPage.new(PageId(1))

        slot_id = page.insert_record(b"Long original value")
        result = page.update_record(slot_id, b"Short")

        assert result is True
        assert page.get_record(slot_id) == b"Short"

    def test_update_record_larger(self) -> None:
        """Larger records allocate new space."""
        page = SlottedPage.new(PageId(1))

        slot_id = page.insert_record(b"Short")
        original_free = page.get_free_space()

        result = page.update_record(slot_id, b"Much longer value")

        assert result is True
        assert page.get_record(slot_id) == b"Much longer value"
        # Free space decreased (new allocation, old space is fragmented)
        assert page.get_free_space() < original_free

    def test_update_nonexistent(self) -> None:
        """Updating nonexistent slot returns False."""
        page = SlottedPage.new(PageId(1))

        assert page.update_record(0, b"data") is False

    def test_can_fit(self) -> None:
        """can_fit correctly determines if record will fit."""
        page = SlottedPage.new(PageId(1))

        # Should fit small records
        assert page.can_fit(100) is True
        assert page.can_fit(1000) is True

        # Should not fit records larger than page
        assert page.can_fit(10000) is False

    def test_insert_fails_when_full(self) -> None:
        """Insert fails when page is full."""
        page = SlottedPage.new(PageId(1), page_size=256)  # Small page for testing

        # Fill the page
        while page.can_fit(10):
            page.insert_record(b"0123456789")

        # Now it should fail
        with pytest.raises(ValueError, match="doesn't fit"):
            page.insert_record(b"0123456789")

    def test_compact(self) -> None:
        """Compact reclaims fragmented space."""
        page = SlottedPage.new(PageId(1))

        # Insert and delete to create fragmentation
        ids = [page.insert_record(f"Record {i}".encode()) for i in range(5)]
        page.delete_record(ids[1])
        page.delete_record(ids[3])

        free_before = page.get_free_space()
        page.compact()
        free_after = page.get_free_space()

        # Should have more free space after compaction
        assert free_after > free_before

        # Remaining records should still be accessible
        assert page.get_record(ids[0]) == b"Record 0"
        assert page.get_record(ids[2]) == b"Record 2"
        assert page.get_record(ids[4]) == b"Record 4"

    def test_checksum(self) -> None:
        """Checksum detects modifications."""
        page = SlottedPage.new(PageId(1))
        page.insert_record(b"Test data")
        page.update_checksum()

        assert page.verify_checksum() is True

        # Serialize and verify
        data = page.serialize()
        restored = SlottedPage.deserialize(data)

        assert restored.verify_checksum() is True

    def test_checksum_detects_corruption(self) -> None:
        """Checksum detects data corruption."""
        page = SlottedPage.new(PageId(1))
        page.insert_record(b"Test data")
        data = page.serialize()

        # Corrupt a byte in the middle
        corrupted = bytearray(data)
        corrupted[2000] ^= 0xFF
        corrupted_page = SlottedPage.deserialize(bytes(corrupted))

        assert corrupted_page.verify_checksum() is False

    def test_serialization_roundtrip(self) -> None:
        """Page can be serialized and deserialized."""
        page = SlottedPage.new(PageId(42))
        page.insert_record(b"First record")
        page.insert_record(b"Second record")
        page.lsn = LSN(12345)

        data = page.serialize()
        assert len(data) == 4096

        restored = SlottedPage.deserialize(data)

        assert restored.page_id == page.page_id
        assert restored.lsn == page.lsn
        assert restored.slot_count == page.slot_count
        assert restored.get_record(0) == b"First record"
        assert restored.get_record(1) == b"Second record"

    def test_lsn_property(self) -> None:
        """LSN can be get and set."""
        page = SlottedPage.new(PageId(1))

        assert page.lsn == INVALID_LSN

        page.lsn = LSN(999)
        assert page.lsn == LSN(999)

    def test_repr(self) -> None:
        """Page has informative repr."""
        page = SlottedPage.new(PageId(42))
        page.insert_record(b"test")

        repr_str = repr(page)
        assert "42" in repr_str
        assert "slots" in repr_str
        assert "records" in repr_str

    def test_custom_page_size(self) -> None:
        """Pages can have custom sizes."""
        page = SlottedPage.new(PageId(1), page_size=8192)

        data = page.serialize()
        assert len(data) == 8192
        # More free space with larger page
        assert page.get_free_space() > 4096 - PageHeader.HEADER_SIZE
