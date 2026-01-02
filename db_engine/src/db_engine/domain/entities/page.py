"""Slotted page implementation for variable-length record storage.

This module implements the slotted page format as described in design.md Section 4.
Slotted pages support variable-length records while allowing record movement
during compaction without updating external references (indexes use slot IDs
which remain stable).

Page Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Page Header (24 bytes)                                       │
    │ ┌─────────┬─────────┬──────────┬─────────┬─────────────────┐│
    │ │ Page ID │   LSN   │ Checksum │ Slot Cnt│  Free Space Ptr ││
    │ │ (4B)    │  (8B)   │   (4B)   │  (2B)   │     (2B)        ││
    │ └─────────┴─────────┴──────────┴─────────┴─────────────────┘│
    ├─────────────────────────────────────────────────────────────┤
    │ Slot Array (grows downward)                                  │
    │ ┌────────┬────────┬────────┬────────┐                       │
    │ │ Slot 0 │ Slot 1 │ Slot 2 │  ...   │                       │
    │ │(offset,│(offset,│(offset,│        │                       │
    │ │ length)│ length)│ length)│        │                       │
    │ └────────┴────────┴────────┴────────┘                       │
    ├─────────────────────────────────────────────────────────────┤
    │                    Free Space                                │
    ├─────────────────────────────────────────────────────────────┤
    │ Records (grow upward from bottom)                            │
    │ ┌────────────────┬────────────────┬────────────────────────┐│
    │ │    Record 2    │    Record 1    │       Record 0         ││
    │ └────────────────┴────────────────┴────────────────────────┘│
    └─────────────────────────────────────────────────────────────┘

References:
    - design.md Section 4 (Data Models & State Machines)
    - PostgreSQL Documentation "Database Physical Storage"
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import ClassVar

from db_engine.domain.value_objects import LSN, INVALID_LSN, INVALID_PAGE_ID, PageId


@dataclass
class PageHeader:
    """24-byte page header containing metadata.

    The header is stored at the beginning of every page and contains:
    - page_id: Unique identifier for this page
    - lsn: Log Sequence Number of the last modification (for recovery)
    - checksum: CRC32 checksum for corruption detection
    - slot_count: Number of slots in the slot array
    - free_space_ptr: Offset to the start of free space

    The checksum covers the entire page except the checksum field itself.
    """

    page_id: PageId
    lsn: LSN
    checksum: int  # CRC32, computed over page content
    slot_count: int
    free_space_ptr: int
    # Reserved bytes for future use (maintains 24-byte alignment)
    _reserved: bytes = field(default=b"\x00\x00\x00\x00", repr=False)

    # Header layout: page_id(4) + lsn(8) + checksum(4) + slot_count(2) + free_space_ptr(2) + reserved(4) = 24
    HEADER_SIZE: ClassVar[int] = 24
    HEADER_FORMAT: ClassVar[str] = ">iQIHH4s"  # big-endian: int, uint64, uint32, uint16, uint16, 4 bytes

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self.HEADER_FORMAT,
            self.page_id,
            self.lsn,
            self.checksum,
            self.slot_count,
            self.free_space_ptr,
            self._reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> PageHeader:
        """Deserialize header from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Header requires {cls.HEADER_SIZE} bytes, got {len(data)}")

        page_id, lsn, checksum, slot_count, free_space_ptr, reserved = struct.unpack(
            cls.HEADER_FORMAT, data[: cls.HEADER_SIZE]
        )

        return cls(
            page_id=PageId(page_id),
            lsn=LSN(lsn),
            checksum=checksum,
            slot_count=slot_count,
            free_space_ptr=free_space_ptr,
            _reserved=reserved,
        )


@dataclass
class Slot:
    """A slot in the slot array, pointing to a record.

    Each slot is 4 bytes: 2 bytes for offset, 2 bytes for length.
    An offset of 0 indicates a deleted/empty slot (tombstone).
    """

    offset: int  # Offset from page start to record data
    length: int  # Length of record in bytes

    SLOT_SIZE: ClassVar[int] = 4
    SLOT_FORMAT: ClassVar[str] = ">HH"  # big-endian: 2 uint16

    def is_empty(self) -> bool:
        """Check if this slot is empty (deleted record)."""
        return self.offset == 0

    def to_bytes(self) -> bytes:
        """Serialize slot to bytes."""
        return struct.pack(self.SLOT_FORMAT, self.offset, self.length)

    @classmethod
    def from_bytes(cls, data: bytes) -> Slot:
        """Deserialize slot from bytes."""
        if len(data) < cls.SLOT_SIZE:
            raise ValueError(f"Slot requires {cls.SLOT_SIZE} bytes, got {len(data)}")

        offset, length = struct.unpack(cls.SLOT_FORMAT, data[: cls.SLOT_SIZE])
        return cls(offset=offset, length=length)

    @classmethod
    def empty(cls) -> Slot:
        """Create an empty (tombstone) slot."""
        return cls(offset=0, length=0)


class SlottedPage:
    """Variable-length record storage using slotted page format.

    The slotted page format allows:
    - Variable-length records
    - Record movement during compaction (slot IDs remain stable)
    - Efficient space utilization with free space coalescing

    Thread Safety:
        This class is NOT thread-safe. External synchronization is required
        when accessing from multiple threads (typically via buffer pool pins).

    Example:
        >>> page = SlottedPage.new(PageId(1))
        >>> slot_id = page.insert_record(b"Hello, World!")
        >>> data = page.get_record(slot_id)
        >>> print(data)
        b'Hello, World!'
    """

    PAGE_SIZE: ClassVar[int] = 4096  # Default 4KB pages

    def __init__(
        self,
        page_id: PageId,
        data: bytearray | None = None,
        *,
        page_size: int = 4096,
    ) -> None:
        """Initialize a slotted page.

        Args:
            page_id: Unique identifier for this page
            data: Raw page data (if loading from disk), or None for new page
            page_size: Size of page in bytes (default 4KB)
        """
        self._page_size = page_size

        if data is not None:
            # Loading existing page from disk
            if len(data) != page_size:
                raise ValueError(f"Page data must be {page_size} bytes, got {len(data)}")
            self._data = bytearray(data)
            self._header = PageHeader.from_bytes(bytes(self._data[: PageHeader.HEADER_SIZE]))
            self._load_slots()
        else:
            # Creating new empty page
            self._data = bytearray(page_size)
            self._header = PageHeader(
                page_id=page_id,
                lsn=INVALID_LSN,
                checksum=0,
                slot_count=0,
                free_space_ptr=page_size,  # Initially, free space starts at end
            )
            self._slots: list[Slot] = []
            self._write_header()

    def _load_slots(self) -> None:
        """Load slot array from page data."""
        self._slots = []
        slot_start = PageHeader.HEADER_SIZE

        for i in range(self._header.slot_count):
            offset = slot_start + i * Slot.SLOT_SIZE
            slot_data = bytes(self._data[offset : offset + Slot.SLOT_SIZE])
            self._slots.append(Slot.from_bytes(slot_data))

    def _write_header(self) -> None:
        """Write header to page data."""
        header_bytes = self._header.to_bytes()
        self._data[: PageHeader.HEADER_SIZE] = header_bytes

    def _write_slots(self) -> None:
        """Write slot array to page data."""
        slot_start = PageHeader.HEADER_SIZE
        for i, slot in enumerate(self._slots):
            offset = slot_start + i * Slot.SLOT_SIZE
            self._data[offset : offset + Slot.SLOT_SIZE] = slot.to_bytes()

    @property
    def page_id(self) -> PageId:
        """Get the page ID."""
        return self._header.page_id

    @property
    def lsn(self) -> LSN:
        """Get the last modification LSN."""
        return self._header.lsn

    @lsn.setter
    def lsn(self, value: LSN) -> None:
        """Set the last modification LSN."""
        self._header.lsn = value
        self._write_header()

    @property
    def slot_count(self) -> int:
        """Get the number of slots (including empty ones)."""
        return self._header.slot_count

    @property
    def record_count(self) -> int:
        """Get the number of non-empty records."""
        return sum(1 for slot in self._slots if not slot.is_empty())

    def get_free_space(self) -> int:
        """Get the amount of free space in bytes.

        Returns the contiguous free space between the slot array and records.
        Does not account for fragmented space from deleted records.
        """
        slot_array_end = PageHeader.HEADER_SIZE + len(self._slots) * Slot.SLOT_SIZE
        return self._header.free_space_ptr - slot_array_end

    def can_fit(self, record_size: int) -> bool:
        """Check if a record of given size can fit in this page.

        Args:
            record_size: Size of record data in bytes

        Returns:
            True if the record can fit (including slot overhead)
        """
        # Need space for: record data + new slot (if no empty slots available)
        needed = record_size
        if not any(slot.is_empty() for slot in self._slots):
            needed += Slot.SLOT_SIZE

        return self.get_free_space() >= needed

    def insert_record(self, record: bytes) -> int:
        """Insert a record into the page.

        Args:
            record: The record data to insert

        Returns:
            The slot ID where the record was inserted

        Raises:
            ValueError: If the record doesn't fit in the page
        """
        if not self.can_fit(len(record)):
            raise ValueError(
                f"Record of size {len(record)} doesn't fit in page "
                f"(free space: {self.get_free_space()})"
            )

        # Find an empty slot or create a new one
        slot_id = None
        for i, slot in enumerate(self._slots):
            if slot.is_empty():
                slot_id = i
                break

        if slot_id is None:
            # Create new slot
            slot_id = len(self._slots)
            self._slots.append(Slot.empty())
            self._header.slot_count += 1

        # Allocate space for record (grows from bottom)
        record_offset = self._header.free_space_ptr - len(record)
        self._header.free_space_ptr = record_offset

        # Write record data
        self._data[record_offset : record_offset + len(record)] = record

        # Update slot
        self._slots[slot_id] = Slot(offset=record_offset, length=len(record))

        # Persist changes
        self._write_header()
        self._write_slots()

        return slot_id

    def get_record(self, slot_id: int) -> bytes | None:
        """Get a record by slot ID.

        Args:
            slot_id: The slot ID of the record

        Returns:
            The record data, or None if the slot is empty or invalid
        """
        if slot_id < 0 or slot_id >= len(self._slots):
            return None

        slot = self._slots[slot_id]
        if slot.is_empty():
            return None

        return bytes(self._data[slot.offset : slot.offset + slot.length])

    def update_record(self, slot_id: int, record: bytes) -> bool:
        """Update a record in place.

        If the new record is the same size or smaller, it's updated in place.
        If larger, the old space is marked as deleted and new space is allocated.

        Args:
            slot_id: The slot ID of the record to update
            record: The new record data

        Returns:
            True if successful, False if slot is invalid or record doesn't fit
        """
        if slot_id < 0 or slot_id >= len(self._slots):
            return False

        slot = self._slots[slot_id]
        if slot.is_empty():
            return False

        if len(record) <= slot.length:
            # Fits in existing space - update in place
            self._data[slot.offset : slot.offset + len(record)] = record
            # Zero out remaining space (optional, for security)
            if len(record) < slot.length:
                self._data[slot.offset + len(record) : slot.offset + slot.length] = (
                    b"\x00" * (slot.length - len(record))
                )
            self._slots[slot_id] = Slot(offset=slot.offset, length=len(record))
            self._write_slots()
            return True
        else:
            # Need more space - delete old and insert new
            if not self.can_fit(len(record) - slot.length):  # Account for reclaimed slot
                return False

            # Mark old slot as empty (creates fragmentation)
            self._slots[slot_id] = Slot.empty()

            # Allocate new space
            record_offset = self._header.free_space_ptr - len(record)
            self._header.free_space_ptr = record_offset

            # Write new record
            self._data[record_offset : record_offset + len(record)] = record
            self._slots[slot_id] = Slot(offset=record_offset, length=len(record))

            self._write_header()
            self._write_slots()
            return True

    def delete_record(self, slot_id: int) -> bool:
        """Delete a record by slot ID.

        The slot is marked as empty (tombstone). The space is not immediately
        reclaimed - call compact() to reclaim fragmented space.

        Args:
            slot_id: The slot ID of the record to delete

        Returns:
            True if successful, False if slot is invalid or already empty
        """
        if slot_id < 0 or slot_id >= len(self._slots):
            return False

        slot = self._slots[slot_id]
        if slot.is_empty():
            return False

        # Mark slot as empty (tombstone)
        self._slots[slot_id] = Slot.empty()
        self._write_slots()

        return True

    def compact(self) -> None:
        """Compact the page to reclaim fragmented space.

        Moves all records to be contiguous at the end of the page,
        reclaiming space from deleted records. Slot IDs remain stable.
        """
        # Collect all non-empty records with their slot IDs
        records: list[tuple[int, bytes]] = []
        for slot_id, slot in enumerate(self._slots):
            if not slot.is_empty():
                record_data = bytes(self._data[slot.offset : slot.offset + slot.length])
                records.append((slot_id, record_data))

        # Reset free space pointer to end of page
        self._header.free_space_ptr = self._page_size

        # Rewrite records from the end
        for slot_id, record_data in records:
            record_offset = self._header.free_space_ptr - len(record_data)
            self._header.free_space_ptr = record_offset
            self._data[record_offset : record_offset + len(record_data)] = record_data
            self._slots[slot_id] = Slot(offset=record_offset, length=len(record_data))

        # Clear the old record area (optional, for security)
        slot_array_end = PageHeader.HEADER_SIZE + len(self._slots) * Slot.SLOT_SIZE
        self._data[slot_array_end : self._header.free_space_ptr] = b"\x00" * (
            self._header.free_space_ptr - slot_array_end
        )

        self._write_header()
        self._write_slots()

    def compute_checksum(self) -> int:
        """Compute CRC32 checksum of page content.

        The checksum covers the entire page except the checksum field itself.
        """
        # Temporarily zero out checksum field for computation
        checksum_offset = 4 + 8  # page_id(4) + lsn(8)
        original = self._data[checksum_offset : checksum_offset + 4]
        self._data[checksum_offset : checksum_offset + 4] = b"\x00\x00\x00\x00"

        checksum = zlib.crc32(bytes(self._data)) & 0xFFFFFFFF

        # Restore original checksum field
        self._data[checksum_offset : checksum_offset + 4] = original

        return checksum

    def verify_checksum(self) -> bool:
        """Verify the page checksum.

        Returns:
            True if checksum is valid, False if corrupted
        """
        return self.compute_checksum() == self._header.checksum

    def update_checksum(self) -> None:
        """Update the stored checksum to match current content."""
        self._header.checksum = self.compute_checksum()
        self._write_header()

    def serialize(self) -> bytes:
        """Serialize page to bytes for writing to disk.

        Updates checksum before serialization.
        """
        self.update_checksum()
        return bytes(self._data)

    @classmethod
    def deserialize(cls, data: bytes, *, page_size: int = 4096) -> SlottedPage:
        """Deserialize page from bytes.

        Args:
            data: Raw page bytes
            page_size: Expected page size

        Returns:
            SlottedPage instance

        Raises:
            ValueError: If data size doesn't match page_size
        """
        # Extract page_id from header for initialization
        header = PageHeader.from_bytes(data[: PageHeader.HEADER_SIZE])
        return cls(page_id=header.page_id, data=bytearray(data), page_size=page_size)

    @classmethod
    def new(cls, page_id: PageId, *, page_size: int = 4096) -> SlottedPage:
        """Create a new empty page.

        Args:
            page_id: Unique identifier for the page
            page_size: Page size in bytes

        Returns:
            New empty SlottedPage
        """
        return cls(page_id=page_id, page_size=page_size)

    def __repr__(self) -> str:
        return (
            f"SlottedPage(id={self._header.page_id}, "
            f"slots={self._header.slot_count}, "
            f"records={self.record_count}, "
            f"free={self.get_free_space()})"
        )
