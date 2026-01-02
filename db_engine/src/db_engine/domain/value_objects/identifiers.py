"""Core identifiers and type-safe primitives for the database engine.

These value objects provide type-safe identifiers that are used throughout
the system to ensure correctness and prevent accidental misuse of raw integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType


# Type-safe identifiers using NewType for zero-cost runtime abstraction
# These provide compile-time type checking without runtime overhead

PageId = NewType("PageId", int)
"""Unique identifier for a database page. Pages are fixed-size blocks (default 4KB)."""

TransactionId = NewType("TransactionId", int)
"""Unique identifier for a transaction. Monotonically increasing."""

LSN = NewType("LSN", int)
"""Log Sequence Number - unique identifier for WAL records. Monotonically increasing."""

# Special sentinel values
INVALID_PAGE_ID = PageId(-1)
INVALID_TXN_ID = TransactionId(0)
INVALID_LSN = LSN(0)


@dataclass(frozen=True, slots=True)
class RecordId:
    """Record Identifier (RID) - uniquely identifies a record within the database.

    A RID is a combination of page_id and slot_id, allowing direct access
    to any record without needing to scan. Used by indexes to point to
    actual data records.

    Attributes:
        page_id: The page containing this record
        slot_id: The slot within the page's slot array

    Example:
        >>> rid = RecordId(PageId(42), 3)
        >>> rid.page_id
        PageId(42)
        >>> rid.slot_id
        3
    """

    page_id: PageId
    slot_id: int

    def __post_init__(self) -> None:
        """Validate the record identifier."""
        if self.slot_id < 0:
            raise ValueError(f"slot_id must be non-negative, got {self.slot_id}")

    def __repr__(self) -> str:
        return f"RID({self.page_id}:{self.slot_id})"

    def __str__(self) -> str:
        return f"({self.page_id}, {self.slot_id})"

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage.

        Format: 4 bytes page_id (big-endian) + 2 bytes slot_id (big-endian)
        """
        return (
            self.page_id.to_bytes(4, byteorder="big", signed=True) +
            self.slot_id.to_bytes(2, byteorder="big", signed=False)
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> RecordId:
        """Deserialize from bytes.

        Args:
            data: 6 bytes representing the RID

        Returns:
            RecordId instance

        Raises:
            ValueError: If data is not exactly 6 bytes
        """
        if len(data) != 6:
            raise ValueError(f"RecordId requires 6 bytes, got {len(data)}")

        page_id = PageId(int.from_bytes(data[0:4], byteorder="big", signed=True))
        slot_id = int.from_bytes(data[4:6], byteorder="big", signed=False)

        return cls(page_id=page_id, slot_id=slot_id)


# Size constants for serialization
RECORD_ID_SIZE = 6  # 4 bytes page_id + 2 bytes slot_id
