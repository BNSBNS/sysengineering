"""Index Manager port for B+Tree index operations.

This inbound port defines the contract for index management,
providing efficient key-based lookup and range scan operations.

Key responsibilities:
- Create and manage B+Tree indexes
- Provide point lookup and range scan operations
- Support concurrent access with latching

References:
    - design.md Section 3 (Index Manager)
    - Bayer & McCreight, "B+Trees" (1972)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Protocol

from db_engine.domain.value_objects import PageId, RecordId


class KeyType(Enum):
    """Supported key types for indexes."""

    INTEGER = "integer"
    VARCHAR = "varchar"
    TIMESTAMP = "timestamp"
    BYTES = "bytes"


@dataclass(frozen=True)
class IndexKey:
    """A key value in an index.

    Keys are comparable and hashable for use in B+Tree nodes.
    """

    value: int | str | bytes
    key_type: KeyType

    def __lt__(self, other: IndexKey) -> bool:
        if self.key_type != other.key_type:
            raise TypeError(f"Cannot compare {self.key_type} with {other.key_type}")
        return self.value < other.value  # type: ignore

    def __le__(self, other: IndexKey) -> bool:
        return self == other or self < other


@dataclass
class IndexMetadata:
    """Metadata for an index."""

    name: str
    table_name: str
    column_name: str
    key_type: KeyType
    root_page_id: PageId
    is_unique: bool
    height: int
    num_entries: int


@dataclass
class IndexStats:
    """Statistics for index monitoring."""

    num_indexes: int
    total_entries: int
    avg_height: float
    search_count: int
    insert_count: int
    delete_count: int


class Index(Protocol):
    """Protocol for a single index instance.

    Each index is a B+Tree that maps keys to RecordIds.
    The index maintains sorted order for efficient range scans.
    """

    @property
    @abstractmethod
    def metadata(self) -> IndexMetadata:
        """Return index metadata."""
        ...

    @abstractmethod
    def search(self, key: IndexKey) -> RecordId | None:
        """Search for a key and return its RecordId.

        Args:
            key: The key to search for.

        Returns:
            The RecordId if found, None otherwise.
        """
        ...

    @abstractmethod
    def insert(self, key: IndexKey, rid: RecordId) -> bool:
        """Insert a key-value pair into the index.

        For unique indexes, this fails if the key already exists.

        Args:
            key: The key to insert.
            rid: The RecordId to associate with the key.

        Returns:
            True if inserted, False if key already exists (unique index).

        Raises:
            IOError: If the insert fails due to I/O error.
        """
        ...

    @abstractmethod
    def delete(self, key: IndexKey) -> bool:
        """Delete a key from the index.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if key not found.
        """
        ...

    @abstractmethod
    def update(self, old_key: IndexKey, new_key: IndexKey, rid: RecordId) -> bool:
        """Update a key in the index.

        Equivalent to delete(old_key) + insert(new_key, rid) but
        may be optimized for in-place updates.

        Args:
            old_key: The current key value.
            new_key: The new key value.
            rid: The RecordId.

        Returns:
            True if updated, False if old_key not found.
        """
        ...

    @abstractmethod
    def range_scan(
        self,
        low: IndexKey | None = None,
        high: IndexKey | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> Iterator[tuple[IndexKey, RecordId]]:
        """Scan a range of keys.

        Args:
            low: Lower bound (None for unbounded).
            high: Upper bound (None for unbounded).
            include_low: Include the low bound in results.
            include_high: Include the high bound in results.

        Yields:
            (key, rid) tuples in sorted order.
        """
        ...

    @abstractmethod
    def scan_all(self) -> Iterator[tuple[IndexKey, RecordId]]:
        """Scan all entries in the index.

        Yields:
            (key, rid) tuples in sorted order.
        """
        ...


class IndexManager(Protocol):
    """Protocol for managing indexes.

    The index manager creates, drops, and provides access to indexes.
    It coordinates with the buffer pool for page access.

    Thread Safety:
        All methods must be thread-safe. B+Tree operations use
        crabbing (lock-coupling) for concurrent access.
    """

    @abstractmethod
    def create_index(
        self,
        name: str,
        table_name: str,
        column_name: str,
        key_type: KeyType,
        is_unique: bool = False,
    ) -> Index:
        """Create a new B+Tree index.

        Args:
            name: The index name.
            table_name: The table being indexed.
            column_name: The column being indexed.
            key_type: The type of the indexed column.
            is_unique: Whether the index enforces uniqueness.

        Returns:
            The created index.

        Raises:
            ValueError: If index name already exists.
        """
        ...

    @abstractmethod
    def get_index(self, name: str) -> Index | None:
        """Get an index by name.

        Args:
            name: The index name.

        Returns:
            The index if found, None otherwise.
        """
        ...

    @abstractmethod
    def drop_index(self, name: str) -> bool:
        """Drop an index.

        Args:
            name: The index name.

        Returns:
            True if dropped, False if not found.
        """
        ...

    @abstractmethod
    def list_indexes(self, table_name: str | None = None) -> list[IndexMetadata]:
        """List all indexes, optionally filtered by table.

        Args:
            table_name: If provided, only return indexes for this table.

        Returns:
            List of index metadata.
        """
        ...

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Return index manager statistics for monitoring."""
        ...
