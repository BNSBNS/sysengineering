"""Disk Manager port for low-level page I/O.

This outbound port defines the contract for disk-based page storage.
Implementations may use O_DIRECT, mmap, or standard file I/O.

The disk manager is responsible for:
- Reading and writing fixed-size pages to disk
- Allocating new pages
- Managing disk space

References:
    - design.md Section 2 (Architecture Overview - Disk Layer)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from db_engine.domain.value_objects import PageId


class DiskManager(Protocol):
    """Protocol for low-level disk I/O operations.

    The disk manager provides the lowest-level abstraction over
    the file system, dealing with fixed-size pages. It has no
    knowledge of page contents or buffer pool caching.

    Thread Safety:
        Implementations must be thread-safe for concurrent reads.
        Writes to the same page must be serialized externally.
    """

    @property
    @abstractmethod
    def page_size(self) -> int:
        """Return the fixed page size in bytes.

        Typically 4096 (4KB) to match OS page size and disk sector size.
        """
        ...

    @abstractmethod
    def read_page(self, page_id: PageId) -> bytes:
        """Read a page from disk.

        Args:
            page_id: The page to read.

        Returns:
            Raw page data as bytes (exactly page_size bytes).

        Raises:
            ValueError: If page_id is invalid.
            IOError: If the read fails.
        """
        ...

    @abstractmethod
    def write_page(self, page_id: PageId, data: bytes) -> None:
        """Write a page to disk.

        This operation should be atomic - either the entire page
        is written or none of it is. Use appropriate OS primitives
        (e.g., pwrite with O_DIRECT) to ensure atomicity.

        Args:
            page_id: The page to write.
            data: Raw page data (must be exactly page_size bytes).

        Raises:
            ValueError: If page_id is invalid or data size is wrong.
            IOError: If the write fails.
        """
        ...

    @abstractmethod
    def allocate_page(self) -> PageId:
        """Allocate a new page and return its ID.

        The page contents are undefined until written.

        Returns:
            The PageId of the newly allocated page.

        Raises:
            IOError: If allocation fails (e.g., disk full).
        """
        ...

    @abstractmethod
    def deallocate_page(self, page_id: PageId) -> None:
        """Mark a page as free for reuse.

        The page contents become undefined after deallocation.
        This operation may not immediately free disk space.

        Args:
            page_id: The page to deallocate.

        Raises:
            ValueError: If page_id is invalid or already free.
        """
        ...

    @abstractmethod
    def get_num_pages(self) -> int:
        """Return the total number of allocated pages.

        Returns:
            The count of pages currently allocated.
        """
        ...

    @abstractmethod
    def sync(self) -> None:
        """Ensure all writes are persisted to stable storage.

        This calls fsync() or equivalent to flush OS buffers.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the disk manager and release resources.

        After calling close(), the disk manager should not be used.
        """
        ...
