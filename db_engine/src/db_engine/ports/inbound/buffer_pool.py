"""Buffer Pool port for in-memory page caching.

This inbound port defines the contract for the buffer pool manager.
The buffer pool caches disk pages in memory to reduce I/O.

Key concepts:
- Pages are "pinned" while in use (cannot be evicted)
- Dirty pages are tracked for write-back
- LRU eviction when pool is full

References:
    - design.md Section 3 (Buffer Pool)
    - Effelsberg & Haerder, "Principles of Database Buffer Management" (1984)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from db_engine.domain.entities import SlottedPage
from db_engine.domain.value_objects import LSN, PageId


@dataclass
class BufferPoolStats:
    """Statistics for buffer pool monitoring."""

    pool_size: int  # Total number of frames
    pages_in_use: int  # Currently pinned pages
    dirty_pages: int  # Pages modified but not flushed
    hit_count: int  # Cache hits
    miss_count: int  # Cache misses
    eviction_count: int  # Pages evicted

    @property
    def hit_ratio(self) -> float:
        """Calculate the hit ratio (target: >0.99 in production)."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class BufferPool(Protocol):
    """Protocol for buffer pool operations.

    The buffer pool manages a fixed number of page frames in memory.
    Pages must be pinned before access and unpinned when done.

    Thread Safety:
        All methods must be thread-safe. Implementations typically
        use fine-grained locking (per-page latches).

    Example:
        page = buffer_pool.fetch_page(page_id)
        try:
            # Read or modify page
            page.insert_record(data)
        finally:
            buffer_pool.unpin_page(page_id, is_dirty=True)
    """

    @property
    @abstractmethod
    def pool_size(self) -> int:
        """Return the maximum number of page frames."""
        ...

    @abstractmethod
    def fetch_page(self, page_id: PageId) -> SlottedPage:
        """Fetch a page from the buffer pool.

        If the page is not in memory, it is read from disk.
        The page is pinned (pin count incremented) before return.

        Args:
            page_id: The page to fetch.

        Returns:
            The requested page (pinned).

        Raises:
            ValueError: If page_id is invalid.
            IOError: If disk read fails.
            BufferPoolFullError: If no frames available for eviction.
        """
        ...

    @abstractmethod
    def new_page(self) -> SlottedPage:
        """Allocate and return a new page.

        The page is pinned before return. The caller must unpin
        when done.

        Returns:
            A new, empty page (pinned).

        Raises:
            IOError: If disk allocation fails.
            BufferPoolFullError: If no frames available.
        """
        ...

    @abstractmethod
    def unpin_page(self, page_id: PageId, is_dirty: bool) -> bool:
        """Unpin a page, allowing it to be evicted.

        Args:
            page_id: The page to unpin.
            is_dirty: True if the page was modified.

        Returns:
            True if successful, False if page was not found.

        Raises:
            ValueError: If page is not pinned.
        """
        ...

    @abstractmethod
    def flush_page(self, page_id: PageId) -> bool:
        """Write a dirty page to disk.

        The page remains in the buffer pool but is no longer dirty.

        Args:
            page_id: The page to flush.

        Returns:
            True if flushed, False if page not found or not dirty.

        Raises:
            IOError: If disk write fails.
        """
        ...

    @abstractmethod
    def flush_all_pages(self) -> None:
        """Write all dirty pages to disk.

        Used during checkpoint and shutdown.

        Raises:
            IOError: If any disk write fails.
        """
        ...

    @abstractmethod
    def delete_page(self, page_id: PageId) -> bool:
        """Remove a page from the buffer pool and disk.

        The page must not be pinned.

        Args:
            page_id: The page to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If page is currently pinned.
        """
        ...

    @abstractmethod
    def get_page_lsn(self, page_id: PageId) -> LSN:
        """Get the LSN of a page (for WAL recovery).

        Args:
            page_id: The page to query.

        Returns:
            The page's LSN.

        Raises:
            ValueError: If page not in buffer pool.
        """
        ...

    @abstractmethod
    def set_page_lsn(self, page_id: PageId, lsn: LSN) -> None:
        """Set the LSN of a page (after WAL write).

        Args:
            page_id: The page to update.
            lsn: The new LSN.

        Raises:
            ValueError: If page not in buffer pool.
        """
        ...

    @abstractmethod
    def get_stats(self) -> BufferPoolStats:
        """Return buffer pool statistics for monitoring."""
        ...


class BufferPoolFullError(Exception):
    """Raised when buffer pool cannot evict any pages.

    This occurs when all pages are pinned and no frames are available.
    """

    pass
