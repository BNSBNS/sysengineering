"""LRU Buffer Pool implementation.

This adapter implements the BufferPool protocol using an LRU eviction
policy. Pages are cached in memory and evicted when the pool is full.

Key concepts:
- Frame: A slot in the buffer pool that can hold one page
- Pin: A reference count preventing eviction
- Dirty: A flag indicating the page has been modified

Thread Safety:
    All operations are thread-safe using fine-grained locking.
    Each frame has its own latch for concurrent access.

References:
    - design.md Section 3 (Buffer Pool)
    - Effelsberg & Haerder, "Principles of Database Buffer Management" (1984)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict

from db_engine.domain.entities import SlottedPage
from db_engine.domain.value_objects import LSN, PageId, INVALID_PAGE_ID
from db_engine.ports.inbound.buffer_pool import BufferPoolFullError, BufferPoolStats
from db_engine.ports.outbound.disk_manager import DiskManager


@dataclass
class Frame:
    """A frame in the buffer pool holding a page.

    Attributes:
        page: The cached page (or None if empty).
        page_id: The ID of the cached page.
        pin_count: Number of active references.
        is_dirty: True if the page has been modified.
        latch: Per-frame lock for thread safety.
    """

    page: SlottedPage | None = None
    page_id: PageId = INVALID_PAGE_ID
    pin_count: int = 0
    is_dirty: bool = False
    latch: threading.RLock = field(default_factory=threading.RLock)


class LRUBufferPool:
    """LRU-based buffer pool implementation.

    Manages a fixed number of page frames in memory. When a page is
    accessed, it moves to the front of the LRU list. When eviction is
    needed, the least recently used unpinned page is evicted.

    Attributes:
        pool_size: Maximum number of pages that can be cached.
        disk_manager: The underlying disk I/O manager.
    """

    def __init__(self, pool_size: int, disk_manager: DiskManager) -> None:
        """Initialize the buffer pool.

        Args:
            pool_size: Number of page frames in the pool.
            disk_manager: The disk manager for page I/O.

        Raises:
            ValueError: If pool_size < 1.
        """
        if pool_size < 1:
            raise ValueError(f"Pool size must be >= 1, got {pool_size}")

        self._pool_size = pool_size
        self._disk_manager = disk_manager
        self._page_size = disk_manager.page_size

        # Global lock for pool-level operations
        self._lock = threading.RLock()

        # Page table: maps page_id -> frame_index
        self._page_table: Dict[PageId, int] = {}

        # Frames array
        self._frames: list[Frame] = [Frame() for _ in range(pool_size)]

        # LRU list: OrderedDict maintains insertion order
        # Key = frame_index, Value = None (we only care about order)
        # Most recently used at the end, least recently at the front
        self._lru: OrderedDict[int, None] = OrderedDict()

        # Initialize all frames as free (in LRU list)
        for i in range(pool_size):
            self._lru[i] = None

        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

    @property
    def pool_size(self) -> int:
        """Return the maximum number of page frames."""
        return self._pool_size

    def _find_victim(self) -> int | None:
        """Find an unpinned frame to evict using LRU policy.

        Returns:
            Frame index if found, None if all frames are pinned.
        """
        # Scan LRU list from least recently used (front)
        for frame_idx in self._lru.keys():
            frame = self._frames[frame_idx]
            if frame.pin_count == 0:
                return frame_idx
        return None

    def _evict_frame(self, frame_idx: int) -> None:
        """Evict a frame, flushing if dirty.

        Args:
            frame_idx: The frame to evict.
        """
        frame = self._frames[frame_idx]

        if frame.page is None:
            return

        # Flush if dirty
        if frame.is_dirty:
            page_data = frame.page.serialize()
            self._disk_manager.write_page(frame.page_id, page_data)
            frame.is_dirty = False

        # Remove from page table
        if frame.page_id in self._page_table:
            del self._page_table[frame.page_id]

        # Reset frame
        frame.page = None
        frame.page_id = INVALID_PAGE_ID
        frame.pin_count = 0

        self._eviction_count += 1

    def fetch_page(self, page_id: PageId) -> SlottedPage:
        """Fetch a page from the buffer pool.

        If the page is not in memory, it is read from disk.
        The page is pinned before return.

        Args:
            page_id: The page to fetch.

        Returns:
            The requested page (pinned).

        Raises:
            ValueError: If page_id is invalid.
            IOError: If disk read fails.
            BufferPoolFullError: If no frames available for eviction.
        """
        with self._lock:
            # Check if page is already in pool
            if page_id in self._page_table:
                frame_idx = self._page_table[page_id]
                frame = self._frames[frame_idx]

                with frame.latch:
                    frame.pin_count += 1

                    # Move to end of LRU (most recently used)
                    if frame_idx in self._lru:
                        self._lru.move_to_end(frame_idx)

                    self._hit_count += 1
                    return frame.page  # type: ignore

            # Page not in pool - need to read from disk
            self._miss_count += 1

            # Find a victim frame
            victim_idx = self._find_victim()
            if victim_idx is None:
                raise BufferPoolFullError("All buffer pool frames are pinned")

            frame = self._frames[victim_idx]

            with frame.latch:
                # Evict current occupant
                self._evict_frame(victim_idx)

                # Read page from disk
                page_data = self._disk_manager.read_page(page_id)
                page = SlottedPage.deserialize(page_data)

                # Install page in frame
                frame.page = page
                frame.page_id = page_id
                frame.pin_count = 1
                frame.is_dirty = False

                # Update page table
                self._page_table[page_id] = victim_idx

                # Move to end of LRU
                self._lru.move_to_end(victim_idx)

                return page

    def new_page(self) -> SlottedPage:
        """Allocate and return a new page.

        The page is pinned before return.

        Returns:
            A new, empty page (pinned).

        Raises:
            IOError: If disk allocation fails.
            BufferPoolFullError: If no frames available.
        """
        with self._lock:
            # Allocate page on disk
            page_id = self._disk_manager.allocate_page()

            # Find a victim frame
            victim_idx = self._find_victim()
            if victim_idx is None:
                # Deallocate the page we just allocated
                self._disk_manager.deallocate_page(page_id)
                raise BufferPoolFullError("All buffer pool frames are pinned")

            frame = self._frames[victim_idx]

            with frame.latch:
                # Evict current occupant
                self._evict_frame(victim_idx)

                # Create new empty page
                page = SlottedPage.new(page_id, page_size=self._page_size)

                # Install page in frame
                frame.page = page
                frame.page_id = page_id
                frame.pin_count = 1
                frame.is_dirty = True  # New page is dirty

                # Update page table
                self._page_table[page_id] = victim_idx

                # Move to end of LRU
                self._lru.move_to_end(victim_idx)

                return page

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
        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_idx = self._page_table[page_id]
            frame = self._frames[frame_idx]

            with frame.latch:
                if frame.pin_count <= 0:
                    raise ValueError(f"Page {page_id} is not pinned")

                frame.pin_count -= 1
                if is_dirty:
                    frame.is_dirty = True

                return True

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
        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_idx = self._page_table[page_id]
            frame = self._frames[frame_idx]

            with frame.latch:
                if not frame.is_dirty or frame.page is None:
                    return False

                page_data = frame.page.serialize()
                self._disk_manager.write_page(page_id, page_data)
                frame.is_dirty = False

                return True

    def flush_all_pages(self) -> None:
        """Write all dirty pages to disk.

        Raises:
            IOError: If any disk write fails.
        """
        with self._lock:
            for frame in self._frames:
                with frame.latch:
                    if frame.is_dirty and frame.page is not None:
                        page_data = frame.page.serialize()
                        self._disk_manager.write_page(frame.page_id, page_data)
                        frame.is_dirty = False

            # Sync to ensure durability
            self._disk_manager.sync()

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
        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_idx = self._page_table[page_id]
            frame = self._frames[frame_idx]

            with frame.latch:
                if frame.pin_count > 0:
                    raise ValueError(f"Cannot delete pinned page {page_id}")

                # Remove from page table
                del self._page_table[page_id]

                # Reset frame
                frame.page = None
                frame.page_id = INVALID_PAGE_ID
                frame.is_dirty = False

                # Deallocate on disk
                self._disk_manager.deallocate_page(page_id)

                return True

    def get_page_lsn(self, page_id: PageId) -> LSN:
        """Get the LSN of a page.

        Args:
            page_id: The page to query.

        Returns:
            The page's LSN.

        Raises:
            ValueError: If page not in buffer pool.
        """
        with self._lock:
            if page_id not in self._page_table:
                raise ValueError(f"Page {page_id} not in buffer pool")

            frame_idx = self._page_table[page_id]
            frame = self._frames[frame_idx]

            with frame.latch:
                if frame.page is None:
                    raise ValueError(f"Page {page_id} not in buffer pool")
                return frame.page.lsn

    def set_page_lsn(self, page_id: PageId, lsn: LSN) -> None:
        """Set the LSN of a page.

        Args:
            page_id: The page to update.
            lsn: The new LSN.

        Raises:
            ValueError: If page not in buffer pool.
        """
        with self._lock:
            if page_id not in self._page_table:
                raise ValueError(f"Page {page_id} not in buffer pool")

            frame_idx = self._page_table[page_id]
            frame = self._frames[frame_idx]

            with frame.latch:
                if frame.page is None:
                    raise ValueError(f"Page {page_id} not in buffer pool")
                frame.page.lsn = lsn
                frame.is_dirty = True

    def get_stats(self) -> BufferPoolStats:
        """Return buffer pool statistics for monitoring."""
        with self._lock:
            pages_in_use = sum(1 for f in self._frames if f.pin_count > 0)
            dirty_pages = sum(1 for f in self._frames if f.is_dirty)

            return BufferPoolStats(
                pool_size=self._pool_size,
                pages_in_use=pages_in_use,
                dirty_pages=dirty_pages,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                eviction_count=self._eviction_count,
            )
