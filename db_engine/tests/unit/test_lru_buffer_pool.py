"""Unit tests for LRUBufferPool."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from db_engine.adapters.outbound import FileDiskManager, LRUBufferPool
from db_engine.domain.value_objects import LSN, PageId
from db_engine.ports.inbound.buffer_pool import BufferPoolFullError


class TestLRUBufferPool:
    """Tests for LRUBufferPool."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def disk_manager(self, temp_dir: Path) -> FileDiskManager:
        """Create a disk manager for testing."""
        db_path = temp_dir / "test.db"
        dm = FileDiskManager(db_path, page_size=4096)
        yield dm
        dm.close()

    @pytest.fixture
    def buffer_pool(self, disk_manager: FileDiskManager) -> LRUBufferPool:
        """Create a buffer pool for testing."""
        return LRUBufferPool(pool_size=4, disk_manager=disk_manager)

    def test_creation(self, disk_manager: FileDiskManager) -> None:
        """Buffer pool can be created."""
        bp = LRUBufferPool(pool_size=10, disk_manager=disk_manager)
        assert bp.pool_size == 10

    def test_invalid_pool_size(self, disk_manager: FileDiskManager) -> None:
        """Pool size must be >= 1."""
        with pytest.raises(ValueError, match="Pool size must be >= 1"):
            LRUBufferPool(pool_size=0, disk_manager=disk_manager)

    def test_new_page(self, buffer_pool: LRUBufferPool) -> None:
        """New pages can be allocated."""
        page = buffer_pool.new_page()

        assert page is not None
        assert page.page_id == PageId(1)
        assert page.get_free_space() > 0

    def test_new_page_is_dirty(
        self, buffer_pool: LRUBufferPool, disk_manager: FileDiskManager
    ) -> None:
        """New pages are marked dirty."""
        page = buffer_pool.new_page()
        stats = buffer_pool.get_stats()

        assert stats.dirty_pages == 1

    def test_fetch_page(
        self, buffer_pool: LRUBufferPool, disk_manager: FileDiskManager
    ) -> None:
        """Pages can be fetched after creation."""
        # Create and write a page
        page = buffer_pool.new_page()
        page_id = page.page_id
        page.insert_record(b"test data")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_page(page_id)

        # Fetch should work
        fetched = buffer_pool.fetch_page(page_id)
        assert fetched.page_id == page_id

        record = fetched.get_record(0)
        assert record == b"test data"

    def test_pin_count(self, buffer_pool: LRUBufferPool) -> None:
        """Pin count tracks page usage."""
        page = buffer_pool.new_page()
        page_id = page.page_id

        stats = buffer_pool.get_stats()
        assert stats.pages_in_use == 1

        buffer_pool.unpin_page(page_id, is_dirty=False)

        stats = buffer_pool.get_stats()
        assert stats.pages_in_use == 0

    def test_multiple_pins(self, buffer_pool: LRUBufferPool) -> None:
        """Multiple fetches increase pin count."""
        page = buffer_pool.new_page()
        page_id = page.page_id

        # Fetch again (increases pin count)
        page2 = buffer_pool.fetch_page(page_id)

        # Need two unpins
        buffer_pool.unpin_page(page_id, is_dirty=False)
        stats = buffer_pool.get_stats()
        assert stats.pages_in_use == 1

        buffer_pool.unpin_page(page_id, is_dirty=False)
        stats = buffer_pool.get_stats()
        assert stats.pages_in_use == 0

    def test_dirty_flag(self, buffer_pool: LRUBufferPool) -> None:
        """Dirty flag is set correctly."""
        page = buffer_pool.new_page()  # New pages are dirty
        page_id = page.page_id

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 1

        buffer_pool.flush_page(page_id)

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 0

    def test_unpin_marks_dirty(self, buffer_pool: LRUBufferPool) -> None:
        """Unpin with is_dirty=True marks page dirty."""
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.flush_page(page_id)  # Clear dirty flag

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 0

        buffer_pool.unpin_page(page_id, is_dirty=True)

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 1

    def test_flush_page(self, buffer_pool: LRUBufferPool) -> None:
        """Flush writes dirty page to disk."""
        page = buffer_pool.new_page()
        page_id = page.page_id
        page.insert_record(b"flushed data")

        result = buffer_pool.flush_page(page_id)
        assert result is True

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 0

    def test_flush_all_pages(self, buffer_pool: LRUBufferPool) -> None:
        """Flush all writes all dirty pages."""
        # Create multiple dirty pages
        for _ in range(3):
            page = buffer_pool.new_page()
            page.insert_record(b"data")
            buffer_pool.unpin_page(page.page_id, is_dirty=True)

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 3

        buffer_pool.flush_all_pages()

        stats = buffer_pool.get_stats()
        assert stats.dirty_pages == 0

    def test_lru_eviction(self, buffer_pool: LRUBufferPool) -> None:
        """LRU eviction works when pool is full."""
        # Fill the pool (size=4)
        pages = []
        for _ in range(4):
            page = buffer_pool.new_page()
            pages.append(page.page_id)
            buffer_pool.unpin_page(page.page_id, is_dirty=True)

        # Allocate one more - should evict oldest
        new_page = buffer_pool.new_page()

        stats = buffer_pool.get_stats()
        assert stats.eviction_count == 1

    def test_buffer_pool_full_all_pinned(self, buffer_pool: LRUBufferPool) -> None:
        """Error when all pages are pinned and eviction needed."""
        # Fill the pool with pinned pages
        for _ in range(4):
            buffer_pool.new_page()  # All pinned

        with pytest.raises(BufferPoolFullError):
            buffer_pool.new_page()

    def test_cache_hit(self, buffer_pool: LRUBufferPool) -> None:
        """Fetching cached page is a hit."""
        page = buffer_pool.new_page()
        page_id = page.page_id

        # Fetch should hit
        buffer_pool.fetch_page(page_id)

        stats = buffer_pool.get_stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 0

    def test_delete_page(self, buffer_pool: LRUBufferPool) -> None:
        """Pages can be deleted."""
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.unpin_page(page_id, is_dirty=False)

        result = buffer_pool.delete_page(page_id)
        assert result is True

    def test_delete_pinned_page(self, buffer_pool: LRUBufferPool) -> None:
        """Cannot delete pinned page."""
        page = buffer_pool.new_page()
        page_id = page.page_id
        # Not unpinned

        with pytest.raises(ValueError, match="pinned"):
            buffer_pool.delete_page(page_id)

    def test_get_set_page_lsn(self, buffer_pool: LRUBufferPool) -> None:
        """Page LSN can be get/set."""
        page = buffer_pool.new_page()
        page_id = page.page_id

        buffer_pool.set_page_lsn(page_id, LSN(100))
        lsn = buffer_pool.get_page_lsn(page_id)

        assert lsn == LSN(100)

    def test_stats(self, buffer_pool: LRUBufferPool) -> None:
        """Stats are tracked correctly."""
        stats = buffer_pool.get_stats()

        assert stats.pool_size == 4
        assert stats.pages_in_use == 0
        assert stats.dirty_pages == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.hit_ratio == 0.0


class TestLRUBufferPoolPersistence:
    """Tests for buffer pool persistence."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_data_persists(self, temp_dir: Path) -> None:
        """Data written through buffer pool persists."""
        db_path = temp_dir / "persist.db"

        # Write through buffer pool
        dm1 = FileDiskManager(db_path, page_size=4096)
        bp1 = LRUBufferPool(pool_size=4, disk_manager=dm1)

        page = bp1.new_page()
        page_id = page.page_id
        page.insert_record(b"persistent data")
        bp1.unpin_page(page_id, is_dirty=True)
        bp1.flush_all_pages()
        dm1.close()

        # Read after restart
        dm2 = FileDiskManager(db_path, page_size=4096)
        bp2 = LRUBufferPool(pool_size=4, disk_manager=dm2)

        fetched = bp2.fetch_page(page_id)
        record = fetched.get_record(0)
        assert record == b"persistent data"
        dm2.close()
