"""Unit tests for FileDiskManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from db_engine.adapters.outbound import FileDiskManager
from db_engine.domain.value_objects import PageId


class TestFileDiskManager:
    """Tests for FileDiskManager."""

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

    def test_creation_new_file(self, temp_dir: Path) -> None:
        """New file is created with header page."""
        db_path = temp_dir / "new.db"
        dm = FileDiskManager(db_path, page_size=4096)

        assert dm.page_size == 4096
        assert dm.get_num_pages() == 0  # No user pages yet
        assert db_path.exists()

        dm.close()

    def test_reopen_existing(self, temp_dir: Path) -> None:
        """Existing file can be reopened."""
        db_path = temp_dir / "existing.db"

        # Create and close
        dm1 = FileDiskManager(db_path, page_size=4096)
        page_id = dm1.allocate_page()
        dm1.write_page(page_id, b"x" * 4096)
        dm1.close()

        # Reopen
        dm2 = FileDiskManager(db_path, page_size=4096)
        assert dm2.get_num_pages() == 1
        data = dm2.read_page(page_id)
        assert data == b"x" * 4096
        dm2.close()

    def test_page_size_mismatch(self, temp_dir: Path) -> None:
        """Opening with wrong page size raises error."""
        db_path = temp_dir / "mismatch.db"

        dm1 = FileDiskManager(db_path, page_size=4096)
        dm1.close()

        with pytest.raises(ValueError, match="Page size mismatch"):
            FileDiskManager(db_path, page_size=8192)

    def test_allocate_page(self, disk_manager: FileDiskManager) -> None:
        """Pages can be allocated."""
        page_id = disk_manager.allocate_page()
        assert page_id == PageId(1)  # First user page

        page_id2 = disk_manager.allocate_page()
        assert page_id2 == PageId(2)

        assert disk_manager.get_num_pages() == 2

    def test_write_read_page(self, disk_manager: FileDiskManager) -> None:
        """Pages can be written and read."""
        page_id = disk_manager.allocate_page()

        # Write data
        data = b"Hello, World!" + b"\x00" * (4096 - 13)
        disk_manager.write_page(page_id, data)

        # Read data
        read_data = disk_manager.read_page(page_id)
        assert read_data == data

    def test_write_wrong_size(self, disk_manager: FileDiskManager) -> None:
        """Writing wrong-sized data raises error."""
        page_id = disk_manager.allocate_page()

        with pytest.raises(ValueError, match="Data size mismatch"):
            disk_manager.write_page(page_id, b"too short")

    def test_read_invalid_page(self, disk_manager: FileDiskManager) -> None:
        """Reading invalid page raises error."""
        with pytest.raises(ValueError, match="Invalid page_id"):
            disk_manager.read_page(PageId(999))

    def test_write_to_header_page(self, disk_manager: FileDiskManager) -> None:
        """Cannot write to header page."""
        with pytest.raises(ValueError, match="reserved"):
            disk_manager.write_page(PageId(0), b"\x00" * 4096)

    def test_deallocate_page(self, disk_manager: FileDiskManager) -> None:
        """Deallocated pages can be reused."""
        page_id1 = disk_manager.allocate_page()
        page_id2 = disk_manager.allocate_page()

        disk_manager.deallocate_page(page_id1)

        # Next allocation should reuse page_id1
        page_id3 = disk_manager.allocate_page()
        assert page_id3 == page_id1

    def test_deallocate_twice(self, disk_manager: FileDiskManager) -> None:
        """Cannot deallocate the same page twice."""
        page_id = disk_manager.allocate_page()
        disk_manager.deallocate_page(page_id)

        with pytest.raises(ValueError, match="already free"):
            disk_manager.deallocate_page(page_id)

    def test_sync(self, disk_manager: FileDiskManager) -> None:
        """Sync flushes data to disk."""
        page_id = disk_manager.allocate_page()
        disk_manager.write_page(page_id, b"\x00" * 4096)
        disk_manager.sync()  # Should not raise

    def test_context_manager(self, temp_dir: Path) -> None:
        """Disk manager works as context manager."""
        db_path = temp_dir / "context.db"

        with FileDiskManager(db_path, page_size=4096) as dm:
            page_id = dm.allocate_page()
            dm.write_page(page_id, b"data" + b"\x00" * 4092)

        # File should still exist and be readable
        with FileDiskManager(db_path, page_size=4096) as dm:
            data = dm.read_page(PageId(1))
            assert data[:4] == b"data"

    def test_multiple_pages(self, disk_manager: FileDiskManager) -> None:
        """Multiple pages can be allocated and written."""
        pages = []
        for i in range(10):
            page_id = disk_manager.allocate_page()
            data = bytes([i] * 4096)
            disk_manager.write_page(page_id, data)
            pages.append(page_id)

        assert disk_manager.get_num_pages() == 10

        # Verify all pages
        for i, page_id in enumerate(pages):
            data = disk_manager.read_page(page_id)
            assert data == bytes([i] * 4096)


class TestFileDiskManagerPersistence:
    """Tests for persistence across restarts."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_data_persists_across_restarts(self, temp_dir: Path) -> None:
        """Data written survives closing and reopening."""
        db_path = temp_dir / "persist.db"

        # Write data
        dm1 = FileDiskManager(db_path, page_size=4096)
        pages = []
        for i in range(5):
            page_id = dm1.allocate_page()
            dm1.write_page(page_id, bytes([i]) * 4096)
            pages.append(page_id)
        dm1.close()

        # Read data after restart
        dm2 = FileDiskManager(db_path, page_size=4096)
        assert dm2.get_num_pages() == 5
        for i, page_id in enumerate(pages):
            data = dm2.read_page(page_id)
            assert data == bytes([i]) * 4096
        dm2.close()
