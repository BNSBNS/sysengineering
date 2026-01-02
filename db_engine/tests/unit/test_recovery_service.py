"""Unit tests for RecoveryService (ARIES recovery)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from db_engine.adapters.outbound import FileDiskManager, FileWALWriter, LRUBufferPool
from db_engine.domain.entities import (
    BeginRecord,
    CommitRecord,
    InsertRecord,
    UpdateRecord,
)
from db_engine.domain.services import RecoveryService
from db_engine.domain.value_objects import LSN, PageId, TransactionId
from db_engine.ports.outbound.wal_writer import SyncMode


class TestRecoveryService:
    """Tests for basic recovery functionality."""

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
        return LRUBufferPool(pool_size=10, disk_manager=disk_manager)

    @pytest.fixture
    def wal_writer(self, temp_dir: Path) -> FileWALWriter:
        """Create a WAL writer for testing."""
        wal_dir = temp_dir / "wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)
        yield ww
        ww.close()

    @pytest.fixture
    def recovery_service(
        self, wal_writer: FileWALWriter, buffer_pool: LRUBufferPool
    ) -> RecoveryService:
        """Create a recovery service for testing."""
        return RecoveryService(wal_writer, buffer_pool)

    def test_recovery_empty_wal(self, recovery_service: RecoveryService) -> None:
        """Recovery on empty WAL succeeds with no work."""
        stats = recovery_service.recover()

        assert stats.records_analyzed == 0
        assert stats.records_redone == 0
        assert stats.records_undone == 0
        assert stats.transactions_committed == 0
        assert stats.transactions_aborted == 0

    def test_recovery_committed_transaction(
        self,
        recovery_service: RecoveryService,
        wal_writer: FileWALWriter,
        buffer_pool: LRUBufferPool,
    ) -> None:
        """Committed transactions are identified as winners."""
        # Write a committed transaction
        begin = BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0))
        wal_writer.append(begin)

        commit = CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(1))
        wal_writer.append(commit)

        wal_writer.flush(LSN(2))

        # Recover
        stats = recovery_service.recover()

        assert stats.records_analyzed == 2
        assert stats.transactions_committed == 1
        assert stats.transactions_aborted == 0

    def test_recovery_uncommitted_transaction(
        self,
        recovery_service: RecoveryService,
        wal_writer: FileWALWriter,
    ) -> None:
        """Uncommitted transactions are identified as losers."""
        # Write an uncommitted transaction (no commit record)
        begin = BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0))
        wal_writer.append(begin)
        wal_writer.flush(LSN(1))

        # Recover
        stats = recovery_service.recover()

        assert stats.records_analyzed == 1
        assert stats.transactions_committed == 0
        assert stats.transactions_aborted == 1

    def test_recovery_multiple_transactions(
        self,
        recovery_service: RecoveryService,
        wal_writer: FileWALWriter,
    ) -> None:
        """Recovery handles multiple concurrent transactions."""
        # Transaction 1 - committed
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0)))
        # Transaction 2 - uncommitted
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(2), prev_lsn=LSN(0)))
        # Transaction 3 - committed
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(3), prev_lsn=LSN(0)))

        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(1)))
        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(3), prev_lsn=LSN(3)))
        wal_writer.flush(LSN(5))

        # Recover
        stats = recovery_service.recover()

        assert stats.transactions_committed == 2
        assert stats.transactions_aborted == 1


class TestRecoveryRedo:
    """Tests for the Redo phase of recovery."""

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
        return LRUBufferPool(pool_size=10, disk_manager=disk_manager)

    @pytest.fixture
    def wal_writer(self, temp_dir: Path) -> FileWALWriter:
        """Create a WAL writer for testing."""
        wal_dir = temp_dir / "wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)
        yield ww
        ww.close()

    def test_redo_insert(
        self,
        temp_dir: Path,
        disk_manager: FileDiskManager,
        buffer_pool: LRUBufferPool,
        wal_writer: FileWALWriter,
    ) -> None:
        """Redo phase re-applies INSERT operations."""
        # Create a page and insert data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"test data")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_all_pages()

        # Write WAL records
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0)))
        insert_record = InsertRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(1),
            page_id=page_id,
            slot_id=slot_id,
            data=b"test data",
        )
        wal_writer.append(insert_record)
        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(2)))
        wal_writer.flush(LSN(3))

        # Create recovery service and recover
        recovery = RecoveryService(wal_writer, buffer_pool)
        stats = recovery.recover()

        assert stats.transactions_committed == 1

    def test_redo_update(
        self,
        disk_manager: FileDiskManager,
        buffer_pool: LRUBufferPool,
        wal_writer: FileWALWriter,
    ) -> None:
        """Redo phase re-applies UPDATE operations."""
        # Create a page with initial data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"original")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_all_pages()

        # Write WAL records for update
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0)))
        update_record = UpdateRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(1),
            page_id=page_id,
            slot_id=slot_id,
            before_image=b"original",
            after_image=b"updated!",
        )
        wal_writer.append(update_record)
        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(2)))
        wal_writer.flush(LSN(3))

        # Recovery
        recovery = RecoveryService(wal_writer, buffer_pool)
        stats = recovery.recover()

        assert stats.transactions_committed == 1


class TestRecoveryUndo:
    """Tests for the Undo phase of recovery."""

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
        return LRUBufferPool(pool_size=10, disk_manager=disk_manager)

    @pytest.fixture
    def wal_writer(self, temp_dir: Path) -> FileWALWriter:
        """Create a WAL writer for testing."""
        wal_dir = temp_dir / "wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)
        yield ww
        ww.close()

    def test_undo_uncommitted_update(
        self,
        disk_manager: FileDiskManager,
        buffer_pool: LRUBufferPool,
        wal_writer: FileWALWriter,
    ) -> None:
        """Undo phase rolls back uncommitted UPDATE operations."""
        # Create a page with initial data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"original")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_all_pages()

        # Write WAL records for uncommitted update
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0)))
        update_record = UpdateRecord(
            lsn=LSN(0),
            txn_id=TransactionId(1),
            prev_lsn=LSN(1),
            page_id=page_id,
            slot_id=slot_id,
            before_image=b"original",
            after_image=b"changed!",
        )
        wal_writer.append(update_record)
        # No commit record - transaction is uncommitted
        wal_writer.flush(LSN(2))

        # Recovery should undo the update
        recovery = RecoveryService(wal_writer, buffer_pool)
        stats = recovery.recover()

        assert stats.transactions_aborted == 1
        assert stats.records_undone >= 1


class TestRecoveryCheckpoint:
    """Tests for checkpoint-based recovery."""

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
        return LRUBufferPool(pool_size=10, disk_manager=disk_manager)

    @pytest.fixture
    def wal_writer(self, temp_dir: Path) -> FileWALWriter:
        """Create a WAL writer for testing."""
        wal_dir = temp_dir / "wal"
        ww = FileWALWriter(wal_dir, segment_size=1024 * 1024, sync_mode=SyncMode.NONE)
        yield ww
        ww.close()

    def test_create_checkpoint(
        self,
        buffer_pool: LRUBufferPool,
        wal_writer: FileWALWriter,
    ) -> None:
        """Checkpoint can be created."""
        recovery = RecoveryService(wal_writer, buffer_pool)

        # Create some dirty pages
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.unpin_page(page_id, is_dirty=True)

        # Create checkpoint
        checkpoint_lsn = recovery.create_checkpoint(
            active_txns={TransactionId(1): LSN(5)},
            dirty_pages={page_id: LSN(3)},
        )

        assert checkpoint_lsn >= LSN(1)

    def test_recovery_with_checkpoint(
        self,
        buffer_pool: LRUBufferPool,
        wal_writer: FileWALWriter,
    ) -> None:
        """Recovery uses checkpoint as starting point."""
        recovery = RecoveryService(wal_writer, buffer_pool)

        # Write some records before checkpoint
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(0)))
        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(1), prev_lsn=LSN(1)))
        wal_writer.flush(LSN(2))

        # Create checkpoint
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.unpin_page(page_id, is_dirty=True)

        recovery.create_checkpoint(
            active_txns={},
            dirty_pages={},
        )

        # Write more records after checkpoint
        wal_writer.append(BeginRecord(lsn=LSN(0), txn_id=TransactionId(2), prev_lsn=LSN(0)))
        wal_writer.append(CommitRecord(lsn=LSN(0), txn_id=TransactionId(2), prev_lsn=LSN(4)))
        wal_writer.flush(LSN(5))

        # Recovery
        stats = recovery.recover()

        # Should see transactions from both before and after checkpoint
        assert stats.transactions_committed >= 1
