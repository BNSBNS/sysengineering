"""Unit tests for MVCCTransactionManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from db_engine.adapters.outbound import FileDiskManager, FileWALWriter, LRUBufferPool
from db_engine.domain.services import LockManager, MVCCTransactionManager
from db_engine.domain.value_objects import PageId, TransactionId
from db_engine.domain.value_objects.transaction_types import (
    IsolationLevel,
    TransactionState,
)
from db_engine.ports.inbound.transaction_manager import LockConflictError
from db_engine.ports.outbound.wal_writer import SyncMode


class TestTransactionManagerBasic:
    """Basic transaction manager tests."""

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
    def txn_manager(
        self, wal_writer: FileWALWriter, buffer_pool: LRUBufferPool
    ) -> MVCCTransactionManager:
        """Create a transaction manager for testing."""
        return MVCCTransactionManager(wal_writer, buffer_pool)

    def test_begin_transaction(self, txn_manager: MVCCTransactionManager) -> None:
        """Transaction can be started."""
        txn = txn_manager.begin()

        assert txn is not None
        assert txn.txn_id == TransactionId(1)
        assert txn.state == TransactionState.ACTIVE
        assert txn.snapshot is not None

    def test_begin_multiple_transactions(
        self, txn_manager: MVCCTransactionManager
    ) -> None:
        """Multiple transactions get unique IDs."""
        txn1 = txn_manager.begin()
        txn2 = txn_manager.begin()
        txn3 = txn_manager.begin()

        assert txn1.txn_id == TransactionId(1)
        assert txn2.txn_id == TransactionId(2)
        assert txn3.txn_id == TransactionId(3)

    def test_commit_transaction(self, txn_manager: MVCCTransactionManager) -> None:
        """Transaction can be committed."""
        txn = txn_manager.begin()
        txn_manager.commit(txn)

        assert txn.state == TransactionState.COMMITTED

    def test_abort_transaction(self, txn_manager: MVCCTransactionManager) -> None:
        """Transaction can be aborted."""
        txn = txn_manager.begin()
        txn_manager.abort(txn)

        assert txn.state == TransactionState.ABORTED

    def test_commit_non_active_raises(self, txn_manager: MVCCTransactionManager) -> None:
        """Committing non-active transaction raises."""
        txn = txn_manager.begin()
        txn_manager.commit(txn)

        with pytest.raises(ValueError, match="not active"):
            txn_manager.commit(txn)

    def test_abort_non_active_raises(self, txn_manager: MVCCTransactionManager) -> None:
        """Aborting non-active transaction raises."""
        txn = txn_manager.begin()
        txn_manager.commit(txn)

        with pytest.raises(ValueError, match="not active"):
            txn_manager.abort(txn)

    def test_get_active_transactions(self, txn_manager: MVCCTransactionManager) -> None:
        """Active transactions are tracked."""
        txn1 = txn_manager.begin()
        txn2 = txn_manager.begin()

        active = txn_manager.get_active_transactions()

        assert TransactionId(1) in active
        assert TransactionId(2) in active

        txn_manager.commit(txn1)

        active = txn_manager.get_active_transactions()
        assert TransactionId(1) not in active
        assert TransactionId(2) in active

    def test_get_stats(self, txn_manager: MVCCTransactionManager) -> None:
        """Statistics are tracked."""
        txn1 = txn_manager.begin()
        txn2 = txn_manager.begin()
        txn_manager.commit(txn1)
        txn_manager.abort(txn2)

        stats = txn_manager.get_stats()

        assert stats.active_count == 0
        assert stats.committed_total == 1
        assert stats.aborted_total == 1


class TestTransactionManagerSnapshot:
    """MVCC snapshot tests."""

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
    def txn_manager(
        self, wal_writer: FileWALWriter, buffer_pool: LRUBufferPool
    ) -> MVCCTransactionManager:
        """Create a transaction manager for testing."""
        return MVCCTransactionManager(wal_writer, buffer_pool)

    def test_snapshot_captures_active_transactions(
        self, txn_manager: MVCCTransactionManager
    ) -> None:
        """Snapshot captures active transactions at start time."""
        txn1 = txn_manager.begin()  # T1 is active

        txn2 = txn_manager.begin()  # T2 starts, sees T1 as active

        # T2's snapshot should include T1
        snapshot = txn_manager.get_snapshot(txn2)
        assert txn1.txn_id in snapshot.active_txns

    def test_snapshot_isolation(self, txn_manager: MVCCTransactionManager) -> None:
        """Snapshot isolation - transaction sees consistent view."""
        # Start with snapshot isolation
        txn = txn_manager.begin(IsolationLevel.SNAPSHOT)

        # Snapshot is taken at begin time
        snapshot = txn_manager.get_snapshot(txn)

        # Same snapshot returned for subsequent calls
        snapshot2 = txn_manager.get_snapshot(txn)

        assert snapshot.txn_id == snapshot2.txn_id

    def test_read_committed_fresh_snapshot(
        self, txn_manager: MVCCTransactionManager
    ) -> None:
        """Read committed gets fresh snapshot each time."""
        txn = txn_manager.begin(IsolationLevel.READ_COMMITTED)

        # Start another transaction
        txn2 = txn_manager.begin()

        # Get snapshot - should see txn2 as active
        snapshot = txn_manager.get_snapshot(txn)
        assert txn2.txn_id in snapshot.active_txns


class TestTransactionManagerOperations:
    """Transaction data operation tests."""

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
    def txn_manager(
        self, wal_writer: FileWALWriter, buffer_pool: LRUBufferPool
    ) -> MVCCTransactionManager:
        """Create a transaction manager for testing."""
        return MVCCTransactionManager(wal_writer, buffer_pool)

    def test_insert_acquires_lock(
        self, txn_manager: MVCCTransactionManager, buffer_pool: LRUBufferPool
    ) -> None:
        """Insert acquires exclusive lock."""
        # Create a page first
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.unpin_page(page_id, is_dirty=False)

        txn = txn_manager.begin()
        rid = txn_manager.insert(txn, page_id, 0, b"test data")

        assert rid.page_id == page_id
        assert (page_id, 0) in txn.locks_held

    def test_insert_conflict(
        self, txn_manager: MVCCTransactionManager, buffer_pool: LRUBufferPool
    ) -> None:
        """Insert conflicts with lock held by another transaction."""
        # Create a page
        page = buffer_pool.new_page()
        page_id = page.page_id
        buffer_pool.unpin_page(page_id, is_dirty=False)

        # T1 inserts and holds lock
        txn1 = txn_manager.begin()
        txn_manager.insert(txn1, page_id, 0, b"t1 data")

        # T2 cannot insert at same location
        txn2 = txn_manager.begin()
        with pytest.raises(LockConflictError):
            txn_manager.insert(txn2, page_id, 0, b"t2 data")

    def test_update_with_transaction(
        self, txn_manager: MVCCTransactionManager, buffer_pool: LRUBufferPool
    ) -> None:
        """Update within a transaction."""
        # Create a page with data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"original")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_page(page_id)

        txn = txn_manager.begin()
        txn_manager.update(txn, page_id, slot_id, b"original", b"updated")
        txn_manager.commit(txn)

        # Verify update persisted
        page2 = buffer_pool.fetch_page(page_id)
        data = page2.get_record(slot_id)
        buffer_pool.unpin_page(page_id, is_dirty=False)

        assert data == b"updated"

    def test_delete_with_transaction(
        self, txn_manager: MVCCTransactionManager, buffer_pool: LRUBufferPool
    ) -> None:
        """Delete within a transaction."""
        # Create a page with data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"to delete")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_page(page_id)

        txn = txn_manager.begin()
        txn_manager.delete(txn, page_id, slot_id, b"to delete")
        txn_manager.commit(txn)

        # Verify delete persisted
        page2 = buffer_pool.fetch_page(page_id)
        data = page2.get_record(slot_id)
        buffer_pool.unpin_page(page_id, is_dirty=False)

        assert data is None  # Deleted

    def test_abort_rolls_back_changes(
        self, txn_manager: MVCCTransactionManager, buffer_pool: LRUBufferPool
    ) -> None:
        """Abort rolls back uncommitted changes."""
        # Create a page with data
        page = buffer_pool.new_page()
        page_id = page.page_id
        slot_id = page.insert_record(b"original")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_page(page_id)

        # Update and abort
        txn = txn_manager.begin()
        txn_manager.update(txn, page_id, slot_id, b"original", b"changed")
        txn_manager.abort(txn)

        # Data should be restored
        page2 = buffer_pool.fetch_page(page_id)
        data = page2.get_record(slot_id)
        buffer_pool.unpin_page(page_id, is_dirty=False)

        assert data == b"original"
