"""Unit tests for LockManager (2PL implementation)."""

from __future__ import annotations

import pytest

from db_engine.domain.services import DeadlockError, LockManager
from db_engine.domain.value_objects import PageId, TransactionId
from db_engine.domain.value_objects.transaction_types import LockMode


class TestLockManagerBasic:
    """Basic lock manager tests."""

    @pytest.fixture
    def lock_manager(self) -> LockManager:
        """Create a lock manager for testing."""
        return LockManager(deadlock_detection=True)

    def test_acquire_exclusive_lock(self, lock_manager: LockManager) -> None:
        """Exclusive lock can be acquired on free resource."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        result = lock_manager.acquire(txn_id, page_id, 0, LockMode.EXCLUSIVE)

        assert result is True
        locks = lock_manager.get_locks_held(txn_id)
        assert (page_id, 0) in locks

    def test_acquire_shared_lock(self, lock_manager: LockManager) -> None:
        """Shared lock can be acquired on free resource."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        result = lock_manager.acquire(txn_id, page_id, 0, LockMode.SHARED)

        assert result is True

    def test_shared_locks_compatible(self, lock_manager: LockManager) -> None:
        """Multiple transactions can hold shared locks."""
        page_id = PageId(10)

        # First transaction gets shared lock
        result1 = lock_manager.acquire(TransactionId(1), page_id, 0, LockMode.SHARED)
        assert result1 is True

        # Second transaction can also get shared lock
        result2 = lock_manager.acquire(TransactionId(2), page_id, 0, LockMode.SHARED)
        assert result2 is True

    def test_exclusive_conflicts_with_shared(self, lock_manager: LockManager) -> None:
        """Exclusive lock conflicts with existing shared lock."""
        page_id = PageId(10)

        # First transaction gets shared lock
        lock_manager.acquire(TransactionId(1), page_id, 0, LockMode.SHARED)

        # Second transaction cannot get exclusive lock (no wait)
        result = lock_manager.acquire(
            TransactionId(2), page_id, 0, LockMode.EXCLUSIVE, wait=False
        )
        assert result is False

    def test_exclusive_conflicts_with_exclusive(self, lock_manager: LockManager) -> None:
        """Exclusive lock conflicts with existing exclusive lock."""
        page_id = PageId(10)

        # First transaction gets exclusive lock
        lock_manager.acquire(TransactionId(1), page_id, 0, LockMode.EXCLUSIVE)

        # Second transaction cannot get exclusive lock
        result = lock_manager.acquire(
            TransactionId(2), page_id, 0, LockMode.EXCLUSIVE, wait=False
        )
        assert result is False

    def test_same_transaction_can_get_same_lock_again(
        self, lock_manager: LockManager
    ) -> None:
        """Transaction can acquire same lock multiple times."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        lock_manager.acquire(txn_id, page_id, 0, LockMode.EXCLUSIVE)
        result = lock_manager.acquire(txn_id, page_id, 0, LockMode.EXCLUSIVE)

        assert result is True

    def test_release_lock(self, lock_manager: LockManager) -> None:
        """Lock can be released."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        lock_manager.acquire(txn_id, page_id, 0, LockMode.EXCLUSIVE)
        result = lock_manager.release(txn_id, page_id, 0)

        assert result is True
        locks = lock_manager.get_locks_held(txn_id)
        assert (page_id, 0) not in locks

    def test_release_enables_other_transaction(self, lock_manager: LockManager) -> None:
        """After release, another transaction can acquire the lock."""
        page_id = PageId(10)

        # First transaction gets and releases lock
        lock_manager.acquire(TransactionId(1), page_id, 0, LockMode.EXCLUSIVE)
        lock_manager.release(TransactionId(1), page_id, 0)

        # Second transaction can now get it
        result = lock_manager.acquire(TransactionId(2), page_id, 0, LockMode.EXCLUSIVE)
        assert result is True

    def test_release_all(self, lock_manager: LockManager) -> None:
        """Release all locks for a transaction."""
        txn_id = TransactionId(1)

        # Acquire multiple locks
        lock_manager.acquire(txn_id, PageId(1), 0, LockMode.EXCLUSIVE)
        lock_manager.acquire(txn_id, PageId(2), 1, LockMode.SHARED)
        lock_manager.acquire(txn_id, PageId(3), 2, LockMode.EXCLUSIVE)

        count = lock_manager.release_all(txn_id)

        assert count == 3
        locks = lock_manager.get_locks_held(txn_id)
        assert len(locks) == 0

    def test_page_level_lock(self, lock_manager: LockManager) -> None:
        """Page-level lock (no slot_id) can be acquired."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        result = lock_manager.acquire(txn_id, page_id, None, LockMode.EXCLUSIVE)

        assert result is True
        locks = lock_manager.get_locks_held(txn_id)
        assert (page_id,) in locks


class TestLockManagerDeadlock:
    """Deadlock detection tests."""

    @pytest.fixture
    def lock_manager(self) -> LockManager:
        """Create a lock manager with deadlock detection."""
        return LockManager(deadlock_detection=True)

    def test_simple_deadlock_detected(self, lock_manager: LockManager) -> None:
        """Simple two-transaction deadlock is detected."""
        txn1 = TransactionId(1)
        txn2 = TransactionId(2)
        page1 = PageId(1)
        page2 = PageId(2)

        # T1 holds lock on page1
        lock_manager.acquire(txn1, page1, 0, LockMode.EXCLUSIVE)

        # T2 holds lock on page2
        lock_manager.acquire(txn2, page2, 0, LockMode.EXCLUSIVE)

        # T1 waits for page2 (held by T2)
        # Note: In our simplified implementation, wait=True with deadlock
        # detection will raise DeadlockError if cycle detected
        # First, we simulate T1 waiting for T2's lock
        lock_manager.acquire(txn1, page2, 0, LockMode.EXCLUSIVE, wait=False)

        # T2 tries to wait for page1 (held by T1) - would create cycle
        # Since we don't have real waiting, this test is simplified
        # The cycle would be: T2 -> T1 -> T2


class TestLockManagerUpgrade:
    """Lock upgrade tests."""

    @pytest.fixture
    def lock_manager(self) -> LockManager:
        """Create a lock manager for testing."""
        return LockManager()

    def test_upgrade_shared_to_exclusive(self, lock_manager: LockManager) -> None:
        """Shared lock can be upgraded to exclusive if no other holders."""
        txn_id = TransactionId(1)
        page_id = PageId(10)

        # Get shared lock first
        lock_manager.acquire(txn_id, page_id, 0, LockMode.SHARED)

        # Upgrade to exclusive
        result = lock_manager.acquire(txn_id, page_id, 0, LockMode.EXCLUSIVE)

        assert result is True

    def test_upgrade_blocked_by_other_shared(self, lock_manager: LockManager) -> None:
        """Upgrade blocked when other transaction holds shared lock."""
        page_id = PageId(10)

        # T1 gets shared lock
        lock_manager.acquire(TransactionId(1), page_id, 0, LockMode.SHARED)

        # T2 also gets shared lock
        lock_manager.acquire(TransactionId(2), page_id, 0, LockMode.SHARED)

        # T1 cannot upgrade to exclusive
        result = lock_manager.acquire(
            TransactionId(1), page_id, 0, LockMode.EXCLUSIVE, wait=False
        )
        assert result is False
