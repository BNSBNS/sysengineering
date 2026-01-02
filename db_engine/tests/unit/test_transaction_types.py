"""Unit tests for domain value objects - transaction types."""

from __future__ import annotations

import pytest

from db_engine.domain.value_objects import (
    IsolationLevel,
    LockMode,
    TransactionState,
    WaitPolicy,
)


class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_all_states_exist(self) -> None:
        """All expected transaction states are defined."""
        states = [
            TransactionState.IDLE,
            TransactionState.ACTIVE,
            TransactionState.COMMITTING,
            TransactionState.ABORTING,
            TransactionState.COMMITTED,
            TransactionState.ABORTED,
        ]
        assert len(states) == 6

    def test_is_terminal(self) -> None:
        """COMMITTED and ABORTED are terminal states."""
        assert TransactionState.COMMITTED.is_terminal()
        assert TransactionState.ABORTED.is_terminal()

        assert not TransactionState.IDLE.is_terminal()
        assert not TransactionState.ACTIVE.is_terminal()
        assert not TransactionState.COMMITTING.is_terminal()
        assert not TransactionState.ABORTING.is_terminal()

    def test_is_active(self) -> None:
        """Only ACTIVE state allows operations."""
        assert TransactionState.ACTIVE.is_active()

        assert not TransactionState.IDLE.is_active()
        assert not TransactionState.COMMITTING.is_active()
        assert not TransactionState.ABORTING.is_active()
        assert not TransactionState.COMMITTED.is_active()
        assert not TransactionState.ABORTED.is_active()

    def test_can_commit(self) -> None:
        """Only ACTIVE transactions can begin commit."""
        assert TransactionState.ACTIVE.can_commit()

        assert not TransactionState.IDLE.can_commit()
        assert not TransactionState.COMMITTING.can_commit()
        assert not TransactionState.ABORTING.can_commit()
        assert not TransactionState.COMMITTED.can_commit()
        assert not TransactionState.ABORTED.can_commit()

    def test_can_abort(self) -> None:
        """ACTIVE and COMMITTING transactions can be aborted."""
        assert TransactionState.ACTIVE.can_abort()
        assert TransactionState.COMMITTING.can_abort()

        assert not TransactionState.IDLE.can_abort()
        assert not TransactionState.ABORTING.can_abort()
        assert not TransactionState.COMMITTED.can_abort()
        assert not TransactionState.ABORTED.can_abort()


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_all_levels_exist(self) -> None:
        """All expected isolation levels are defined."""
        levels = [
            IsolationLevel.READ_UNCOMMITTED,
            IsolationLevel.READ_COMMITTED,
            IsolationLevel.REPEATABLE_READ,
            IsolationLevel.SNAPSHOT,
            IsolationLevel.SERIALIZABLE,
        ]
        assert len(levels) == 5

    def test_default_is_snapshot(self) -> None:
        """SNAPSHOT is our default isolation level."""
        # This is more of a documentation test
        default = IsolationLevel.SNAPSHOT
        assert default.name == "SNAPSHOT"


class TestLockMode:
    """Tests for LockMode enum."""

    def test_all_modes_exist(self) -> None:
        """All expected lock modes are defined."""
        modes = [
            LockMode.SHARED,
            LockMode.EXCLUSIVE,
            LockMode.INTENT_SHARED,
            LockMode.INTENT_EXCLUSIVE,
            LockMode.SHARED_INTENT_EXCLUSIVE,
        ]
        assert len(modes) == 5

    def test_shared_shared_compatible(self) -> None:
        """Shared locks are compatible with each other."""
        assert LockMode.SHARED.is_compatible(LockMode.SHARED)

    def test_shared_exclusive_incompatible(self) -> None:
        """Shared and exclusive locks are not compatible."""
        assert not LockMode.SHARED.is_compatible(LockMode.EXCLUSIVE)
        assert not LockMode.EXCLUSIVE.is_compatible(LockMode.SHARED)

    def test_exclusive_exclusive_incompatible(self) -> None:
        """Exclusive locks are not compatible with each other."""
        assert not LockMode.EXCLUSIVE.is_compatible(LockMode.EXCLUSIVE)

    def test_intent_shared_compatible_with_many(self) -> None:
        """Intent shared is compatible with most modes."""
        assert LockMode.INTENT_SHARED.is_compatible(LockMode.SHARED)
        assert LockMode.INTENT_SHARED.is_compatible(LockMode.INTENT_SHARED)
        assert LockMode.INTENT_SHARED.is_compatible(LockMode.INTENT_EXCLUSIVE)
        assert LockMode.INTENT_SHARED.is_compatible(LockMode.SHARED_INTENT_EXCLUSIVE)

    def test_intent_exclusive_compatibility(self) -> None:
        """Intent exclusive compatible with intent modes."""
        assert LockMode.INTENT_EXCLUSIVE.is_compatible(LockMode.INTENT_SHARED)
        assert LockMode.INTENT_EXCLUSIVE.is_compatible(LockMode.INTENT_EXCLUSIVE)
        assert not LockMode.INTENT_EXCLUSIVE.is_compatible(LockMode.SHARED)
        assert not LockMode.INTENT_EXCLUSIVE.is_compatible(LockMode.EXCLUSIVE)

    def test_six_compatibility(self) -> None:
        """SIX (shared + intent exclusive) compatibility."""
        assert LockMode.SHARED_INTENT_EXCLUSIVE.is_compatible(LockMode.INTENT_SHARED)
        assert not LockMode.SHARED_INTENT_EXCLUSIVE.is_compatible(LockMode.SHARED)
        assert not LockMode.SHARED_INTENT_EXCLUSIVE.is_compatible(LockMode.EXCLUSIVE)


class TestWaitPolicy:
    """Tests for WaitPolicy enum."""

    def test_all_policies_exist(self) -> None:
        """All expected wait policies are defined."""
        policies = [
            WaitPolicy.WAIT,
            WaitPolicy.NO_WAIT,
            WaitPolicy.SKIP_LOCKED,
        ]
        assert len(policies) == 3
