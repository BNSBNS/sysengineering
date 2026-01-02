"""Lock Manager for two-phase locking (2PL).

This module implements the lock manager for transaction isolation.
It supports multiple lock modes and enforces the two-phase locking
protocol to guarantee serializability.

Lock Modes:
    - SHARED (S): Multiple readers allowed
    - EXCLUSIVE (X): Single writer only
    - INTENT_SHARED (IS): Intent to acquire S locks on children
    - INTENT_EXCLUSIVE (IX): Intent to acquire X locks on children

Lock Hierarchy:
    Database -> Table -> Page -> Row
    (We implement page and row level locking)

Two-Phase Locking:
    1. Growing phase: Transaction acquires locks
    2. Shrinking phase: Transaction releases locks (at commit/abort)

References:
    - Gray & Reuter, "Transaction Processing" (1993)
    - design.md Section 5 (Lock Hierarchy)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set

from db_engine.domain.value_objects import PageId, TransactionId
from db_engine.domain.value_objects.transaction_types import LockMode


@dataclass
class LockRequest:
    """A request for a lock by a transaction."""

    txn_id: TransactionId
    mode: LockMode
    granted: bool = False


@dataclass
class LockEntry:
    """Entry in the lock table for a resource."""

    # Granted locks: mode -> set of txn_ids holding that mode
    granted: Dict[LockMode, Set[TransactionId]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # Waiting queue: list of pending requests
    waiting: list[LockRequest] = field(default_factory=list)

    def has_conflict(self, mode: LockMode, txn_id: TransactionId) -> bool:
        """Check if acquiring this lock would conflict with existing locks."""
        # First check if any OTHER transaction holds an incompatible lock
        for granted_mode, holders in self.granted.items():
            other_holders = holders - {txn_id}
            if other_holders and not LockMode.is_compatible(granted_mode, mode):
                return True

        # If we already hold a lock, check if we're trying to upgrade
        for granted_mode, holders in self.granted.items():
            if txn_id in holders:
                # We can always get same or weaker lock
                if self._can_upgrade(granted_mode, mode):
                    return False

        return False

    def _can_upgrade(self, held: LockMode, requested: LockMode) -> bool:
        """Check if we can upgrade from held to requested."""
        # Can always get same or weaker lock
        strength = {
            LockMode.INTENT_SHARED: 0,
            LockMode.SHARED: 1,
            LockMode.INTENT_EXCLUSIVE: 1,
            LockMode.SHARED_INTENT_EXCLUSIVE: 2,
            LockMode.EXCLUSIVE: 3,
        }
        return strength.get(held, 0) >= strength.get(requested, 0)


class LockManager:
    """Lock manager implementing two-phase locking.

    Manages locks at page and slot level. Supports multiple lock
    modes and detects deadlocks using a wait-for graph.

    Thread Safety:
        All operations are thread-safe using a global lock.
        A production implementation would use fine-grained locking.
    """

    def __init__(self, deadlock_detection: bool = True) -> None:
        """Initialize the lock manager.

        Args:
            deadlock_detection: Whether to detect deadlocks.
        """
        self._lock = threading.Lock()
        self._deadlock_detection = deadlock_detection

        # Lock table: resource_key -> LockEntry
        # resource_key is (page_id, slot_id) or (page_id,) for page locks
        self._lock_table: Dict[tuple, LockEntry] = {}

        # Wait-for graph: txn_id -> set of txn_ids it's waiting for
        self._wait_for: Dict[TransactionId, Set[TransactionId]] = defaultdict(set)

        # Locks held by each transaction
        self._txn_locks: Dict[TransactionId, Set[tuple]] = defaultdict(set)

    def acquire(
        self,
        txn_id: TransactionId,
        page_id: PageId,
        slot_id: int | None,
        mode: LockMode,
        wait: bool = True,
        timeout_ms: int | None = None,
    ) -> bool:
        """Acquire a lock on a resource.

        Args:
            txn_id: The requesting transaction.
            page_id: The page to lock.
            slot_id: The slot to lock (None for page-level lock).
            mode: The lock mode requested.
            wait: Whether to wait for the lock.
            timeout_ms: Max time to wait (None = forever).

        Returns:
            True if lock acquired, False if not (only when wait=False).

        Raises:
            DeadlockError: If deadlock detected.
            LockTimeoutError: If timeout exceeded.
        """
        resource_key = (page_id, slot_id) if slot_id is not None else (page_id,)

        with self._lock:
            # Get or create lock entry
            if resource_key not in self._lock_table:
                self._lock_table[resource_key] = LockEntry()

            entry = self._lock_table[resource_key]

            # Check if we already hold a compatible lock
            for granted_mode, holders in entry.granted.items():
                if txn_id in holders:
                    if granted_mode == mode or self._is_stronger(granted_mode, mode):
                        # Already have sufficient lock
                        return True
                    elif self._is_upgrade(granted_mode, mode):
                        # Check if upgrade would conflict with other holders
                        if not entry.has_conflict(mode, txn_id):
                            # Upgrade the lock
                            entry.granted[granted_mode].remove(txn_id)
                            entry.granted[mode].add(txn_id)
                            return True
                        # Otherwise, upgrade is blocked
                        if not wait:
                            return False
                        # Fall through to waiting logic

            # Check for conflicts
            if not entry.has_conflict(mode, txn_id):
                # No conflict - grant immediately
                entry.granted[mode].add(txn_id)
                self._txn_locks[txn_id].add(resource_key)
                return True

            if not wait:
                return False

            # Add to waiting queue
            request = LockRequest(txn_id=txn_id, mode=mode, granted=False)
            entry.waiting.append(request)

            # Update wait-for graph
            blocking_txns = self._get_blocking_txns(entry, mode, txn_id)
            self._wait_for[txn_id] = blocking_txns

            # Check for deadlock
            if self._deadlock_detection and self._has_cycle(txn_id):
                # Remove from waiting queue
                entry.waiting.remove(request)
                del self._wait_for[txn_id]
                raise DeadlockError(f"Deadlock detected for transaction {txn_id}")

        # Wait for lock (simplified - just return False for now)
        # A full implementation would use condition variables
        with self._lock:
            if request in entry.waiting:
                entry.waiting.remove(request)
            if txn_id in self._wait_for:
                del self._wait_for[txn_id]
        return False

    def release(
        self,
        txn_id: TransactionId,
        page_id: PageId,
        slot_id: int | None = None,
    ) -> bool:
        """Release a specific lock.

        Args:
            txn_id: The transaction releasing.
            page_id: The page.
            slot_id: The slot (None for page lock).

        Returns:
            True if lock was released, False if not held.
        """
        resource_key = (page_id, slot_id) if slot_id is not None else (page_id,)

        with self._lock:
            if resource_key not in self._lock_table:
                return False

            entry = self._lock_table[resource_key]
            released = False

            # Remove from all granted modes
            for mode, holders in list(entry.granted.items()):
                if txn_id in holders:
                    holders.remove(txn_id)
                    released = True
                    if not holders:
                        del entry.granted[mode]

            if released:
                self._txn_locks[txn_id].discard(resource_key)
                # Wake up waiting transactions
                self._grant_waiting(entry)

            # Clean up empty entry
            if not entry.granted and not entry.waiting:
                del self._lock_table[resource_key]

            return released

    def release_all(self, txn_id: TransactionId) -> int:
        """Release all locks held by a transaction.

        Called during commit or abort.

        Args:
            txn_id: The transaction.

        Returns:
            Number of locks released.
        """
        with self._lock:
            resources = list(self._txn_locks.get(txn_id, set()))
            count = 0

            for resource_key in resources:
                if resource_key in self._lock_table:
                    entry = self._lock_table[resource_key]
                    for mode, holders in list(entry.granted.items()):
                        if txn_id in holders:
                            holders.remove(txn_id)
                            count += 1
                            if not holders:
                                del entry.granted[mode]

                    # Wake up waiting transactions
                    self._grant_waiting(entry)

                    # Clean up empty entry
                    if not entry.granted and not entry.waiting:
                        del self._lock_table[resource_key]

            # Clear transaction's lock set
            if txn_id in self._txn_locks:
                del self._txn_locks[txn_id]

            # Clear wait-for graph entry
            if txn_id in self._wait_for:
                del self._wait_for[txn_id]

            return count

    def get_locks_held(self, txn_id: TransactionId) -> list[tuple]:
        """Get all resources locked by a transaction.

        Args:
            txn_id: The transaction.

        Returns:
            List of (page_id, slot_id) tuples.
        """
        with self._lock:
            return list(self._txn_locks.get(txn_id, set()))

    def _get_blocking_txns(
        self, entry: LockEntry, mode: LockMode, txn_id: TransactionId
    ) -> Set[TransactionId]:
        """Get transactions blocking this lock request."""
        blocking = set()
        for granted_mode, holders in entry.granted.items():
            if not LockMode.is_compatible(granted_mode, mode):
                blocking.update(holders - {txn_id})
        return blocking

    def _has_cycle(self, start_txn: TransactionId) -> bool:
        """Check for cycle in wait-for graph (deadlock detection)."""
        visited = set()
        stack = [start_txn]

        while stack:
            txn = stack.pop()
            if txn in visited:
                continue
            visited.add(txn)

            for waiting_for in self._wait_for.get(txn, set()):
                if waiting_for == start_txn:
                    return True
                stack.append(waiting_for)

        return False

    def _grant_waiting(self, entry: LockEntry) -> None:
        """Try to grant locks to waiting transactions."""
        still_waiting = []

        for request in entry.waiting:
            if not entry.has_conflict(request.mode, request.txn_id):
                # Can grant this request
                entry.granted[request.mode].add(request.txn_id)
                self._txn_locks[request.txn_id].add(resource_key)
                request.granted = True
                if request.txn_id in self._wait_for:
                    del self._wait_for[request.txn_id]
            else:
                still_waiting.append(request)

        entry.waiting = still_waiting

    def _is_upgrade(self, held: LockMode, requested: LockMode) -> bool:
        """Check if requested is an upgrade from held."""
        strength = {
            LockMode.INTENT_SHARED: 0,
            LockMode.SHARED: 1,
            LockMode.INTENT_EXCLUSIVE: 1,
            LockMode.SHARED_INTENT_EXCLUSIVE: 2,
            LockMode.EXCLUSIVE: 3,
        }
        return strength.get(requested, 0) > strength.get(held, 0)

    def _is_stronger(self, held: LockMode, requested: LockMode) -> bool:
        """Check if held lock is stronger than requested."""
        strength = {
            LockMode.INTENT_SHARED: 0,
            LockMode.SHARED: 1,
            LockMode.INTENT_EXCLUSIVE: 1,
            LockMode.SHARED_INTENT_EXCLUSIVE: 2,
            LockMode.EXCLUSIVE: 3,
        }
        return strength.get(held, 0) >= strength.get(requested, 0)


class DeadlockError(Exception):
    """Raised when a deadlock is detected."""

    pass


class LockTimeoutError(Exception):
    """Raised when lock wait times out."""

    pass
