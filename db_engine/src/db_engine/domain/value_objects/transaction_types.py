"""Transaction-related types and enumerations.

These types define the lifecycle states and isolation levels for database
transactions, following the state machine defined in docs/design.md.
"""

from __future__ import annotations

from enum import Enum, auto


class TransactionState(Enum):
    """Transaction lifecycle states.

    State machine (per design.md Section 4):

        IDLE ──begin()──> ACTIVE
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         commit()        abort()          │
              │             │             │
              v             v             │
         COMMITTING    ABORTING           │
              │             │             │
              v             v             │
         COMMITTED      ABORTED ──────────┘
                        (returns to IDLE)

    The distinction between COMMITTING/ABORTING and COMMITTED/ABORTED
    is important for recovery - if we crash during COMMITTING, we may
    need to complete the commit on recovery.
    """

    IDLE = auto()
    """Transaction has not started yet."""

    ACTIVE = auto()
    """Transaction is running and can execute operations."""

    COMMITTING = auto()
    """Transaction is in the process of committing (WAL flush in progress)."""

    ABORTING = auto()
    """Transaction is in the process of aborting (undo in progress)."""

    COMMITTED = auto()
    """Transaction has successfully committed. All changes are durable."""

    ABORTED = auto()
    """Transaction has been aborted. All changes have been rolled back."""

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (COMMITTED or ABORTED)."""
        return self in (TransactionState.COMMITTED, TransactionState.ABORTED)

    def is_active(self) -> bool:
        """Check if transaction can still perform operations."""
        return self == TransactionState.ACTIVE

    def can_commit(self) -> bool:
        """Check if transaction can begin commit process."""
        return self == TransactionState.ACTIVE

    def can_abort(self) -> bool:
        """Check if transaction can be aborted."""
        return self in (TransactionState.ACTIVE, TransactionState.COMMITTING)


class IsolationLevel(Enum):
    """Transaction isolation levels.

    Our default is SNAPSHOT isolation (also known as REPEATABLE READ in some
    databases). This provides a good balance between consistency and performance
    by allowing readers to never block writers.

    Isolation levels from weakest to strongest:
    - READ_UNCOMMITTED: Can see uncommitted changes (dirty reads)
    - READ_COMMITTED: Only sees committed changes, but may see different
                      snapshots for different queries
    - REPEATABLE_READ: Same snapshot for entire transaction, but phantom
                       reads possible in some implementations
    - SNAPSHOT: Same snapshot for entire transaction (our default)
    - SERIALIZABLE: Full isolation, transactions appear to execute serially

    References:
        - design.md Section 5 (Concurrency Model)
        - Reed, D. "Naming and Synchronization in a Decentralized Computer System" (1978)
    """

    READ_UNCOMMITTED = auto()
    """Allows dirty reads. Not recommended for most use cases."""

    READ_COMMITTED = auto()
    """Each query sees only committed data, but different queries may see different snapshots."""

    REPEATABLE_READ = auto()
    """Same snapshot for entire transaction. May allow phantom reads in some implementations."""

    SNAPSHOT = auto()
    """MVCC-based snapshot isolation. Our default. Readers never block writers."""

    SERIALIZABLE = auto()
    """Full serializability. Highest isolation but may have more aborts due to conflicts."""


class LockMode(Enum):
    """Lock modes for two-phase locking (2PL).

    Lock compatibility matrix:

              | S | X | IS | IX | SIX |
        ------|---|---|----|----|-----|
        S     | Y | N | Y  | N  | N   |
        X     | N | N | N  | N  | N   |
        IS    | Y | N | Y  | Y  | Y   |
        IX    | N | N | Y  | Y  | N   |
        SIX   | N | N | Y  | N  | N   |

    Where Y = compatible, N = not compatible

    References:
        - design.md Section 5 (Lock Hierarchy)
        - Gray & Reuter "Transaction Processing" (1993)
    """

    SHARED = auto()
    """Shared lock (S) - allows concurrent reads, blocks writes."""

    EXCLUSIVE = auto()
    """Exclusive lock (X) - blocks all other access."""

    INTENT_SHARED = auto()
    """Intent shared (IS) - indicates intention to acquire S lock on descendant."""

    INTENT_EXCLUSIVE = auto()
    """Intent exclusive (IX) - indicates intention to acquire X lock on descendant."""

    SHARED_INTENT_EXCLUSIVE = auto()
    """Shared + intent exclusive (SIX) - S lock on current, IX on descendants."""

    def is_compatible(self, other: LockMode) -> bool:
        """Check if this lock mode is compatible with another.

        Args:
            other: The other lock mode to check compatibility with

        Returns:
            True if the locks are compatible (can be held simultaneously)
        """
        compatibility = {
            LockMode.SHARED: {LockMode.SHARED, LockMode.INTENT_SHARED},
            LockMode.EXCLUSIVE: set(),
            LockMode.INTENT_SHARED: {
                LockMode.SHARED, LockMode.INTENT_SHARED,
                LockMode.INTENT_EXCLUSIVE, LockMode.SHARED_INTENT_EXCLUSIVE
            },
            LockMode.INTENT_EXCLUSIVE: {
                LockMode.INTENT_SHARED, LockMode.INTENT_EXCLUSIVE
            },
            LockMode.SHARED_INTENT_EXCLUSIVE: {LockMode.INTENT_SHARED},
        }
        return other in compatibility.get(self, set())


class WaitPolicy(Enum):
    """Policy for handling lock conflicts.

    Determines behavior when a lock request cannot be immediately granted.
    """

    WAIT = auto()
    """Block until lock is available."""

    NO_WAIT = auto()
    """Immediately fail if lock is not available."""

    SKIP_LOCKED = auto()
    """Skip the locked row/page (useful for certain scan patterns)."""
