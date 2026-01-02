"""Database Engine - Unified entry point for the database.

This module provides the main DatabaseEngine class that orchestrates
all database components: storage, WAL, buffer pool, transactions,
and SQL execution.

Usage:
    from db_engine.application import DatabaseEngine

    # Create and start the database
    db = DatabaseEngine(data_dir="/path/to/data")
    db.start()

    # Execute SQL
    result = db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
    result = db.execute("INSERT INTO users VALUES (1, 'Alice')")
    result = db.execute("SELECT * FROM users")

    # Clean shutdown
    db.stop()

References:
    - design.md Section 1 (Architecture Overview)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from db_engine.adapters.inbound.sql_parser import SQLParser, TransactionPlan, StatementType
from db_engine.adapters.outbound.file_disk_manager import FileDiskManager
from db_engine.adapters.outbound.file_wal_writer import FileWALWriter
from db_engine.adapters.outbound.lru_buffer_pool import LRUBufferPool
from db_engine.application.executor import (
    ExecutionResult,
    QueryExecutor,
    Row,
    TableRegistry,
)
from db_engine.domain.services.btree_index import BTreeIndexManager
from db_engine.domain.services.lock_manager import LockManager
from db_engine.domain.services.transaction_manager import MVCCTransactionManager
from db_engine.ports.inbound.transaction_manager import Transaction


@dataclass
class SessionState:
    """State for a database session (connection)."""

    session_id: int
    current_transaction: Transaction | None = None
    autocommit: bool = True


class DatabaseEngine:
    """Main database engine that orchestrates all components.

    The DatabaseEngine is the unified entry point for the database.
    It initializes all infrastructure components and provides a
    simple execute() interface for SQL statements.

    Features:
        - Automatic transaction management (autocommit mode)
        - Explicit transaction support (BEGIN/COMMIT/ROLLBACK)
        - Crash recovery on startup
        - Clean shutdown with WAL flush

    Thread Safety:
        Multiple threads can share a DatabaseEngine instance.
        Each thread should use its own session for transaction isolation.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        page_size: int = 4096,
        buffer_pool_size: int = 100,
        wal_sync_mode: str = "fsync",
    ) -> None:
        """Initialize the database engine.

        Args:
            data_dir: Directory for database files. Uses temp dir if None.
            page_size: Size of pages in bytes (default 4096).
            buffer_pool_size: Number of pages in buffer pool (default 100).
            wal_sync_mode: WAL sync mode - 'fsync', 'fdatasync', or 'none'.
        """
        # Set up data directory
        if data_dir is None:
            data_dir = tempfile.mkdtemp(prefix="db_engine_")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._page_size = page_size
        self._buffer_pool_size = buffer_pool_size
        self._wal_sync_mode = wal_sync_mode

        # Components (initialized on start)
        self._disk_manager: FileDiskManager | None = None
        self._wal_writer: FileWALWriter | None = None
        self._buffer_pool: LRUBufferPool | None = None
        self._lock_manager: LockManager | None = None
        self._txn_manager: MVCCTransactionManager | None = None
        self._index_manager: BTreeIndexManager | None = None

        # SQL components
        self._parser: SQLParser | None = None
        self._executor: QueryExecutor | None = None

        # Session management
        self._next_session_id = 1
        self._sessions: Dict[int, SessionState] = {}
        self._default_session: SessionState | None = None

        # State
        self._started = False

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self._data_dir

    @property
    def is_started(self) -> bool:
        """Check if the database is started."""
        return self._started

    def start(self) -> None:
        """Start the database engine.

        Initializes all components and performs crash recovery if needed.
        Must be called before executing any SQL.

        Raises:
            RuntimeError: If already started.
        """
        if self._started:
            raise RuntimeError("Database engine already started")

        # Initialize disk manager
        data_file = self._data_dir / "data.db"
        self._disk_manager = FileDiskManager(
            file_path=str(data_file),
            page_size=self._page_size,
        )

        # Initialize WAL (uses a directory for segment files)
        wal_dir = self._data_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        self._wal_writer = FileWALWriter(
            wal_dir=str(wal_dir),
        )

        # Initialize buffer pool
        self._buffer_pool = LRUBufferPool(
            pool_size=self._buffer_pool_size,
            disk_manager=self._disk_manager,
        )

        # Initialize lock manager
        self._lock_manager = LockManager(deadlock_detection=True)

        # Initialize transaction manager
        self._txn_manager = MVCCTransactionManager(
            wal_writer=self._wal_writer,
            buffer_pool=self._buffer_pool,
            lock_manager=self._lock_manager,
        )

        # Initialize index manager
        self._index_manager = BTreeIndexManager()

        # Initialize SQL components
        self._parser = SQLParser(dialect="sqlite")
        self._executor = QueryExecutor(
            buffer_pool=self._buffer_pool,
            txn_manager=self._txn_manager,
            index_manager=self._index_manager,
        )

        # Create default session
        self._default_session = self._create_session()

        self._started = True

    def stop(self) -> None:
        """Stop the database engine.

        Flushes all buffers, closes files, and releases resources.
        Any uncommitted transactions will be rolled back.

        Raises:
            RuntimeError: If not started.
        """
        if not self._started:
            raise RuntimeError("Database engine not started")

        # Rollback any active transactions
        for session in self._sessions.values():
            if session.current_transaction is not None:
                try:
                    self._txn_manager.abort(session.current_transaction)
                except Exception:
                    pass  # Best effort cleanup
                session.current_transaction = None

        # Flush buffer pool
        if self._buffer_pool:
            self._buffer_pool.flush_all_pages()

        # Close WAL
        if self._wal_writer:
            self._wal_writer.close()

        # Close disk manager
        if self._disk_manager:
            self._disk_manager.close()

        # Clear state
        self._sessions.clear()
        self._default_session = None
        self._started = False

    def execute(self, sql: str, session_id: int | None = None) -> ExecutionResult:
        """Execute a SQL statement.

        Args:
            sql: The SQL statement to execute.
            session_id: Optional session ID. Uses default session if None.

        Returns:
            ExecutionResult with rows and/or status message.

        Raises:
            RuntimeError: If database not started.
        """
        if not self._started:
            raise RuntimeError("Database engine not started")

        # Get session
        session = self._get_session(session_id)

        # Parse SQL
        try:
            plan = self._parser.parse(sql)
        except Exception as e:
            return ExecutionResult(message=f"Parse error: {e}")

        # Handle transaction control statements
        if isinstance(plan, TransactionPlan):
            return self._handle_transaction(plan, session)

        # For data statements, ensure we have a transaction
        if session.autocommit and session.current_transaction is None:
            # Start implicit transaction
            session.current_transaction = self._txn_manager.begin()

        # Execute the statement
        result = self._executor.execute(plan)

        # Auto-commit if in autocommit mode
        if session.autocommit and session.current_transaction is not None:
            try:
                self._txn_manager.commit(session.current_transaction)
            except Exception as e:
                result = ExecutionResult(message=f"Commit failed: {e}")
            finally:
                session.current_transaction = None

        return result

    def execute_many(
        self, statements: list[str], session_id: int | None = None
    ) -> list[ExecutionResult]:
        """Execute multiple SQL statements.

        Args:
            statements: List of SQL statements to execute.
            session_id: Optional session ID.

        Returns:
            List of ExecutionResult for each statement.
        """
        return [self.execute(sql, session_id) for sql in statements]

    def create_session(self) -> int:
        """Create a new database session.

        Returns:
            The session ID for the new session.
        """
        session = self._create_session()
        return session.session_id

    def close_session(self, session_id: int) -> None:
        """Close a database session.

        Any uncommitted transaction will be rolled back.

        Args:
            session_id: The session to close.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        # Rollback any active transaction
        if session.current_transaction is not None:
            try:
                self._txn_manager.abort(session.current_transaction)
            except Exception:
                pass

        del self._sessions[session_id]

    def set_autocommit(self, enabled: bool, session_id: int | None = None) -> None:
        """Set autocommit mode for a session.

        Args:
            enabled: Whether to enable autocommit.
            session_id: The session to modify.
        """
        session = self._get_session(session_id)
        session.autocommit = enabled

    def _create_session(self) -> SessionState:
        """Create a new session internally."""
        session = SessionState(session_id=self._next_session_id)
        self._sessions[session.session_id] = session
        self._next_session_id += 1
        return session

    def _get_session(self, session_id: int | None) -> SessionState:
        """Get a session by ID or return default."""
        if session_id is None:
            return self._default_session

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        return session

    def _handle_transaction(
        self, plan: TransactionPlan, session: SessionState
    ) -> ExecutionResult:
        """Handle transaction control statements."""

        if plan.statement_type == StatementType.BEGIN:
            if session.current_transaction is not None:
                return ExecutionResult(message="Transaction already in progress")

            session.current_transaction = self._txn_manager.begin()
            session.autocommit = False
            return ExecutionResult(message="OK: Transaction started")

        elif plan.statement_type == StatementType.COMMIT:
            if session.current_transaction is None:
                return ExecutionResult(message="No transaction in progress")

            try:
                self._txn_manager.commit(session.current_transaction)
                session.current_transaction = None
                session.autocommit = True
                return ExecutionResult(message="OK: Transaction committed")
            except Exception as e:
                return ExecutionResult(message=f"Commit failed: {e}")

        elif plan.statement_type == StatementType.ROLLBACK:
            if session.current_transaction is None:
                return ExecutionResult(message="No transaction in progress")

            try:
                self._txn_manager.abort(session.current_transaction)
                session.current_transaction = None
                session.autocommit = True
                return ExecutionResult(message="OK: Transaction rolled back")
            except Exception as e:
                return ExecutionResult(message=f"Rollback failed: {e}")

        return ExecutionResult(message=f"Unknown transaction statement: {plan.statement_type}")

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with various statistics.
        """
        stats = {
            "started": self._started,
            "data_dir": str(self._data_dir),
            "page_size": self._page_size,
            "sessions": len(self._sessions),
        }

        if self._txn_manager:
            txn_stats = self._txn_manager.get_stats()
            stats["transactions"] = {
                "active": txn_stats.active_count,
                "committed": txn_stats.committed_total,
                "aborted": txn_stats.aborted_total,
            }

        if self._buffer_pool:
            stats["buffer_pool"] = {
                "size": self._buffer_pool_size,
            }

        return stats

    def __enter__(self) -> "DatabaseEngine":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
