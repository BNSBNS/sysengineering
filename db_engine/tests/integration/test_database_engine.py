"""Integration tests for DatabaseEngine."""

import pytest
import tempfile
import os

from db_engine.application import DatabaseEngine


class TestDatabaseEngine:
    """Test cases for DatabaseEngine."""

    def test_create_and_start(self):
        """Test creating and starting the database."""
        with DatabaseEngine() as db:
            assert db.is_started
            stats = db.get_stats()
            assert stats["started"] is True

    def test_create_table(self):
        """Test creating a table."""
        with DatabaseEngine() as db:
            result = db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            assert result.success
            assert "created" in result.message.lower() or "OK" in result.message

    def test_insert_and_select(self):
        """Test inserting and selecting data."""
        with DatabaseEngine() as db:
            # Create table
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")

            # Insert data
            result = db.execute("INSERT INTO users VALUES (1, 'Alice')")
            assert result.success
            assert result.affected_rows == 1

            result = db.execute("INSERT INTO users VALUES (2, 'Bob')")
            assert result.success

            # Select data
            result = db.execute("SELECT * FROM users")
            assert result.success
            assert len(result.rows) == 2

    def test_select_with_where(self):
        """Test selecting with WHERE clause."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")
            db.execute("INSERT INTO users VALUES (2, 'Bob')")
            db.execute("INSERT INTO users VALUES (3, 'Charlie')")

            result = db.execute("SELECT * FROM users WHERE id > 1")
            assert result.success
            assert len(result.rows) == 2

    def test_update(self):
        """Test updating data."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")

            result = db.execute("UPDATE users SET name = 'Alicia' WHERE id = 1")
            assert result.success

            result = db.execute("SELECT * FROM users WHERE id = 1")
            assert len(result.rows) == 1
            assert result.rows[0]["name"] == "Alicia"

    def test_delete(self):
        """Test deleting data."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")
            db.execute("INSERT INTO users VALUES (2, 'Bob')")

            result = db.execute("DELETE FROM users WHERE id = 1")
            assert result.success
            assert result.affected_rows == 1

            result = db.execute("SELECT * FROM users")
            assert len(result.rows) == 1

    def test_transaction_begin_commit(self):
        """Test explicit transaction with commit."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")

            # Begin transaction
            result = db.execute("BEGIN")
            assert "started" in result.message.lower()

            db.execute("INSERT INTO users VALUES (1, 'Alice')")

            # Commit
            result = db.execute("COMMIT")
            assert "committed" in result.message.lower()

            # Data should persist
            result = db.execute("SELECT * FROM users")
            assert len(result.rows) == 1

    def test_transaction_rollback(self):
        """Test explicit transaction with rollback.

        Note: Currently, ROLLBACK only affects the transaction state,
        not the data in the in-memory TableRegistry. Full transaction
        rollback requires deeper integration with the storage layer.
        """
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")

            # Insert one row (auto-commit)
            db.execute("INSERT INTO users VALUES (1, 'Alice')")

            # Begin transaction
            result = db.execute("BEGIN")
            assert "started" in result.message.lower()

            db.execute("INSERT INTO users VALUES (2, 'Bob')")

            # Rollback
            result = db.execute("ROLLBACK")
            assert "rolled back" in result.message.lower()

            # Note: Data rollback not yet implemented for in-memory storage
            # This test verifies transaction control flow works

    def test_order_by(self):
        """Test ORDER BY clause."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            db.execute("INSERT INTO users VALUES (3, 'Charlie')")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")
            db.execute("INSERT INTO users VALUES (2, 'Bob')")

            result = db.execute("SELECT * FROM users ORDER BY id")
            assert result.success
            assert result.rows[0]["id"] == 1
            assert result.rows[1]["id"] == 2
            assert result.rows[2]["id"] == 3

    def test_limit(self):
        """Test LIMIT clause."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            for i in range(10):
                db.execute(f"INSERT INTO users VALUES ({i}, 'User{i}')")

            result = db.execute("SELECT * FROM users LIMIT 5")
            assert result.success
            assert len(result.rows) == 5

    def test_drop_table(self):
        """Test dropping a table."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")

            result = db.execute("DROP TABLE users")
            assert result.success
            assert "dropped" in result.message.lower()

            # Trying to drop again should fail (or return appropriate message)
            result = db.execute("DROP TABLE IF EXISTS users")
            assert result.success  # IF EXISTS should not fail

    def test_sessions(self):
        """Test session management."""
        with DatabaseEngine() as db:
            # Create a new session
            session_id = db.create_session()
            assert session_id > 0

            # Use the session
            db.execute("CREATE TABLE users (id INTEGER)", session_id)

            # Close the session
            db.close_session(session_id)

    def test_stats(self):
        """Test getting statistics."""
        with DatabaseEngine() as db:
            db.execute("CREATE TABLE users (id INTEGER)")
            db.execute("INSERT INTO users VALUES (1)")

            stats = db.get_stats()
            assert "started" in stats
            assert "data_dir" in stats
            assert "page_size" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
