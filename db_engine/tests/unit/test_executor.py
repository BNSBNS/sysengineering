"""Unit tests for Query Executor."""

from __future__ import annotations

import pytest

from db_engine.adapters.inbound import SQLParser
from db_engine.application import ExecutionResult, QueryExecutor, Row


class TestQueryExecutor:
    """Tests for QueryExecutor."""

    @pytest.fixture
    def executor(self) -> QueryExecutor:
        """Create a query executor for testing."""
        return QueryExecutor()

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_create_table(self, executor: QueryExecutor, parser: SQLParser) -> None:
        """Execute CREATE TABLE."""
        plan = parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR(100))")
        result = executor.execute(plan)

        assert result.success
        assert "created" in result.message.lower()

    def test_create_table_duplicate(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Create duplicate table fails."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER)"))
        result = executor.execute(parser.parse("CREATE TABLE users (id INTEGER)"))

        assert not result.success
        assert "exists" in result.message.lower()

    def test_drop_table(self, executor: QueryExecutor, parser: SQLParser) -> None:
        """Execute DROP TABLE."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER)"))
        result = executor.execute(parser.parse("DROP TABLE users"))

        assert result.success
        assert "dropped" in result.message.lower()

    def test_drop_nonexistent_table(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Drop nonexistent table fails."""
        result = executor.execute(parser.parse("DROP TABLE nonexistent"))

        assert not result.success
        assert "not exist" in result.message.lower()

    def test_insert_into_table(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute INSERT."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        result = executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        )

        assert result.success
        assert result.affected_rows == 1

    def test_insert_multiple_rows(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute INSERT with multiple rows."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        result = executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )

        assert result.success
        assert result.affected_rows == 2

    def test_select_from_table(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(parser.parse("SELECT id, name FROM users"))

        assert result.success
        assert len(result.rows) == 2
        assert result.rows[0]["id"] == 1
        assert result.rows[0]["name"] == "Alice"
        assert result.rows[1]["id"] == 2
        assert result.rows[1]["name"] == "Bob"

    def test_select_star(self, executor: QueryExecutor, parser: SQLParser) -> None:
        """Execute SELECT *."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice')"))
        result = executor.execute(parser.parse("SELECT * FROM users"))

        assert result.success
        assert len(result.rows) == 1
        assert result.rows[0]["id"] == 1
        assert result.rows[0]["name"] == "Alice"

    def test_select_with_where(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with WHERE clause."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse(
                "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')"
            )
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users WHERE id > 1")
        )

        assert result.success
        assert len(result.rows) == 2
        assert result.rows[0]["id"] == 2
        assert result.rows[1]["id"] == 3

    def test_select_with_equality(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with equality WHERE clause."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users WHERE name = 'Bob'")
        )

        assert result.success
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Bob"

    def test_select_with_and_condition(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with AND in WHERE clause."""
        executor.execute(
            parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR, age INTEGER)")
        )
        executor.execute(
            parser.parse(
                "INSERT INTO users (id, name, age) VALUES "
                "(1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 25)"
            )
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users WHERE age = 25 AND id > 1")
        )

        assert result.success
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Charlie"

    def test_select_with_order_by(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with ORDER BY."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (2, 'Bob'), (1, 'Alice')")
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users ORDER BY id ASC")
        )

        assert result.success
        assert len(result.rows) == 2
        assert result.rows[0]["id"] == 1
        assert result.rows[1]["id"] == 2

    def test_select_with_order_by_desc(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with ORDER BY DESC."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users ORDER BY id DESC")
        )

        assert result.success
        assert result.rows[0]["id"] == 2
        assert result.rows[1]["id"] == 1

    def test_select_with_limit(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with LIMIT."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse(
                "INSERT INTO users (id, name) VALUES "
                "(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')"
            )
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users LIMIT 2")
        )

        assert result.success
        assert len(result.rows) == 2

    def test_select_with_limit_offset(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute SELECT with LIMIT and OFFSET."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse(
                "INSERT INTO users (id, name) VALUES "
                "(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')"
            )
        )
        result = executor.execute(
            parser.parse("SELECT id, name FROM users LIMIT 2 OFFSET 1")
        )

        assert result.success
        assert len(result.rows) == 2
        assert result.rows[0]["id"] == 2
        assert result.rows[1]["id"] == 3

    def test_update_rows(self, executor: QueryExecutor, parser: SQLParser) -> None:
        """Execute UPDATE."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(
            parser.parse("UPDATE users SET name = 'Updated' WHERE id = 1")
        )

        assert result.success
        assert result.affected_rows == 1

        # Verify update
        result = executor.execute(
            parser.parse("SELECT name FROM users WHERE id = 1")
        )
        assert result.rows[0]["name"] == "Updated"

    def test_update_all_rows(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute UPDATE without WHERE."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(parser.parse("UPDATE users SET name = 'Updated'"))

        assert result.success
        assert result.affected_rows == 2

    def test_delete_rows(self, executor: QueryExecutor, parser: SQLParser) -> None:
        """Execute DELETE."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(parser.parse("DELETE FROM users WHERE id = 1"))

        assert result.success
        assert result.affected_rows == 1

        # Verify delete
        result = executor.execute(parser.parse("SELECT * FROM users"))
        assert len(result.rows) == 1
        assert result.rows[0]["id"] == 2

    def test_delete_all_rows(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute DELETE without WHERE."""
        executor.execute(parser.parse("CREATE TABLE users (id INTEGER, name VARCHAR)"))
        executor.execute(
            parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        )
        result = executor.execute(parser.parse("DELETE FROM users"))

        assert result.success
        assert result.affected_rows == 2

        # Verify delete
        result = executor.execute(parser.parse("SELECT * FROM users"))
        assert len(result.rows) == 0

    def test_transaction_begin(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute BEGIN TRANSACTION."""
        result = executor.execute(parser.parse("BEGIN TRANSACTION"))

        assert result.success
        assert "started" in result.message.lower()

    def test_transaction_commit(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute COMMIT."""
        result = executor.execute(parser.parse("COMMIT"))

        assert result.success
        assert "committed" in result.message.lower()

    def test_transaction_rollback(
        self, executor: QueryExecutor, parser: SQLParser
    ) -> None:
        """Execute ROLLBACK."""
        result = executor.execute(parser.parse("ROLLBACK"))

        assert result.success
        assert "rolled back" in result.message.lower()


class TestRow:
    """Tests for Row class."""

    def test_row_access_by_name(self) -> None:
        """Row can be accessed by column name."""
        row = Row(columns=["id", "name"], values=[1, "Alice"])

        assert row["id"] == 1
        assert row["name"] == "Alice"

    def test_row_access_by_index(self) -> None:
        """Row can be accessed by index."""
        row = Row(columns=["id", "name"], values=[1, "Alice"])

        assert row[0] == 1
        assert row[1] == "Alice"

    def test_row_get_with_default(self) -> None:
        """Row.get returns default for missing columns."""
        row = Row(columns=["id"], values=[1])

        assert row.get("name", "default") == "default"

    def test_row_repr(self) -> None:
        """Row has readable string representation."""
        row = Row(columns=["id", "name"], values=[1, "Alice"])

        assert "id=1" in repr(row)
        assert "name='Alice'" in repr(row)


class TestExecutionResult:
    """Tests for ExecutionResult class."""

    def test_success_property(self) -> None:
        """ExecutionResult.success returns True for OK messages."""
        result = ExecutionResult(message="OK")
        assert result.success is True

        result = ExecutionResult(message="OK: 1 row inserted")
        assert result.success is True

        result = ExecutionResult(message="Error: something went wrong")
        assert result.success is False
