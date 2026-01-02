"""Unit tests for SQL Parser."""

from __future__ import annotations

import pytest

from db_engine.adapters.inbound import (
    AggregateExpr,
    AggregateFunc,
    ColumnExpr,
    ComparisonExpr,
    ComparisonOp,
    CreateTablePlan,
    DataType,
    DeletePlan,
    Filter,
    InsertPlan,
    Limit,
    LiteralExpr,
    LogicalExpr,
    LogicalOp,
    ParseError,
    Project,
    Sort,
    SQLParser,
    StatementType,
    TableScan,
    TransactionPlan,
    UpdatePlan,
)


class TestSQLParserSelect:
    """Tests for SELECT statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_simple_select(self, parser: SQLParser) -> None:
        """Parse simple SELECT."""
        plan = parser.parse("SELECT id, name FROM users")

        assert isinstance(plan, Project)
        assert len(plan.items) == 2
        assert isinstance(plan.input, TableScan)
        assert plan.input.table_name == "users"

    def test_select_star(self, parser: SQLParser) -> None:
        """Parse SELECT *."""
        plan = parser.parse("SELECT * FROM users")

        assert isinstance(plan, Project)
        assert len(plan.items) == 1
        assert isinstance(plan.items[0].expr, ColumnExpr)
        assert plan.items[0].expr.column.name == "*"

    def test_select_with_where(self, parser: SQLParser) -> None:
        """Parse SELECT with WHERE clause."""
        plan = parser.parse("SELECT id FROM users WHERE age > 18")

        assert isinstance(plan, Project)
        assert isinstance(plan.input, Filter)
        assert isinstance(plan.input.predicate, ComparisonExpr)
        assert plan.input.predicate.op == ComparisonOp.GT

    def test_select_with_multiple_conditions(self, parser: SQLParser) -> None:
        """Parse SELECT with multiple WHERE conditions."""
        plan = parser.parse("SELECT id FROM users WHERE age > 18 AND name = 'John'")

        assert isinstance(plan, Project)
        assert isinstance(plan.input, Filter)
        assert isinstance(plan.input.predicate, LogicalExpr)
        assert plan.input.predicate.op == LogicalOp.AND
        assert len(plan.input.predicate.operands) == 2

    def test_select_with_or_condition(self, parser: SQLParser) -> None:
        """Parse SELECT with OR condition."""
        plan = parser.parse("SELECT id FROM users WHERE age > 18 OR active = 1")

        assert isinstance(plan, Project)
        assert isinstance(plan.input, Filter)
        assert isinstance(plan.input.predicate, LogicalExpr)
        assert plan.input.predicate.op == LogicalOp.OR

    def test_select_with_order_by(self, parser: SQLParser) -> None:
        """Parse SELECT with ORDER BY."""
        plan = parser.parse("SELECT id, name FROM users ORDER BY name ASC")

        assert isinstance(plan, Sort)
        assert len(plan.order_by) == 1
        assert plan.order_by[0].ascending is True

    def test_select_with_order_by_desc(self, parser: SQLParser) -> None:
        """Parse SELECT with ORDER BY DESC."""
        plan = parser.parse("SELECT id FROM users ORDER BY created_at DESC")

        assert isinstance(plan, Sort)
        assert plan.order_by[0].ascending is False

    def test_select_with_limit(self, parser: SQLParser) -> None:
        """Parse SELECT with LIMIT."""
        plan = parser.parse("SELECT id FROM users LIMIT 10")

        assert isinstance(plan, Limit)
        assert plan.count == 10

    def test_select_with_limit_offset(self, parser: SQLParser) -> None:
        """Parse SELECT with LIMIT and OFFSET."""
        plan = parser.parse("SELECT id FROM users LIMIT 10 OFFSET 20")

        assert isinstance(plan, Limit)
        assert plan.count == 10
        assert plan.offset == 20

    def test_select_with_aggregate_count(self, parser: SQLParser) -> None:
        """Parse SELECT with COUNT."""
        plan = parser.parse("SELECT COUNT(*) FROM users")

        assert isinstance(plan, Project)
        assert len(plan.items) == 1
        assert isinstance(plan.items[0].expr, AggregateExpr)
        assert plan.items[0].expr.func == AggregateFunc.COUNT
        assert plan.items[0].expr.arg is None  # COUNT(*)

    def test_select_with_aggregate_sum(self, parser: SQLParser) -> None:
        """Parse SELECT with SUM."""
        plan = parser.parse("SELECT SUM(amount) FROM orders")

        assert isinstance(plan, Project)
        assert isinstance(plan.items[0].expr, AggregateExpr)
        assert plan.items[0].expr.func == AggregateFunc.SUM

    def test_select_with_alias(self, parser: SQLParser) -> None:
        """Parse SELECT with column alias."""
        plan = parser.parse("SELECT id AS user_id FROM users")

        assert isinstance(plan, Project)
        assert plan.items[0].alias == "user_id"

    def test_select_comparison_operators(self, parser: SQLParser) -> None:
        """Parse SELECT with various comparison operators."""
        tests = [
            ("SELECT * FROM t WHERE a = 1", ComparisonOp.EQ),
            ("SELECT * FROM t WHERE a <> 1", ComparisonOp.NE),
            ("SELECT * FROM t WHERE a < 1", ComparisonOp.LT),
            ("SELECT * FROM t WHERE a <= 1", ComparisonOp.LE),
            ("SELECT * FROM t WHERE a > 1", ComparisonOp.GT),
            ("SELECT * FROM t WHERE a >= 1", ComparisonOp.GE),
        ]

        for sql, expected_op in tests:
            plan = parser.parse(sql)
            assert isinstance(plan, Project)
            assert isinstance(plan.input, Filter)
            assert isinstance(plan.input.predicate, ComparisonExpr)
            assert plan.input.predicate.op == expected_op

    def test_select_with_like(self, parser: SQLParser) -> None:
        """Parse SELECT with LIKE."""
        plan = parser.parse("SELECT * FROM users WHERE name LIKE '%john%'")

        assert isinstance(plan, Project)
        assert isinstance(plan.input, Filter)
        assert isinstance(plan.input.predicate, ComparisonExpr)
        assert plan.input.predicate.op == ComparisonOp.LIKE


class TestSQLParserInsert:
    """Tests for INSERT statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_simple_insert(self, parser: SQLParser) -> None:
        """Parse simple INSERT."""
        plan = parser.parse("INSERT INTO users (id, name) VALUES (1, 'John')")

        assert isinstance(plan, InsertPlan)
        assert plan.table_name == "users"
        assert plan.columns == ["id", "name"]
        assert len(plan.values) == 1
        assert plan.values[0][0].value == 1
        assert plan.values[0][1].value == "John"

    def test_insert_multiple_rows(self, parser: SQLParser) -> None:
        """Parse INSERT with multiple rows."""
        plan = parser.parse(
            "INSERT INTO users (id, name) VALUES (1, 'John'), (2, 'Jane')"
        )

        assert isinstance(plan, InsertPlan)
        assert len(plan.values) == 2
        assert plan.values[0][0].value == 1
        assert plan.values[1][0].value == 2


class TestSQLParserUpdate:
    """Tests for UPDATE statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_simple_update(self, parser: SQLParser) -> None:
        """Parse simple UPDATE."""
        plan = parser.parse("UPDATE users SET name = 'John' WHERE id = 1")

        assert isinstance(plan, UpdatePlan)
        assert plan.table_name == "users"
        assert "name" in plan.assignments
        assert isinstance(plan.predicate, ComparisonExpr)

    def test_update_without_where(self, parser: SQLParser) -> None:
        """Parse UPDATE without WHERE."""
        plan = parser.parse("UPDATE users SET active = 1")

        assert isinstance(plan, UpdatePlan)
        assert plan.predicate is None


class TestSQLParserDelete:
    """Tests for DELETE statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_simple_delete(self, parser: SQLParser) -> None:
        """Parse simple DELETE."""
        plan = parser.parse("DELETE FROM users WHERE id = 1")

        assert isinstance(plan, DeletePlan)
        assert plan.table_name == "users"
        assert isinstance(plan.predicate, ComparisonExpr)

    def test_delete_without_where(self, parser: SQLParser) -> None:
        """Parse DELETE without WHERE."""
        plan = parser.parse("DELETE FROM users")

        assert isinstance(plan, DeletePlan)
        assert plan.predicate is None


class TestSQLParserDDL:
    """Tests for DDL statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_create_table(self, parser: SQLParser) -> None:
        """Parse CREATE TABLE."""
        plan = parser.parse(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email TEXT
            )
            """
        )

        assert isinstance(plan, CreateTablePlan)
        assert plan.table_name == "users"
        assert len(plan.columns) == 3

        # Check id column
        id_col = plan.columns[0]
        assert id_col.name == "id"
        assert id_col.data_type == DataType.INTEGER
        assert id_col.primary_key is True

        # Check name column
        name_col = plan.columns[1]
        assert name_col.name == "name"
        assert name_col.data_type == DataType.VARCHAR
        assert name_col.nullable is False

        # Check email column
        email_col = plan.columns[2]
        assert email_col.name == "email"
        assert email_col.data_type == DataType.TEXT
        assert email_col.nullable is True

    def test_create_table_data_types(self, parser: SQLParser) -> None:
        """Parse CREATE TABLE with various data types."""
        plan = parser.parse(
            """
            CREATE TABLE test (
                a INTEGER,
                b BIGINT,
                c FLOAT,
                d DOUBLE,
                e VARCHAR(255),
                f TEXT,
                g BOOLEAN,
                h TIMESTAMP
            )
            """
        )

        assert isinstance(plan, CreateTablePlan)
        types = [col.data_type for col in plan.columns]
        assert DataType.INTEGER in types
        assert DataType.BIGINT in types
        assert DataType.FLOAT in types
        assert DataType.DOUBLE in types
        assert DataType.VARCHAR in types
        assert DataType.TEXT in types
        assert DataType.BOOLEAN in types
        assert DataType.TIMESTAMP in types

    def test_drop_table(self, parser: SQLParser) -> None:
        """Parse DROP TABLE."""
        plan = parser.parse("DROP TABLE users")

        from db_engine.adapters.inbound import DropTablePlan

        assert isinstance(plan, DropTablePlan)
        assert plan.table_name == "users"


class TestSQLParserTransaction:
    """Tests for transaction statement parsing."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_begin_transaction(self, parser: SQLParser) -> None:
        """Parse BEGIN TRANSACTION."""
        plan = parser.parse("BEGIN TRANSACTION")

        assert isinstance(plan, TransactionPlan)
        assert plan.statement_type == StatementType.BEGIN

    def test_commit(self, parser: SQLParser) -> None:
        """Parse COMMIT."""
        plan = parser.parse("COMMIT")

        assert isinstance(plan, TransactionPlan)
        assert plan.statement_type == StatementType.COMMIT

    def test_rollback(self, parser: SQLParser) -> None:
        """Parse ROLLBACK."""
        plan = parser.parse("ROLLBACK")

        assert isinstance(plan, TransactionPlan)
        assert plan.statement_type == StatementType.ROLLBACK


class TestSQLParserErrors:
    """Tests for error handling."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_invalid_sql(self, parser: SQLParser) -> None:
        """Invalid SQL raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("NOT VALID SQL AT ALL")

    def test_empty_sql(self, parser: SQLParser) -> None:
        """Empty SQL raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("")

    def test_select_without_from(self, parser: SQLParser) -> None:
        """SELECT without FROM raises ParseError."""
        with pytest.raises(ParseError, match="FROM"):
            parser.parse("SELECT 1")


class TestLogicalPlanStr:
    """Tests for logical plan string representations."""

    @pytest.fixture
    def parser(self) -> SQLParser:
        """Create a SQL parser for testing."""
        return SQLParser()

    def test_table_scan_str(self, parser: SQLParser) -> None:
        """TableScan has readable string representation."""
        plan = parser.parse("SELECT id FROM users")
        # Walk to TableScan
        while hasattr(plan, "input"):
            plan = plan.input
        assert "TableScan" in str(plan)
        assert "users" in str(plan)

    def test_filter_str(self, parser: SQLParser) -> None:
        """Filter has readable string representation."""
        plan = parser.parse("SELECT id FROM users WHERE age > 18")
        assert isinstance(plan, Project)
        assert isinstance(plan.input, Filter)
        plan_str = str(plan.input)
        assert "Filter" in plan_str

    def test_project_str(self, parser: SQLParser) -> None:
        """Project has readable string representation."""
        plan = parser.parse("SELECT id, name FROM users")
        plan_str = str(plan)
        assert "Project" in plan_str
        assert "id" in plan_str
        assert "name" in plan_str

    def test_insert_str(self, parser: SQLParser) -> None:
        """InsertPlan has readable string representation."""
        plan = parser.parse("INSERT INTO users (id) VALUES (1)")
        assert "Insert" in str(plan)
        assert "users" in str(plan)
