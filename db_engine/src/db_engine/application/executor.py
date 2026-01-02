"""Query Executor using Volcano iterator model.

This module implements a query executor that interprets logical plans
and executes them against the database using a pull-based iterator model.

The Volcano model:
    - Each operator is an iterator with open(), next(), close() methods
    - Operators pull tuples from their children on demand
    - Enables pipelining without materializing intermediate results

References:
    - Graefe, "Volcano" (1994)
    - design.md Section 2 (Query Layer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from db_engine.adapters.inbound.sql_parser import (
    AggregateExpr,
    AggregateFunc,
    ColumnExpr,
    ComparisonExpr,
    ComparisonOp,
    CreateTablePlan,
    DeletePlan,
    DropTablePlan,
    Expression,
    Filter,
    InsertPlan,
    Limit,
    LiteralExpr,
    LogicalExpr,
    LogicalOp,
    LogicalPlan,
    Project,
    Sort,
    StatementType,
    TableScan,
    TransactionPlan,
    UpdatePlan,
)
from db_engine.domain.value_objects import PageId, RecordId

if TYPE_CHECKING:
    from db_engine.adapters.inbound.sql_parser import Aggregate
    from db_engine.domain.services import BTreeIndexManager, MVCCTransactionManager
    from db_engine.ports.inbound.buffer_pool import BufferPool


@dataclass
class Row:
    """A row of data returned by the executor.

    Rows are named tuples that can be accessed by column name or index.
    """

    columns: list[str]
    values: list[Any]

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self.values[key]
        try:
            idx = self.columns.index(key)
            return self.values[idx]
        except ValueError as e:
            raise KeyError(f"Column '{key}' not found") from e

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self) -> str:
        pairs = ", ".join(f"{c}={v!r}" for c, v in zip(self.columns, self.values))
        return f"Row({pairs})"


@dataclass
class ExecutionResult:
    """Result of query execution."""

    rows: list[Row] = field(default_factory=list)
    affected_rows: int = 0
    message: str = ""
    columns: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.message == "" or self.message.startswith("OK")


class Operator(ABC):
    """Base class for executor operators (Volcano model)."""

    @abstractmethod
    def open(self) -> None:
        """Initialize the operator."""
        pass

    @abstractmethod
    def next(self) -> Row | None:
        """Return the next row or None if exhausted."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    def __iter__(self) -> Iterator[Row]:
        """Allow iteration over operator results."""
        self.open()
        try:
            while True:
                row = self.next()
                if row is None:
                    break
                yield row
        finally:
            self.close()


class SeqScanOperator(Operator):
    """Sequential scan operator.

    Scans all records in a table by iterating through pages.
    """

    def __init__(
        self,
        table_name: str,
        table_registry: TableRegistry,
        buffer_pool: BufferPool | None = None,
    ) -> None:
        self._table_name = table_name
        self._registry = table_registry
        self._buffer_pool = buffer_pool
        self._current_row = 0
        self._rows: list[Row] = []

    def open(self) -> None:
        """Load rows from the table registry."""
        table = self._registry.get_table(self._table_name)
        if table:
            self._rows = list(table.rows)
        self._current_row = 0

    def next(self) -> Row | None:
        if self._current_row >= len(self._rows):
            return None
        row = self._rows[self._current_row]
        self._current_row += 1
        return row

    def close(self) -> None:
        self._rows = []
        self._current_row = 0


class FilterOperator(Operator):
    """Filter operator that applies a predicate."""

    def __init__(self, child: Operator, predicate: Expression) -> None:
        self._child = child
        self._predicate = predicate

    def open(self) -> None:
        self._child.open()

    def next(self) -> Row | None:
        while True:
            row = self._child.next()
            if row is None:
                return None
            if self._evaluate_predicate(row):
                return row

    def close(self) -> None:
        self._child.close()

    def _evaluate_predicate(self, row: Row) -> bool:
        """Evaluate the predicate against a row."""
        return self._evaluate_expression(self._predicate, row)

    def _evaluate_expression(self, expr: Expression, row: Row) -> Any:
        """Evaluate an expression against a row."""
        if isinstance(expr, LiteralExpr):
            return expr.literal.value
        elif isinstance(expr, ColumnExpr):
            col_name = expr.column.name
            return row.get(col_name)
        elif isinstance(expr, ComparisonExpr):
            left = self._evaluate_expression(expr.left, row)
            if expr.op == ComparisonOp.IS_NULL:
                return left is None
            if expr.op == ComparisonOp.IS_NOT_NULL:
                return left is not None
            if expr.right is None:
                return False
            right = self._evaluate_expression(expr.right, row)
            return self._compare(left, expr.op, right)
        elif isinstance(expr, LogicalExpr):
            if expr.op == LogicalOp.AND:
                return all(self._evaluate_expression(o, row) for o in expr.operands)
            elif expr.op == LogicalOp.OR:
                return any(self._evaluate_expression(o, row) for o in expr.operands)
            elif expr.op == LogicalOp.NOT:
                return not self._evaluate_expression(expr.operands[0], row)
        return False

    def _compare(self, left: Any, op: ComparisonOp, right: Any) -> bool:
        """Compare two values with the given operator."""
        if left is None or right is None:
            return False
        if op == ComparisonOp.EQ:
            return left == right
        elif op == ComparisonOp.NE:
            return left != right
        elif op == ComparisonOp.LT:
            return left < right
        elif op == ComparisonOp.LE:
            return left <= right
        elif op == ComparisonOp.GT:
            return left > right
        elif op == ComparisonOp.GE:
            return left >= right
        elif op == ComparisonOp.LIKE:
            # Simple LIKE implementation
            pattern = str(right).replace("%", ".*").replace("_", ".")
            import re

            return bool(re.match(f"^{pattern}$", str(left), re.IGNORECASE))
        return False


class ProjectOperator(Operator):
    """Project operator that selects specific columns."""

    def __init__(
        self, child: Operator, columns: list[str], aliases: list[str | None]
    ) -> None:
        self._child = child
        self._columns = columns
        self._aliases = aliases

    def open(self) -> None:
        self._child.open()

    def next(self) -> Row | None:
        row = self._child.next()
        if row is None:
            return None

        # Handle SELECT *
        if len(self._columns) == 1 and self._columns[0] == "*":
            return row

        values = []
        output_columns = []
        for i, col in enumerate(self._columns):
            if col == "*":
                values.extend(row.values)
                output_columns.extend(row.columns)
            else:
                values.append(row.get(col))
                output_columns.append(self._aliases[i] or col)

        return Row(columns=output_columns, values=values)

    def close(self) -> None:
        self._child.close()


class SortOperator(Operator):
    """Sort operator that orders rows."""

    def __init__(
        self, child: Operator, sort_keys: list[str], ascending: list[bool]
    ) -> None:
        self._child = child
        self._sort_keys = sort_keys
        self._ascending = ascending
        self._sorted_rows: list[Row] = []
        self._current_idx = 0

    def open(self) -> None:
        self._child.open()
        # Materialize all rows and sort
        rows = []
        while True:
            row = self._child.next()
            if row is None:
                break
            rows.append(row)

        # Sort by keys
        def sort_key(row: Row) -> tuple:
            values = []
            for i, key in enumerate(self._sort_keys):
                val = row.get(key)
                # Handle None values
                if val is None:
                    val = "" if isinstance(val, str) else 0
                # Negate for descending
                if not self._ascending[i]:
                    if isinstance(val, (int, float)):
                        val = -val
                values.append(val)
            return tuple(values)

        self._sorted_rows = sorted(rows, key=sort_key)
        self._current_idx = 0

    def next(self) -> Row | None:
        if self._current_idx >= len(self._sorted_rows):
            return None
        row = self._sorted_rows[self._current_idx]
        self._current_idx += 1
        return row

    def close(self) -> None:
        self._child.close()
        self._sorted_rows = []
        self._current_idx = 0


class LimitOperator(Operator):
    """Limit operator that restricts row count."""

    def __init__(self, child: Operator, limit: int, offset: int = 0) -> None:
        self._child = child
        self._limit = limit
        self._offset = offset
        self._returned = 0
        self._skipped = 0
        self._offset_done = False

    def open(self) -> None:
        self._child.open()
        self._returned = 0
        self._skipped = 0
        self._offset_done = False

    def next(self) -> Row | None:
        # Skip offset rows (only once at the beginning)
        while not self._offset_done and self._skipped < self._offset:
            row = self._child.next()
            if row is None:
                self._offset_done = True
                return None
            self._skipped += 1
        
        self._offset_done = True

        if self._returned >= self._limit:
            return None
        row = self._child.next()
        if row is None:
            return None
        self._returned += 1
        return row

    def close(self) -> None:
        self._child.close()


@dataclass
class TableInfo:
    """Information about a table."""

    name: str
    columns: list[str]
    rows: list[Row] = field(default_factory=list)


class TableRegistry:
    """In-memory table registry for storing table metadata and data.

    This is a simplified in-memory implementation. A production implementation
    would store metadata in the system catalog and data in heap pages.
    """

    def __init__(self) -> None:
        self._tables: dict[str, TableInfo] = {}

    def create_table(self, name: str, columns: list[str]) -> bool:
        """Create a new table."""
        if name in self._tables:
            return False
        self._tables[name] = TableInfo(name=name, columns=columns)
        return True

    def drop_table(self, name: str) -> bool:
        """Drop a table."""
        if name not in self._tables:
            return False
        del self._tables[name]
        return True

    def get_table(self, name: str) -> TableInfo | None:
        """Get table information."""
        return self._tables.get(name)

    def insert_row(self, table_name: str, row: Row) -> bool:
        """Insert a row into a table."""
        table = self._tables.get(table_name)
        if table is None:
            return False
        table.rows.append(row)
        return True

    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        return name in self._tables


class QueryExecutor:
    """Executes logical plans against the database.

    The executor converts logical plans into physical operator trees
    and executes them using the Volcano iterator model.
    """

    def __init__(
        self,
        buffer_pool: BufferPool | None = None,
        txn_manager: MVCCTransactionManager | None = None,
        index_manager: BTreeIndexManager | None = None,
    ) -> None:
        self._buffer_pool = buffer_pool
        self._txn_manager = txn_manager
        self._index_manager = index_manager
        self._table_registry = TableRegistry()

    def execute(self, plan: LogicalPlan) -> ExecutionResult:
        """Execute a logical plan.

        Args:
            plan: The logical plan to execute.

        Returns:
            ExecutionResult with rows and/or status message.
        """
        if isinstance(plan, Project):
            return self._execute_query(plan)
        elif isinstance(plan, Sort):
            return self._execute_query(plan)
        elif isinstance(plan, Limit):
            return self._execute_query(plan)
        elif isinstance(plan, InsertPlan):
            return self._execute_insert(plan)
        elif isinstance(plan, UpdatePlan):
            return self._execute_update(plan)
        elif isinstance(plan, DeletePlan):
            return self._execute_delete(plan)
        elif isinstance(plan, CreateTablePlan):
            return self._execute_create_table(plan)
        elif isinstance(plan, DropTablePlan):
            return self._execute_drop_table(plan)
        elif isinstance(plan, TransactionPlan):
            return self._execute_transaction(plan)
        else:
            return ExecutionResult(message=f"Unsupported plan type: {type(plan).__name__}")

    def _execute_query(self, plan: LogicalPlan) -> ExecutionResult:
        """Execute a SELECT query."""
        try:
            operator = self._build_operator_tree(plan)
            rows = list(operator)

            columns = []
            if rows:
                columns = rows[0].columns

            return ExecutionResult(rows=rows, columns=columns, message="OK")
        except Exception as e:
            return ExecutionResult(message=f"Error: {e}")

    def _build_operator_tree(self, plan: LogicalPlan) -> Operator:
        """Build a physical operator tree from a logical plan."""
        if isinstance(plan, TableScan):
            return SeqScanOperator(
                table_name=plan.table_name,
                table_registry=self._table_registry,
                buffer_pool=self._buffer_pool,
            )
        elif isinstance(plan, Filter):
            child = self._build_operator_tree(plan.input)
            return FilterOperator(child=child, predicate=plan.predicate)
        elif isinstance(plan, Project):
            child = self._build_operator_tree(plan.input)
            columns = []
            aliases = []
            for item in plan.items:
                if isinstance(item.expr, ColumnExpr):
                    columns.append(item.expr.column.name)
                elif isinstance(item.expr, AggregateExpr):
                    # For aggregates, use the string representation
                    columns.append(str(item.expr))
                else:
                    columns.append(str(item.expr))
                aliases.append(item.alias)
            return ProjectOperator(child=child, columns=columns, aliases=aliases)
        elif isinstance(plan, Sort):
            child = self._build_operator_tree(plan.input)
            sort_keys = []
            ascending = []
            for item in plan.order_by:
                if isinstance(item.expr, ColumnExpr):
                    sort_keys.append(item.expr.column.name)
                else:
                    sort_keys.append(str(item.expr))
                ascending.append(item.ascending)
            return SortOperator(child=child, sort_keys=sort_keys, ascending=ascending)
        elif isinstance(plan, Limit):
            child = self._build_operator_tree(plan.input)
            return LimitOperator(child=child, limit=plan.count, offset=plan.offset)
        else:
            raise ValueError(f"Unsupported plan node: {type(plan).__name__}")

    def _execute_insert(self, plan: InsertPlan) -> ExecutionResult:
        """Execute an INSERT statement."""
        table = self._table_registry.get_table(plan.table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{plan.table_name}' does not exist")

        columns = plan.columns if plan.columns else table.columns
        count = 0
        for value_row in plan.values:
            values = [lit.value for lit in value_row]
            row = Row(columns=columns, values=values)
            if self._table_registry.insert_row(plan.table_name, row):
                count += 1

        return ExecutionResult(
            affected_rows=count, message=f"OK: {count} row(s) inserted"
        )

    def _execute_update(self, plan: UpdatePlan) -> ExecutionResult:
        """Execute an UPDATE statement."""
        table = self._table_registry.get_table(plan.table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{plan.table_name}' does not exist")

        count = 0
        for row in table.rows:
            # Check predicate
            if plan.predicate:
                filter_op = FilterOperator(child=_SingleRowOperator(row), predicate=plan.predicate)
                filter_op.open()
                matched = filter_op.next() is not None
                filter_op.close()
                if not matched:
                    continue

            # Apply updates
            for col_name, expr in plan.assignments.items():
                if col_name in row.columns:
                    idx = row.columns.index(col_name)
                    if isinstance(expr, LiteralExpr):
                        row.values[idx] = expr.literal.value
            count += 1

        return ExecutionResult(
            affected_rows=count, message=f"OK: {count} row(s) updated"
        )

    def _execute_delete(self, plan: DeletePlan) -> ExecutionResult:
        """Execute a DELETE statement."""
        table = self._table_registry.get_table(plan.table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{plan.table_name}' does not exist")

        rows_to_keep = []
        count = 0
        for row in table.rows:
            # Check predicate
            if plan.predicate:
                filter_op = FilterOperator(child=_SingleRowOperator(row), predicate=plan.predicate)
                filter_op.open()
                matched = filter_op.next() is not None
                filter_op.close()
                if not matched:
                    rows_to_keep.append(row)
                    continue

            count += 1

        table.rows = rows_to_keep
        return ExecutionResult(
            affected_rows=count, message=f"OK: {count} row(s) deleted"
        )

    def _execute_create_table(self, plan: CreateTablePlan) -> ExecutionResult:
        """Execute a CREATE TABLE statement."""
        if self._table_registry.table_exists(plan.table_name):
            if plan.if_not_exists:
                return ExecutionResult(message="OK: Table already exists")
            return ExecutionResult(message=f"Table '{plan.table_name}' already exists")

        columns = [col.name for col in plan.columns]
        if self._table_registry.create_table(plan.table_name, columns):
            return ExecutionResult(message=f"OK: Table '{plan.table_name}' created")
        return ExecutionResult(message="Failed to create table")

    def _execute_drop_table(self, plan: DropTablePlan) -> ExecutionResult:
        """Execute a DROP TABLE statement."""
        if not self._table_registry.table_exists(plan.table_name):
            if plan.if_exists:
                return ExecutionResult(message="OK: Table does not exist")
            return ExecutionResult(message=f"Table '{plan.table_name}' does not exist")

        if self._table_registry.drop_table(plan.table_name):
            return ExecutionResult(message=f"OK: Table '{plan.table_name}' dropped")
        return ExecutionResult(message="Failed to drop table")

    def _execute_transaction(self, plan: TransactionPlan) -> ExecutionResult:
        """Execute a transaction control statement."""
        if plan.statement_type == StatementType.BEGIN:
            return ExecutionResult(message="OK: Transaction started")
        elif plan.statement_type == StatementType.COMMIT:
            return ExecutionResult(message="OK: Transaction committed")
        elif plan.statement_type == StatementType.ROLLBACK:
            return ExecutionResult(message="OK: Transaction rolled back")
        return ExecutionResult(message=f"Unknown transaction statement: {plan.statement_type}")


class _SingleRowOperator(Operator):
    """Helper operator that returns a single row."""

    def __init__(self, row: Row) -> None:
        self._row = row
        self._returned = False

    def open(self) -> None:
        self._returned = False

    def next(self) -> Row | None:
        if self._returned:
            return None
        self._returned = True
        return self._row

    def close(self) -> None:
        pass
