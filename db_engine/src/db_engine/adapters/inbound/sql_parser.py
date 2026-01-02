"""SQL Parser using sqlglot.

This module provides SQL parsing and logical plan generation using sqlglot.
It converts SQL strings into an internal logical plan representation that
can be optimized and executed.

Supported statements:
    - SELECT (with WHERE, ORDER BY, LIMIT)
    - INSERT
    - UPDATE
    - DELETE
    - CREATE TABLE
    - DROP TABLE

References:
    - sqlglot documentation: https://sqlglot.com/
    - design.md Section 2 (Architecture)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import sqlglot
from sqlglot import exp


class StatementType(Enum):
    """Types of SQL statements."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    BEGIN = "begin"
    COMMIT = "commit"
    ROLLBACK = "rollback"


class ComparisonOp(Enum):
    """Comparison operators for predicates."""

    EQ = "="
    NE = "<>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    LIKE = "LIKE"
    IN = "IN"
    BETWEEN = "BETWEEN"


class LogicalOp(Enum):
    """Logical operators for combining predicates."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class AggregateFunc(Enum):
    """Aggregate functions."""

    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"


class DataType(Enum):
    """SQL data types."""

    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    BLOB = "BLOB"


@dataclass
class ColumnDef:
    """Column definition for CREATE TABLE."""

    name: str
    data_type: DataType
    nullable: bool = True
    primary_key: bool = False
    max_length: int | None = None  # For VARCHAR


@dataclass
class ColumnRef:
    """Reference to a column, optionally qualified with table name."""

    name: str
    table: str | None = None

    def __str__(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


@dataclass
class Literal:
    """A literal value."""

    value: Any
    data_type: DataType | None = None


@dataclass
class Expression(ABC):
    """Base class for expressions."""

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class ColumnExpr(Expression):
    """Column reference expression."""

    column: ColumnRef

    def __str__(self) -> str:
        return str(self.column)


@dataclass
class LiteralExpr(Expression):
    """Literal value expression."""

    literal: Literal

    def __str__(self) -> str:
        if isinstance(self.literal.value, str):
            return f"'{self.literal.value}'"
        return str(self.literal.value)


@dataclass
class ComparisonExpr(Expression):
    """Comparison expression (e.g., col = value)."""

    left: Expression
    op: ComparisonOp
    right: Expression | None = None  # None for IS NULL / IS NOT NULL

    def __str__(self) -> str:
        if self.right is None:
            return f"{self.left} {self.op.value}"
        return f"{self.left} {self.op.value} {self.right}"


@dataclass
class LogicalExpr(Expression):
    """Logical expression combining other expressions."""

    op: LogicalOp
    operands: list[Expression] = field(default_factory=list)

    def __str__(self) -> str:
        if self.op == LogicalOp.NOT:
            return f"NOT ({self.operands[0]})"
        op_str = f" {self.op.value} "
        return f"({op_str.join(str(o) for o in self.operands)})"


@dataclass
class AggregateExpr(Expression):
    """Aggregate function expression."""

    func: AggregateFunc
    arg: Expression | None = None  # None for COUNT(*)
    distinct: bool = False

    def __str__(self) -> str:
        if self.arg is None:
            return f"{self.func.value}(*)"
        distinct_str = "DISTINCT " if self.distinct else ""
        return f"{self.func.value}({distinct_str}{self.arg})"


@dataclass
class SelectItem:
    """An item in a SELECT list."""

    expr: Expression
    alias: str | None = None


@dataclass
class OrderByItem:
    """An item in an ORDER BY clause."""

    expr: Expression
    ascending: bool = True


# Logical Plan Nodes


@dataclass
class LogicalPlan(ABC):
    """Base class for logical plan nodes."""

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class TableScan(LogicalPlan):
    """Scan a table."""

    table_name: str
    alias: str | None = None

    def __str__(self) -> str:
        if self.alias:
            return f"TableScan({self.table_name} AS {self.alias})"
        return f"TableScan({self.table_name})"


@dataclass
class Filter(LogicalPlan):
    """Filter rows based on a predicate."""

    input: LogicalPlan
    predicate: Expression

    def __str__(self) -> str:
        return f"Filter({self.predicate})\n  -> {self.input}"


@dataclass
class Project(LogicalPlan):
    """Project (select) specific columns."""

    input: LogicalPlan
    items: list[SelectItem]

    def __str__(self) -> str:
        cols = ", ".join(str(item.expr) for item in self.items)
        return f"Project({cols})\n  -> {self.input}"


@dataclass
class Sort(LogicalPlan):
    """Sort rows by specified columns."""

    input: LogicalPlan
    order_by: list[OrderByItem]

    def __str__(self) -> str:
        cols = ", ".join(
            f"{item.expr} {'ASC' if item.ascending else 'DESC'}" for item in self.order_by
        )
        return f"Sort({cols})\n  -> {self.input}"


@dataclass
class Limit(LogicalPlan):
    """Limit the number of rows returned."""

    input: LogicalPlan
    count: int
    offset: int = 0

    def __str__(self) -> str:
        return f"Limit({self.count}, offset={self.offset})\n  -> {self.input}"


@dataclass
class Aggregate(LogicalPlan):
    """Aggregate rows with GROUP BY."""

    input: LogicalPlan
    group_by: list[Expression]
    aggregates: list[AggregateExpr]

    def __str__(self) -> str:
        groups = ", ".join(str(g) for g in self.group_by)
        aggs = ", ".join(str(a) for a in self.aggregates)
        return f"Aggregate(group=[{groups}], agg=[{aggs}])\n  -> {self.input}"


@dataclass
class InsertPlan(LogicalPlan):
    """Insert rows into a table."""

    table_name: str
    columns: list[str]
    values: list[list[Literal]]

    def __str__(self) -> str:
        return f"Insert({self.table_name}, cols={self.columns}, rows={len(self.values)})"


@dataclass
class UpdatePlan(LogicalPlan):
    """Update rows in a table."""

    table_name: str
    assignments: dict[str, Expression]
    predicate: Expression | None = None

    def __str__(self) -> str:
        assigns = ", ".join(f"{k}={v}" for k, v in self.assignments.items())
        where = f" WHERE {self.predicate}" if self.predicate else ""
        return f"Update({self.table_name}, SET {assigns}{where})"


@dataclass
class DeletePlan(LogicalPlan):
    """Delete rows from a table."""

    table_name: str
    predicate: Expression | None = None

    def __str__(self) -> str:
        where = f" WHERE {self.predicate}" if self.predicate else ""
        return f"Delete({self.table_name}{where})"


@dataclass
class CreateTablePlan(LogicalPlan):
    """Create a new table."""

    table_name: str
    columns: list[ColumnDef]
    if_not_exists: bool = False

    def __str__(self) -> str:
        cols = ", ".join(f"{c.name} {c.data_type.value}" for c in self.columns)
        return f"CreateTable({self.table_name}, [{cols}])"


@dataclass
class DropTablePlan(LogicalPlan):
    """Drop a table."""

    table_name: str
    if_exists: bool = False

    def __str__(self) -> str:
        return f"DropTable({self.table_name})"


@dataclass
class TransactionPlan(LogicalPlan):
    """Transaction control statement."""

    statement_type: StatementType

    def __str__(self) -> str:
        return f"Transaction({self.statement_type.value})"


class ParseError(Exception):
    """Error during SQL parsing."""

    pass


class SQLParser:
    """SQL parser using sqlglot.

    Parses SQL strings and produces logical plans that can be
    optimized and executed.

    Example:
        >>> parser = SQLParser()
        >>> plan = parser.parse("SELECT id, name FROM users WHERE age > 18")
        >>> print(plan)
        Project(id, name)
          -> Filter(age > 18)
            -> TableScan(users)
    """

    def __init__(self, dialect: str = "sqlite") -> None:
        """Initialize the parser.

        Args:
            dialect: SQL dialect to use for parsing (default: sqlite).
        """
        self._dialect = dialect

    def parse(self, sql: str) -> LogicalPlan:
        """Parse a SQL string into a logical plan.

        Args:
            sql: The SQL statement to parse.

        Returns:
            A logical plan representing the query.

        Raises:
            ParseError: If the SQL is invalid or unsupported.
        """
        try:
            statements = sqlglot.parse(sql, dialect=self._dialect)
        except Exception as e:
            raise ParseError(f"Failed to parse SQL: {e}") from e

        if not statements:
            raise ParseError("Empty SQL statement")

        if len(statements) > 1:
            raise ParseError("Multiple statements not supported")

        stmt = statements[0]
        if stmt is None:
            raise ParseError("Failed to parse SQL statement")

        return self._convert_statement(stmt)

    def _convert_statement(self, stmt: exp.Expression) -> LogicalPlan:
        """Convert a sqlglot expression to a logical plan."""
        if isinstance(stmt, exp.Select):
            return self._convert_select(stmt)
        elif isinstance(stmt, exp.Insert):
            return self._convert_insert(stmt)
        elif isinstance(stmt, exp.Update):
            return self._convert_update(stmt)
        elif isinstance(stmt, exp.Delete):
            return self._convert_delete(stmt)
        elif isinstance(stmt, exp.Create):
            return self._convert_create(stmt)
        elif isinstance(stmt, exp.Drop):
            return self._convert_drop(stmt)
        elif isinstance(stmt, exp.Transaction):
            return self._convert_transaction(stmt)
        elif isinstance(stmt, exp.Commit):
            return TransactionPlan(StatementType.COMMIT)
        elif isinstance(stmt, exp.Rollback):
            return TransactionPlan(StatementType.ROLLBACK)
        else:
            raise ParseError(f"Unsupported statement type: {type(stmt).__name__}")

    def _convert_select(self, stmt: exp.Select) -> LogicalPlan:
        """Convert a SELECT statement to a logical plan."""
        # Start with table scan
        from_clause = stmt.find(exp.From)
        if from_clause is None:
            raise ParseError("SELECT requires FROM clause")

        table = from_clause.find(exp.Table)
        if table is None:
            raise ParseError("Invalid FROM clause")

        plan: LogicalPlan = TableScan(
            table_name=table.name,
            alias=table.alias if hasattr(table, "alias") and table.alias else None,
        )

        # Apply WHERE filter
        where = stmt.find(exp.Where)
        if where:
            predicate = self._convert_expression(where.this)
            plan = Filter(input=plan, predicate=predicate)

        # Apply GROUP BY
        group = stmt.find(exp.Group)
        aggregates = self._find_aggregates(stmt)
        if group or aggregates:
            group_by = []
            if group:
                for expr in group.expressions:
                    group_by.append(self._convert_expression(expr))
            plan = Aggregate(input=plan, group_by=group_by, aggregates=aggregates)

        # Apply projection
        select_items = []
        for col in stmt.expressions:
            item = self._convert_select_item(col)
            select_items.append(item)
        plan = Project(input=plan, items=select_items)

        # Apply ORDER BY
        order = stmt.find(exp.Order)
        if order:
            order_items = []
            for expr in order.expressions:
                ascending = True
                if isinstance(expr, exp.Ordered):
                    ascending = not expr.args.get("desc", False)
                    expr = expr.this
                order_items.append(
                    OrderByItem(expr=self._convert_expression(expr), ascending=ascending)
                )
            plan = Sort(input=plan, order_by=order_items)

        # Apply LIMIT
        limit = stmt.find(exp.Limit)
        if limit:
            # Extract count from various possible structures
            count = 10  # default
            # Try limit.expression first (newer sqlglot versions use this)
            if hasattr(limit, "expression") and limit.expression:
                expr = limit.expression
                if isinstance(expr, exp.Literal):
                    count = int(expr.this)
                else:
                    try:
                        count = int(str(expr))
                    except (ValueError, TypeError):
                        pass
            elif limit.this:
                if isinstance(limit.this, exp.Literal):
                    count = int(limit.this.this)
                elif hasattr(limit.this, "this"):
                    # Nested structure
                    count = int(limit.this.this)
                else:
                    # Direct value
                    try:
                        count = int(str(limit.this))
                    except (ValueError, TypeError):
                        pass

            offset = 0
            offset_expr = stmt.find(exp.Offset)
            if offset_expr:
                # Try offset_expr.expression first (newer sqlglot versions use this)
                if hasattr(offset_expr, "expression") and offset_expr.expression:
                    expr = offset_expr.expression
                    if isinstance(expr, exp.Literal):
                        offset = int(expr.this)
                    else:
                        try:
                            offset = int(str(expr))
                        except (ValueError, TypeError):
                            pass
                elif offset_expr.this:
                    # Handle older structure where offset_expr.this has the value
                    if isinstance(offset_expr.this, exp.Literal):
                        offset = int(offset_expr.this.this)
                    elif hasattr(offset_expr.this, "this"):
                        offset = int(offset_expr.this.this)
                    else:
                        try:
                            offset = int(str(offset_expr.this))
                        except (ValueError, TypeError):
                            pass

            plan = Limit(input=plan, count=count, offset=offset)

        return plan

    def _convert_select_item(self, col: exp.Expression) -> SelectItem:
        """Convert a SELECT item."""
        alias = None
        if isinstance(col, exp.Alias):
            alias = col.alias
            col = col.this

        expr = self._convert_expression(col)
        return SelectItem(expr=expr, alias=alias)

    def _find_aggregates(self, stmt: exp.Select) -> list[AggregateExpr]:
        """Find all aggregate functions in a SELECT."""
        aggregates = []
        for func in stmt.find_all(exp.AggFunc):
            agg = self._convert_aggregate(func)
            if agg:
                aggregates.append(agg)
        return aggregates

    def _convert_aggregate(self, func: exp.AggFunc) -> AggregateExpr | None:
        """Convert an aggregate function."""
        func_map = {
            "COUNT": AggregateFunc.COUNT,
            "SUM": AggregateFunc.SUM,
            "AVG": AggregateFunc.AVG,
            "MIN": AggregateFunc.MIN,
            "MAX": AggregateFunc.MAX,
        }

        func_name = func.key.upper()
        if func_name not in func_map:
            return None

        distinct = func.args.get("distinct", False)

        # Handle COUNT(*)
        if isinstance(func, exp.Count) and isinstance(func.this, exp.Star):
            return AggregateExpr(func=func_map[func_name], arg=None, distinct=distinct)

        arg = self._convert_expression(func.this) if func.this else None
        return AggregateExpr(func=func_map[func_name], arg=arg, distinct=distinct)

    def _convert_expression(self, expr: exp.Expression) -> Expression:
        """Convert a sqlglot expression to our internal representation."""
        if isinstance(expr, exp.Column):
            return ColumnExpr(
                column=ColumnRef(
                    name=expr.name,
                    table=expr.table if hasattr(expr, "table") and expr.table else None,
                )
            )
        elif isinstance(expr, exp.Literal):
            if expr.is_number:
                value = int(expr.this) if "." not in expr.this else float(expr.this)
                dtype = DataType.INTEGER if isinstance(value, int) else DataType.FLOAT
            else:
                value = expr.this
                dtype = DataType.VARCHAR
            return LiteralExpr(literal=Literal(value=value, data_type=dtype))
        elif isinstance(expr, exp.Star):
            return ColumnExpr(column=ColumnRef(name="*"))
        elif isinstance(expr, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
            op_map = {
                exp.EQ: ComparisonOp.EQ,
                exp.NEQ: ComparisonOp.NE,
                exp.LT: ComparisonOp.LT,
                exp.LTE: ComparisonOp.LE,
                exp.GT: ComparisonOp.GT,
                exp.GTE: ComparisonOp.GE,
            }
            return ComparisonExpr(
                left=self._convert_expression(expr.left),
                op=op_map[type(expr)],
                right=self._convert_expression(expr.right),
            )
        elif isinstance(expr, exp.And):
            return LogicalExpr(
                op=LogicalOp.AND,
                operands=[
                    self._convert_expression(expr.left),
                    self._convert_expression(expr.right),
                ],
            )
        elif isinstance(expr, exp.Or):
            return LogicalExpr(
                op=LogicalOp.OR,
                operands=[
                    self._convert_expression(expr.left),
                    self._convert_expression(expr.right),
                ],
            )
        elif isinstance(expr, exp.Not):
            return LogicalExpr(
                op=LogicalOp.NOT, operands=[self._convert_expression(expr.this)]
            )
        elif isinstance(expr, exp.Is):
            if isinstance(expr.right, exp.Null):
                return ComparisonExpr(
                    left=self._convert_expression(expr.left),
                    op=ComparisonOp.IS_NULL,
                    right=None,
                )
            return ComparisonExpr(
                left=self._convert_expression(expr.left),
                op=ComparisonOp.IS_NOT_NULL,
                right=None,
            )
        elif isinstance(expr, exp.Like):
            return ComparisonExpr(
                left=self._convert_expression(expr.this),
                op=ComparisonOp.LIKE,
                right=self._convert_expression(expr.expression),
            )
        elif isinstance(expr, exp.AggFunc):
            agg = self._convert_aggregate(expr)
            if agg:
                return agg
            raise ParseError(f"Unsupported aggregate function: {expr}")
        elif isinstance(expr, exp.Alias):
            return self._convert_expression(expr.this)
        else:
            raise ParseError(f"Unsupported expression type: {type(expr).__name__}")

    def _convert_insert(self, stmt: exp.Insert) -> LogicalPlan:
        """Convert an INSERT statement to a logical plan."""
        table = stmt.find(exp.Table)
        if table is None:
            raise ParseError("INSERT requires table name")

        columns = []
        col_list = stmt.find(exp.Schema)
        if col_list:
            columns = [col.name for col in col_list.expressions]

        values_list = []
        values = stmt.find(exp.Values)
        if values:
            for tuple_expr in values.expressions:
                row = []
                for val in tuple_expr.expressions:
                    if isinstance(val, exp.Literal):
                        if val.is_number:
                            value = int(val.this) if "." not in val.this else float(val.this)
                            dtype = DataType.INTEGER if isinstance(value, int) else DataType.FLOAT
                        else:
                            value = val.this
                            dtype = DataType.VARCHAR
                        row.append(Literal(value=value, data_type=dtype))
                    else:
                        raise ParseError(f"Unsupported value type: {type(val).__name__}")
                values_list.append(row)

        return InsertPlan(table_name=table.name, columns=columns, values=values_list)

    def _convert_update(self, stmt: exp.Update) -> LogicalPlan:
        """Convert an UPDATE statement to a logical plan."""
        table = stmt.find(exp.Table)
        if table is None:
            raise ParseError("UPDATE requires table name")

        assignments = {}

        # Look for SET expressions - they contain the assignments
        set_exprs = stmt.find_all(exp.Set)
        for set_expr in set_exprs:
            for eq in set_expr.expressions:
                if isinstance(eq, exp.EQ):
                    col = eq.left
                    val = eq.right
                    if isinstance(col, exp.Column):
                        assignments[col.name] = self._convert_expression(val)

        # Alternative: look for EQ expressions directly under the statement
        # when Set wrapper is not used
        if not assignments:
            for child in stmt.expressions:
                if isinstance(child, exp.EQ):
                    col = child.left
                    val = child.right
                    if isinstance(col, exp.Column):
                        assignments[col.name] = self._convert_expression(val)

        predicate = None
        where = stmt.find(exp.Where)
        if where:
            predicate = self._convert_expression(where.this)

        return UpdatePlan(
            table_name=table.name, assignments=assignments, predicate=predicate
        )

    def _convert_delete(self, stmt: exp.Delete) -> LogicalPlan:
        """Convert a DELETE statement to a logical plan."""
        table = stmt.find(exp.Table)
        if table is None:
            raise ParseError("DELETE requires table name")

        predicate = None
        where = stmt.find(exp.Where)
        if where:
            predicate = self._convert_expression(where.this)

        return DeletePlan(table_name=table.name, predicate=predicate)

    def _convert_create(self, stmt: exp.Create) -> LogicalPlan:
        """Convert a CREATE TABLE statement to a logical plan."""
        table = stmt.find(exp.Table)
        if table is None:
            raise ParseError("CREATE TABLE requires table name")

        columns = []
        schema = stmt.find(exp.Schema)
        if schema:
            for col_def in schema.expressions:
                if isinstance(col_def, exp.ColumnDef):
                    col_name = col_def.name
                    col_type = self._convert_data_type(col_def.kind)
                    nullable = True
                    primary_key = False

                    for constraint in col_def.constraints:
                        if isinstance(constraint.kind, exp.NotNullColumnConstraint):
                            nullable = False
                        elif isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                            primary_key = True

                    max_length = None
                    if col_def.kind and hasattr(col_def.kind, "expressions"):
                        for expr in col_def.kind.expressions:
                            if isinstance(expr, exp.Literal) and expr.is_number:
                                max_length = int(expr.this)

                    columns.append(
                        ColumnDef(
                            name=col_name,
                            data_type=col_type,
                            nullable=nullable,
                            primary_key=primary_key,
                            max_length=max_length,
                        )
                    )

        if_not_exists = stmt.args.get("exists", False)

        return CreateTablePlan(
            table_name=table.name, columns=columns, if_not_exists=if_not_exists
        )

    def _convert_data_type(self, dtype: exp.DataType | None) -> DataType:
        """Convert a sqlglot data type to our internal representation."""
        if dtype is None:
            return DataType.TEXT

        type_map = {
            "INT": DataType.INTEGER,
            "INTEGER": DataType.INTEGER,
            "BIGINT": DataType.BIGINT,
            "FLOAT": DataType.FLOAT,
            "DOUBLE": DataType.DOUBLE,
            "REAL": DataType.DOUBLE,
            "VARCHAR": DataType.VARCHAR,
            "CHAR": DataType.VARCHAR,
            "TEXT": DataType.TEXT,
            "BOOLEAN": DataType.BOOLEAN,
            "BOOL": DataType.BOOLEAN,
            "TIMESTAMP": DataType.TIMESTAMP,
            "DATETIME": DataType.TIMESTAMP,
            "BLOB": DataType.BLOB,
            "BINARY": DataType.BLOB,
            "VARBINARY": DataType.BLOB,
            "LONGBLOB": DataType.BLOB,
            "MEDIUMBLOB": DataType.BLOB,
            "TINYBLOB": DataType.BLOB,
        }

        # dtype.this is a DataType.Type enum, we need its name
        if hasattr(dtype.this, "name"):
            type_name = dtype.this.name.upper()
        else:
            type_name = str(dtype.this).upper() if dtype.this else "TEXT"
        return type_map.get(type_name, DataType.TEXT)

    def _convert_drop(self, stmt: exp.Drop) -> LogicalPlan:
        """Convert a DROP TABLE statement to a logical plan."""
        table = stmt.find(exp.Table)
        if table is None:
            raise ParseError("DROP TABLE requires table name")

        if_exists = stmt.args.get("exists", False)

        return DropTablePlan(table_name=table.name, if_exists=if_exists)

    def _convert_transaction(self, stmt: exp.Transaction) -> LogicalPlan:
        """Convert a transaction statement to a logical plan."""
        return TransactionPlan(StatementType.BEGIN)
