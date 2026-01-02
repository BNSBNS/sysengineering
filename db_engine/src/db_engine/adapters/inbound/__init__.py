"""Inbound adapters for the database engine.

Inbound adapters handle incoming requests and convert them to
internal domain operations.

Exports:
    SQL Parser:
        - SQLParser: Parser that converts SQL strings to logical plans
        - LogicalPlan: Base class for all logical plan nodes
        - ParseError: Exception for parsing errors
    REST API (optional, requires fastapi):
        - create_app: Create a FastAPI application
        - run_server: Run the REST API server
"""

# REST API is optional - requires fastapi and uvicorn
try:
    from db_engine.adapters.inbound.rest_api import (
        create_app,
        run_server,
        SQLRequest,
        SQLResponse,
    )
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    create_app = None
    run_server = None
    SQLRequest = None
    SQLResponse = None

from db_engine.adapters.inbound.sql_parser import (
    AggregateExpr,
    AggregateFunc,
    ColumnDef,
    ColumnExpr,
    ColumnRef,
    ComparisonExpr,
    ComparisonOp,
    CreateTablePlan,
    DataType,
    DeletePlan,
    DropTablePlan,
    Expression,
    Filter,
    InsertPlan,
    Limit,
    Literal,
    LiteralExpr,
    LogicalExpr,
    LogicalOp,
    LogicalPlan,
    OrderByItem,
    ParseError,
    Project,
    SelectItem,
    Sort,
    SQLParser,
    StatementType,
    TableScan,
    TransactionPlan,
    UpdatePlan,
)

__all__ = [
    # REST API
    "create_app",
    "run_server",
    "SQLRequest",
    "SQLResponse",
    # SQL Parser
    "SQLParser",
    "ParseError",
    # Types
    "StatementType",
    "ComparisonOp",
    "LogicalOp",
    "AggregateFunc",
    "DataType",
    # Expressions
    "Expression",
    "ColumnRef",
    "ColumnExpr",
    "Literal",
    "LiteralExpr",
    "ComparisonExpr",
    "LogicalExpr",
    "AggregateExpr",
    # Column/Table definitions
    "ColumnDef",
    "SelectItem",
    "OrderByItem",
    # Logical Plans
    "LogicalPlan",
    "TableScan",
    "Filter",
    "Project",
    "Sort",
    "Limit",
    "InsertPlan",
    "UpdatePlan",
    "DeletePlan",
    "CreateTablePlan",
    "DropTablePlan",
    "TransactionPlan",
]
