"""Application layer for the database engine.

The application layer orchestrates domain logic to fulfill use cases.
It contains commands (write operations) and queries (read operations).

Exports:
    DatabaseEngine:
        - DatabaseEngine: Main entry point for the database
    Executor:
        - QueryExecutor: Executes SQL plans using Volcano iterator model
        - ExecutionResult: Result of query execution
        - Row: A row of data
        - Operator: Base class for executor operators
        - TableRegistry: In-memory table registry
"""

from db_engine.application.database_engine import DatabaseEngine
from db_engine.application.executor import (
    ExecutionResult,
    FilterOperator,
    LimitOperator,
    Operator,
    ProjectOperator,
    QueryExecutor,
    Row,
    SeqScanOperator,
    SortOperator,
    TableInfo,
    TableRegistry,
)

__all__ = [
    "DatabaseEngine",
    "QueryExecutor",
    "ExecutionResult",
    "Row",
    "Operator",
    "TableRegistry",
    "TableInfo",
    "SeqScanOperator",
    "FilterOperator",
    "ProjectOperator",
    "SortOperator",
    "LimitOperator",
]
