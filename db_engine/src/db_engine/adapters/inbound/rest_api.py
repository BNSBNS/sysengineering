"""REST API adapter for the database engine.

This module provides a FastAPI-based REST API for executing SQL
queries against the database engine.

Endpoints:
    POST /execute - Execute a SQL statement
    GET /health - Health check
    GET /stats - Database statistics

Usage:
    from db_engine.adapters.inbound.rest_api import create_app
    from db_engine.application import DatabaseEngine

    db = DatabaseEngine(data_dir="/path/to/data")
    db.start()

    app = create_app(db)
    # Run with uvicorn: uvicorn app:app --host 0.0.0.0 --port 8000

References:
    - FastAPI documentation: https://fastapi.tiangolo.com/
    - design.md Section 1 (API Layer)
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db_engine.application import DatabaseEngine, ExecutionResult


class SQLRequest(BaseModel):
    """Request model for SQL execution."""

    sql: str = Field(..., description="SQL statement to execute")
    session_id: int | None = Field(None, description="Optional session ID")


class SQLResponse(BaseModel):
    """Response model for SQL execution."""

    success: bool = Field(..., description="Whether the query succeeded")
    message: str = Field("", description="Status or error message")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Result rows")
    columns: list[str] = Field(default_factory=list, description="Column names")
    affected_rows: int = Field(0, description="Number of affected rows")


class SessionResponse(BaseModel):
    """Response model for session creation."""

    session_id: int = Field(..., description="The created session ID")


class StatsResponse(BaseModel):
    """Response model for database statistics."""

    started: bool = Field(..., description="Whether database is started")
    data_dir: str = Field(..., description="Data directory path")
    page_size: int = Field(..., description="Page size in bytes")
    sessions: int = Field(..., description="Number of active sessions")
    transactions: dict[str, int] = Field(default_factory=dict, description="Transaction stats")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")


def _result_to_response(result: ExecutionResult) -> SQLResponse:
    """Convert ExecutionResult to SQLResponse."""
    rows = []
    for row in result.rows:
        rows.append(dict(zip(row.columns, row.values)))

    return SQLResponse(
        success=result.success,
        message=result.message,
        rows=rows,
        columns=result.columns,
        affected_rows=result.affected_rows,
    )


def create_app(db: DatabaseEngine) -> FastAPI:
    """Create a FastAPI application for the database engine.

    Args:
        db: The database engine to use.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(
        title="DB Engine API",
        description="REST API for executing SQL queries",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if db.is_started else "unhealthy",
            version="1.0.0",
        )

    @app.get("/stats", response_model=StatsResponse, tags=["Stats"])
    async def get_stats() -> StatsResponse:
        """Get database statistics."""
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        stats = db.get_stats()
        return StatsResponse(
            started=stats.get("started", False),
            data_dir=stats.get("data_dir", ""),
            page_size=stats.get("page_size", 0),
            sessions=stats.get("sessions", 0),
            transactions=stats.get("transactions", {}),
        )

    @app.post("/execute", response_model=SQLResponse, tags=["SQL"])
    async def execute_sql(request: SQLRequest) -> SQLResponse:
        """Execute a SQL statement.

        Args:
            request: The SQL request containing the statement.

        Returns:
            The execution result.
        """
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        try:
            result = db.execute(request.sql, request.session_id)
            return _result_to_response(result)
        except Exception as e:
            return SQLResponse(
                success=False,
                message=f"Error: {e}",
            )

    @app.post("/execute/batch", response_model=list[SQLResponse], tags=["SQL"])
    async def execute_batch(statements: list[str], session_id: int | None = None) -> list[SQLResponse]:
        """Execute multiple SQL statements.

        Args:
            statements: List of SQL statements.
            session_id: Optional session ID.

        Returns:
            List of execution results.
        """
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        results = db.execute_many(statements, session_id)
        return [_result_to_response(r) for r in results]

    @app.post("/session", response_model=SessionResponse, tags=["Sessions"])
    async def create_session() -> SessionResponse:
        """Create a new database session."""
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        session_id = db.create_session()
        return SessionResponse(session_id=session_id)

    @app.delete("/session/{session_id}", tags=["Sessions"])
    async def close_session(session_id: int) -> dict[str, str]:
        """Close a database session."""
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        try:
            db.close_session(session_id)
            return {"message": f"Session {session_id} closed"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/session/{session_id}/autocommit", tags=["Sessions"])
    async def set_autocommit(session_id: int, enabled: bool = True) -> dict[str, str]:
        """Set autocommit mode for a session."""
        if not db.is_started:
            raise HTTPException(status_code=503, detail="Database not started")

        try:
            db.set_autocommit(enabled, session_id)
            return {"message": f"Autocommit {'enabled' if enabled else 'disabled'}"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app


def run_server(
    db: DatabaseEngine,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run the REST API server.

    Args:
        db: The database engine.
        host: Host to bind to.
        port: Port to bind to.
    """
    import uvicorn

    app = create_app(db)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Example usage
    import tempfile

    with DatabaseEngine(data_dir=tempfile.mkdtemp()) as db:
        print(f"Starting server with data_dir: {db.data_dir}")
        run_server(db)
