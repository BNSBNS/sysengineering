"""REST API adapter for the vector database.

This module provides a FastAPI-based REST API for vector similarity search.

Endpoints:
    POST /vectors - Insert a vector
    POST /search - Search for similar vectors
    GET /vectors/{id} - Get a vector by ID
    DELETE /vectors/{id} - Delete a vector
    GET /health - Health check
    GET /stats - Database statistics

Usage:
    from vector_db.adapters.inbound.rest_api import create_app
    from vector_db.application import VectorDatabase

    db = VectorDatabase(dim=128)
    app = create_app(db)
    # Run with uvicorn: uvicorn app:app --host 0.0.0.0 --port 8000

References:
    - FastAPI documentation: https://fastapi.tiangolo.com/
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from vector_db.ports.inbound import VectorDatabasePort


class VectorInsertRequest(BaseModel):
    """Request model for vector insertion."""

    id: str = Field(..., description="Unique vector identifier")
    vector: list[float] = Field(..., description="Vector data as list of floats")


class VectorInsertResponse(BaseModel):
    """Response model for vector insertion."""

    success: bool = Field(..., description="Whether insertion succeeded")
    id: str = Field(..., description="The inserted vector ID")


class SearchRequest(BaseModel):
    """Request model for similarity search."""

    vector: list[float] = Field(..., description="Query vector")
    k: int = Field(10, ge=1, le=1000, description="Number of results to return")


class SearchResultItem(BaseModel):
    """A single search result."""

    id: str = Field(..., description="Vector ID")
    distance: float = Field(..., description="Distance to query")


class SearchResponse(BaseModel):
    """Response model for similarity search."""

    results: list[SearchResultItem] = Field(..., description="Search results")
    query_dim: int = Field(..., description="Query vector dimension")


class VectorResponse(BaseModel):
    """Response model for vector retrieval."""

    id: str = Field(..., description="Vector ID")
    vector: list[float] = Field(..., description="Vector data")
    dim: int = Field(..., description="Vector dimension")


class StatsResponse(BaseModel):
    """Response model for database statistics."""

    dim: int = Field(..., description="Vector dimensionality")
    index_type: str = Field(..., description="Index type")
    metric: str = Field(..., description="Distance metric")
    num_vectors: int = Field(..., description="Number of vectors")
    is_trained: bool = Field(..., description="Whether index is trained")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")


def create_app(db: VectorDatabasePort) -> FastAPI:
    """Create FastAPI application for vector database.

    Args:
        db: VectorDatabase instance implementing VectorDatabasePort

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Vector Database API",
        description="REST API for vector similarity search",
        version="0.1.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version="0.1.0")

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        """Get database statistics."""
        db_stats = db.stats()
        return StatsResponse(
            dim=db_stats["dim"],
            index_type=db_stats["index_type"],
            metric=db_stats["metric"],
            num_vectors=db_stats["num_vectors"],
            is_trained=db_stats["is_trained"],
        )

    @app.post("/vectors", response_model=VectorInsertResponse)
    async def insert_vector(request: VectorInsertRequest) -> VectorInsertResponse:
        """Insert a vector into the database."""
        try:
            vector = np.array(request.vector, dtype=np.float32)
            if vector.shape[0] != db.dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vector dimension mismatch: expected {db.dim}, got {vector.shape[0]}",
                )
            db.insert(request.id, vector)
            return VectorInsertResponse(success=True, id=request.id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """Search for similar vectors."""
        try:
            query = np.array(request.vector, dtype=np.float32)
            if query.shape[0] != db.dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query dimension mismatch: expected {db.dim}, got {query.shape[0]}",
                )
            results = db.search(query, k=request.k)
            return SearchResponse(
                results=[
                    SearchResultItem(id=r.vector_id, distance=r.distance)
                    for r in results
                ],
                query_dim=len(request.vector),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/vectors/{vector_id}", response_model=VectorResponse)
    async def get_vector(vector_id: str) -> VectorResponse:
        """Get a vector by ID."""
        vector = db.get(vector_id)
        if vector is None:
            raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")
        return VectorResponse(
            id=vector_id,
            vector=vector.tolist(),
            dim=len(vector),
        )

    @app.delete("/vectors/{vector_id}")
    async def delete_vector(vector_id: str) -> dict[str, Any]:
        """Delete a vector by ID."""
        if not db.contains(vector_id):
            raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")
        try:
            db.delete(vector_id)
            return {"success": True, "id": vector_id}
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))

    return app
