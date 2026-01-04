"""Application layer - orchestrates domain services for vector database operations.

This module provides the main VectorDatabase service that coordinates
index creation, vector insertion, and similarity search operations.
"""

from vector_db.application.vector_database import VectorDatabase, IndexType

__all__ = [
    "VectorDatabase",
    "IndexType",
]
