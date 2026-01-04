"""Outbound adapters - implementations for external dependencies.

Outbound adapters implement storage interfaces for persisting
vectors and index data.
"""

from vector_db.adapters.outbound.file_storage import FileVectorStorage
from vector_db.adapters.outbound.memory_storage import InMemoryVectorStorage

__all__ = ["FileVectorStorage", "InMemoryVectorStorage"]
