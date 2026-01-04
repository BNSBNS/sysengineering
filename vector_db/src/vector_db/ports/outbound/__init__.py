"""Outbound ports - interfaces for external dependencies.

Outbound ports define contracts for external systems that the
vector database depends on, such as persistent storage.
"""

from vector_db.ports.outbound.vector_storage_port import VectorStoragePort

__all__ = ["VectorStoragePort"]
