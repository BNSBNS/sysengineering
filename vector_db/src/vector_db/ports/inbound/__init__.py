"""Inbound ports - interfaces for external callers.

Inbound ports define the operations that can be performed
on the vector database from the outside world.
"""

from vector_db.ports.inbound.vector_database_port import VectorDatabasePort

__all__ = ["VectorDatabasePort"]
