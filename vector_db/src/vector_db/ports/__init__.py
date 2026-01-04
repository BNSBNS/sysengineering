"""Ports layer - interfaces for adapters.

Ports define contracts that adapters implement:
- Inbound ports: interfaces for external callers (API, CLI)
- Outbound ports: interfaces for external dependencies (storage)
"""

from vector_db.ports.inbound import VectorDatabasePort
from vector_db.ports.outbound import VectorStoragePort

__all__ = [
    "VectorDatabasePort",
    "VectorStoragePort",
]
