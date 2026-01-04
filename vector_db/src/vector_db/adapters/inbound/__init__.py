"""Inbound adapters - implementations for external callers.

Inbound adapters expose the vector database to the outside world
via REST API, gRPC, CLI, or other interfaces.
"""

from vector_db.adapters.inbound.rest_api import create_app

__all__ = ["create_app"]
