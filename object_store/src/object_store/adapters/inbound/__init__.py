"""Inbound adapters for the Object Store.

Provides REST API adapter for object storage with versioning and erasure coding.
"""

try:
    from object_store.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
