"""Inbound adapters for the Container Runtime.

Provides REST API adapter for container and job management.
"""

try:
    from container_runtime.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
