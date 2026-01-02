"""Inbound adapters for the GPU Platform.

Provides REST API adapter for job submission and cluster management.
"""

try:
    from gpu_platform.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
