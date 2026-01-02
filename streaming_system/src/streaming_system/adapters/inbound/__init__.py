"""Inbound adapters for the Streaming System.

Provides REST API adapter for message streaming with Raft consensus.
"""

try:
    from streaming_system.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
