"""Inbound adapters for the Secure Platform.

Provides REST API adapter for certificate management, authorization, and audit.
"""

try:
    from secure_platform.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
