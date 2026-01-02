"""Inbound adapters for the Security Agent.

Provides REST API adapter for threat detection and response management.
"""

try:
    from security_agent.adapters.inbound.rest_api import create_app

    __all__ = ["create_app"]
except ImportError:
    # FastAPI not installed - rest_api unavailable
    __all__ = []
