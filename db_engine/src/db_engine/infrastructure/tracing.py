"""OpenTelemetry tracing configuration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str = "db_engine",
    otlp_endpoint: str | None = None,
    console_export: bool = False,
) -> trace.Tracer:
    """
    Set up OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
        console_export: Whether to also export to console (for debugging)

    Returns:
        Configured tracer instance
    """
    global _tracer

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter for debugging
    if console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _tracer = trace.get_tracer(service_name)

    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("db_engine")
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """
    Context manager for creating a trace span.

    Args:
        name: Name of the span
        attributes: Optional attributes to add to the span

    Yields:
        The created span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def trace_function(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """
    Decorator for tracing a function.

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional attributes to add to the span
    """
    def decorator(func):
        span_name = name or func.__name__

        def wrapper(*args, **kwargs):
            with trace_span(span_name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator
