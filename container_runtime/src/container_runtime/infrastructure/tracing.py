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
    service_name: str = "container_runtime",
    otlp_endpoint: str | None = None,
    console_export: bool = False,
) -> trace.Tracer:
    """Set up OpenTelemetry tracing."""
    global _tracer

    resource = Resource.create({"service.name": service_name, "service.version": "0.1.0"})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("container_runtime")
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[trace.Span, None, None]:
    """Context manager for creating a trace span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
