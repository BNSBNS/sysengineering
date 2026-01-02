"""OpenTelemetry tracing configuration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "secure_platform", otlp_endpoint: str | None = None) -> trace.Tracer:
    """Set up OpenTelemetry tracing."""
    global _tracer
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the tracer."""
    global _tracer
    return _tracer or trace.get_tracer("secure_platform")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[trace.Span, None, None]:
    """Create a trace span."""
    with get_tracer().start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
