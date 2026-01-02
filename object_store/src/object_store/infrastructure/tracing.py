"""OpenTelemetry tracing configuration for Object Store."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from object_store.infrastructure.config import get_config


def setup_tracing() -> trace.Tracer:
    """Configure OpenTelemetry tracing for object store."""
    config = get_config()

    resource = Resource.create(
        {
            "service.name": "object_store",
            "service.version": "0.1.0",
            "deployment.environment": config.observability.environment,
        }
    )

    provider = TracerProvider(resource=resource)

    if config.observability.otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.observability.otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    return trace.get_tracer("object_store")


def get_tracer(name: str = "object_store") -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)
