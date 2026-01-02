"""OpenTelemetry tracing for GPU platform.

Provides distributed tracing for job lifecycle tracking and performance analysis.
Exports traces to Jaeger for visualization.

References:
    - design.md Section 7 (Observability)
    - OpenTelemetry Python SDK
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

if TYPE_CHECKING:
    from gpu_platform.domain.entities.job import Job


class OpenTelemetryTracer:
    """Distributed tracing for GPU platform using OpenTelemetry."""
    
    def __init__(
        self,
        service_name: str = "gpu-platform",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
    ):
        """Initialize OpenTelemetry tracer.
        
        Args:
            service_name: Name of the service for traces.
            jaeger_host: Jaeger agent host.
            jaeger_port: Jaeger agent port.
        """
        if not OTEL_AVAILABLE:
            self.tracer = None
            return
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )
        
        # Create tracer provider and add exporter
        trace_provider = TracerProvider()
        trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        trace.set_tracer_provider(trace_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        self.service_name = service_name
    
    @contextmanager
    def trace_job_submit(self, job_id: str) -> Generator[None, None, None]:
        """Trace job submission.
        
        Args:
            job_id: Job identifier.
            
        Yields:
            Span context.
        """
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(
            f"job.submit",
            attributes={
                "job.id": job_id,
                "service.name": self.service_name,
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_job_schedule(self, job_id: str) -> Generator[None, None, None]:
        """Trace job scheduling.
        
        Args:
            job_id: Job identifier.
            
        Yields:
            Span context.
        """
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(
            f"job.schedule",
            attributes={
                "job.id": job_id,
                "service.name": self.service_name,
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_job_run(self, job_id: str, gpu_ids: list[str]) -> Generator[None, None, None]:
        """Trace job execution.
        
        Args:
            job_id: Job identifier.
            gpu_ids: GPUs allocated to job.
            
        Yields:
            Span context.
        """
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(
            f"job.run",
            attributes={
                "job.id": job_id,
                "job.gpu_count": len(gpu_ids),
                "job.gpus": ",".join(gpu_ids),
                "service.name": self.service_name,
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_placement_decision(self, job_id: str) -> Generator[None, None, None]:
        """Trace GPU placement decision.
        
        Args:
            job_id: Job identifier.
            
        Yields:
            Span context.
        """
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(
            f"scheduler.placement",
            attributes={
                "job.id": job_id,
                "service.name": self.service_name,
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_health_check(self, gpu_id: str) -> Generator[None, None, None]:
        """Trace GPU health check.
        
        Args:
            gpu_id: GPU identifier.
            
        Yields:
            Span context.
        """
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(
            f"health.check",
            attributes={
                "gpu.id": gpu_id,
                "service.name": self.service_name,
            }
        ) as span:
            yield span
    
    def record_alert(self, alert_type: str, gpu_id: str, severity: str) -> None:
        """Record an alert event.
        
        Args:
            alert_type: Type of alert.
            gpu_id: GPU affected.
            severity: Alert severity.
        """
        if not self.tracer:
            return
        
        with self.tracer.start_as_current_span(
            f"alert.{alert_type}",
            attributes={
                "alert.type": alert_type,
                "alert.severity": severity,
                "gpu.id": gpu_id,
                "service.name": self.service_name,
            }
        ):
            pass
