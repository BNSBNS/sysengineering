"""Prometheus metrics for GPU platform."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server, REGISTRY, CollectorRegistry


class MetricsRegistry:
    """GPU platform metrics."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or REGISTRY

        self.gpu_count = Gauge("gpu_count", "GPUs by state", ["state"], registry=self._registry)
        self.gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", ["gpu_id"], registry=self._registry)
        self.gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used", ["gpu_id"], registry=self._registry)
        self.gpu_temperature_celsius = Gauge("gpu_temperature_celsius", "GPU temperature", ["gpu_id"], registry=self._registry)
        self.gpu_ecc_errors_total = Counter("gpu_ecc_errors_total", "ECC errors", ["gpu_id", "type"], registry=self._registry)

        self.job_count = Gauge("job_count", "Jobs by state", ["state"], registry=self._registry)
        self.job_queue_wait_seconds = Histogram("job_queue_wait_seconds", "Job queue wait time", buckets=(1, 5, 10, 30, 60, 120, 300), registry=self._registry)
        self.scheduler_latency_seconds = Histogram("scheduler_latency_seconds", "Scheduling latency", buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5), registry=self._registry)

        self.info = Info("gpu_platform", "Platform info", registry=self._registry)


_metrics: MetricsRegistry | None = None


def setup_metrics(port: int = 8004, registry: CollectorRegistry | None = None) -> MetricsRegistry:
    global _metrics
    _metrics = MetricsRegistry(registry)
    start_http_server(port, registry=registry or REGISTRY)
    return _metrics


def get_metrics() -> MetricsRegistry:
    global _metrics
    return _metrics or MetricsRegistry()
