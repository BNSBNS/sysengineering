"""Prometheus metrics for the container runtime."""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
    CollectorRegistry,
)


class MetricsRegistry:
    """Registry of all container runtime metrics."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics registry."""
        self._registry = registry or REGISTRY

        # Container metrics
        self.container_count = Gauge(
            "container_count",
            "Number of containers by state",
            ["state"],  # running, stopped, failed, creating
            registry=self._registry,
        )

        self.container_startup_seconds = Histogram(
            "container_startup_seconds",
            "Container startup latency in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self._registry,
        )

        self.container_operations_total = Counter(
            "container_operations_total",
            "Total container operations",
            ["operation", "status"],  # create/start/stop/delete, success/error
            registry=self._registry,
        )

        # Scheduler metrics
        self.scheduler_queue_depth = Gauge(
            "scheduler_queue_depth",
            "Number of pending jobs in scheduler queue",
            registry=self._registry,
        )

        self.scheduler_placement_latency_seconds = Histogram(
            "scheduler_placement_latency_seconds",
            "Time to make scheduling decision",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self._registry,
        )

        self.scheduler_placements_total = Counter(
            "scheduler_placements_total",
            "Total scheduling placements",
            ["status"],  # success, failed, timeout
            registry=self._registry,
        )

        # Resource metrics
        self.resource_cpu_usage_percent = Gauge(
            "resource_cpu_usage_percent",
            "CPU usage percentage per container",
            ["container_id"],
            registry=self._registry,
        )

        self.resource_memory_bytes = Gauge(
            "resource_memory_bytes",
            "Memory usage in bytes per container",
            ["container_id"],
            registry=self._registry,
        )

        self.resource_memory_limit_bytes = Gauge(
            "resource_memory_limit_bytes",
            "Memory limit in bytes per container",
            ["container_id"],
            registry=self._registry,
        )

        # GPU metrics
        self.resource_gpu_count = Gauge(
            "resource_gpu_count",
            "Number of GPUs by state",
            ["state"],  # available, allocated, failed
            registry=self._registry,
        )

        self.resource_gpu_utilization_percent = Gauge(
            "resource_gpu_utilization_percent",
            "GPU utilization per device",
            ["gpu_id", "container_id"],
            registry=self._registry,
        )

        self.resource_gpu_memory_bytes = Gauge(
            "resource_gpu_memory_bytes",
            "GPU memory usage per device",
            ["gpu_id", "container_id"],
            registry=self._registry,
        )

        # Image metrics
        self.image_pull_duration_seconds = Histogram(
            "image_pull_duration_seconds",
            "Image pull duration in seconds",
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self._registry,
        )

        self.image_pulls_total = Counter(
            "image_pulls_total",
            "Total image pull operations",
            ["status"],  # success, failed
            registry=self._registry,
        )

        # OOM events
        self.oom_events_total = Counter(
            "oom_events_total",
            "Total OOM kill events",
            registry=self._registry,
        )

        # Server info
        self.info = Info(
            "container_runtime",
            "Container runtime information",
            registry=self._registry,
        )


_metrics: MetricsRegistry | None = None


def setup_metrics(port: int = 8002, registry: CollectorRegistry | None = None) -> MetricsRegistry:
    """Set up Prometheus metrics server."""
    global _metrics
    _metrics = MetricsRegistry(registry)

    from container_runtime import __version__
    _metrics.info.info({"version": __version__})

    start_http_server(port, registry=registry or REGISTRY)
    return _metrics


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics
