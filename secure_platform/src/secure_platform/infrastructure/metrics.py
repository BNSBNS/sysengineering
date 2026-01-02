"""Prometheus metrics for the secure platform."""

from __future__ import annotations

from prometheus_client import (
    Counter, Gauge, Histogram, Info, start_http_server, REGISTRY, CollectorRegistry,
)


class MetricsRegistry:
    """Registry of all secure platform metrics."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or REGISTRY

        # Authentication metrics
        self.auth_requests_total = Counter(
            "auth_requests_total", "Authentication requests", ["status"],
            registry=self._registry,
        )

        self.auth_latency_seconds = Histogram(
            "auth_latency_seconds", "Authentication latency",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
            registry=self._registry,
        )

        # Authorization metrics
        self.authz_decisions_total = Counter(
            "authz_decisions_total", "Authorization decisions", ["decision"],
            registry=self._registry,
        )

        self.policy_eval_latency_seconds = Histogram(
            "policy_eval_latency_seconds", "Policy evaluation latency",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01),
            registry=self._registry,
        )

        # Certificate metrics
        self.cert_rotation_duration_seconds = Histogram(
            "cert_rotation_duration_seconds", "Certificate rotation duration",
            buckets=(1, 5, 10, 15, 20, 25, 30),
            registry=self._registry,
        )

        self.cert_expiry_seconds = Gauge(
            "cert_expiry_seconds", "Seconds until certificate expiry", ["subject"],
            registry=self._registry,
        )

        self.cert_rotations_total = Counter(
            "cert_rotations_total", "Certificate rotations", ["status"],
            registry=self._registry,
        )

        # Audit metrics
        self.audit_records_total = Counter(
            "audit_records_total", "Audit records written",
            registry=self._registry,
        )

        self.audit_verification_failures_total = Counter(
            "audit_verification_failures_total", "Audit tamper detection events",
            registry=self._registry,
        )

        self.info = Info("secure_platform", "Platform information", registry=self._registry)


_metrics: MetricsRegistry | None = None


def setup_metrics(port: int = 8003, registry: CollectorRegistry | None = None) -> MetricsRegistry:
    """Set up Prometheus metrics."""
    global _metrics
    _metrics = MetricsRegistry(registry)
    from secure_platform import __version__
    _metrics.info.info({"version": __version__})
    start_http_server(port, registry=registry or REGISTRY)
    return _metrics


def get_metrics() -> MetricsRegistry:
    """Get the metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics
