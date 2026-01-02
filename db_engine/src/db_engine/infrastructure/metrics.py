"""Prometheus metrics for the database engine."""

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
    """Registry of all database engine metrics."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics registry."""
        self._registry = registry or REGISTRY

        # Transaction metrics
        self.transactions_total = Counter(
            "db_transactions_total",
            "Total number of transactions",
            ["status"],  # commit, abort
            registry=self._registry,
        )

        self.transactions_active = Gauge(
            "db_transactions_active",
            "Number of active transactions",
            registry=self._registry,
        )

        # Query metrics
        self.query_latency_seconds = Histogram(
            "db_query_latency_seconds",
            "Query latency in seconds",
            ["query_type"],  # select, insert, update, delete
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        self.queries_total = Counter(
            "db_queries_total",
            "Total number of queries executed",
            ["query_type", "status"],  # status: success, error
            registry=self._registry,
        )

        # Buffer pool metrics
        self.buffer_pool_hits_total = Counter(
            "db_buffer_pool_hits_total",
            "Total buffer pool cache hits",
            registry=self._registry,
        )

        self.buffer_pool_misses_total = Counter(
            "db_buffer_pool_misses_total",
            "Total buffer pool cache misses",
            registry=self._registry,
        )

        self.buffer_pool_size_bytes = Gauge(
            "db_buffer_pool_size_bytes",
            "Current buffer pool size in bytes",
            registry=self._registry,
        )

        self.buffer_pool_dirty_pages = Gauge(
            "db_buffer_pool_dirty_pages",
            "Number of dirty pages in buffer pool",
            registry=self._registry,
        )

        # WAL metrics
        self.wal_bytes_written_total = Counter(
            "db_wal_bytes_written_total",
            "Total WAL bytes written",
            registry=self._registry,
        )

        self.wal_flushes_total = Counter(
            "db_wal_flushes_total",
            "Total WAL flush operations",
            registry=self._registry,
        )

        self.wal_flush_latency_seconds = Histogram(
            "db_wal_flush_latency_seconds",
            "WAL flush latency in seconds",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self._registry,
        )

        # Checkpoint metrics
        self.checkpoint_duration_seconds = Histogram(
            "db_checkpoint_duration_seconds",
            "Checkpoint duration in seconds",
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self._registry,
        )

        self.checkpoints_total = Counter(
            "db_checkpoints_total",
            "Total number of checkpoints",
            registry=self._registry,
        )

        # Lock metrics
        self.lock_wait_seconds = Histogram(
            "db_lock_wait_seconds",
            "Time spent waiting for locks",
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self._registry,
        )

        self.deadlocks_total = Counter(
            "db_deadlocks_total",
            "Total number of deadlocks detected",
            registry=self._registry,
        )

        # Index metrics
        self.index_lookups_total = Counter(
            "db_index_lookups_total",
            "Total index lookup operations",
            ["index_name"],
            registry=self._registry,
        )

        self.index_scans_total = Counter(
            "db_index_scans_total",
            "Total index scan operations",
            ["index_name"],
            registry=self._registry,
        )

        # Recovery metrics
        self.recovery_duration_seconds = Gauge(
            "db_recovery_duration_seconds",
            "Duration of last recovery in seconds",
            registry=self._registry,
        )

        self.recovery_records_replayed = Counter(
            "db_recovery_records_replayed_total",
            "Total WAL records replayed during recovery",
            registry=self._registry,
        )

        # Server info
        self.info = Info(
            "db_engine",
            "Database engine information",
            registry=self._registry,
        )


# Global metrics registry
_metrics: MetricsRegistry | None = None


def setup_metrics(port: int = 8001, registry: CollectorRegistry | None = None) -> MetricsRegistry:
    """
    Set up Prometheus metrics server.

    Args:
        port: Port for the metrics HTTP server
        registry: Optional custom registry

    Returns:
        The metrics registry
    """
    global _metrics
    _metrics = MetricsRegistry(registry)

    # Set server info
    from db_engine import __version__
    _metrics.info.info({
        "version": __version__,
    })

    # Start HTTP server for Prometheus scraping
    start_http_server(port, registry=registry or REGISTRY)

    return _metrics


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics
