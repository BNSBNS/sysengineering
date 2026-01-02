"""Prometheus metrics for Object Store."""

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry, REGISTRY


class ObjectStoreMetrics:
    """Metrics collector for the object store."""

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        # Object Operations
        self.objects_created = Counter(
            "object_store_objects_created_total",
            "Total objects created",
            ["bucket"],
            registry=registry,
        )
        self.objects_deleted = Counter(
            "object_store_objects_deleted_total",
            "Total objects deleted",
            ["bucket"],
            registry=registry,
        )
        self.objects_read = Counter(
            "object_store_objects_read_total",
            "Total objects read",
            ["bucket"],
            registry=registry,
        )
        self.bytes_uploaded = Counter(
            "object_store_bytes_uploaded_total",
            "Total bytes uploaded",
            ["bucket"],
            registry=registry,
        )
        self.bytes_downloaded = Counter(
            "object_store_bytes_downloaded_total",
            "Total bytes downloaded",
            ["bucket"],
            registry=registry,
        )

        # Multipart Upload
        self.multipart_uploads_started = Counter(
            "object_store_multipart_uploads_started_total",
            "Total multipart uploads initiated",
            ["bucket"],
            registry=registry,
        )
        self.multipart_uploads_completed = Counter(
            "object_store_multipart_uploads_completed_total",
            "Total multipart uploads completed",
            ["bucket"],
            registry=registry,
        )
        self.multipart_uploads_aborted = Counter(
            "object_store_multipart_uploads_aborted_total",
            "Total multipart uploads aborted",
            ["bucket"],
            registry=registry,
        )
        self.multipart_parts_uploaded = Counter(
            "object_store_multipart_parts_uploaded_total",
            "Total parts uploaded in multipart uploads",
            ["bucket"],
            registry=registry,
        )

        # Storage Metrics
        self.total_objects = Gauge(
            "object_store_total_objects",
            "Total number of objects stored",
            ["bucket"],
            registry=registry,
        )
        self.total_bytes = Gauge(
            "object_store_total_bytes",
            "Total bytes stored",
            ["bucket"],
            registry=registry,
        )
        self.buckets_count = Gauge(
            "object_store_buckets_count",
            "Total number of buckets",
            registry=registry,
        )

        # Deduplication Metrics
        self.dedup_bytes_saved = Counter(
            "object_store_dedup_bytes_saved_total",
            "Total bytes saved through deduplication",
            registry=registry,
        )
        self.dedup_ratio = Gauge(
            "object_store_dedup_ratio",
            "Current deduplication ratio",
            registry=registry,
        )
        self.chunks_deduplicated = Counter(
            "object_store_chunks_deduplicated_total",
            "Total chunks deduplicated",
            registry=registry,
        )

        # Erasure Coding Metrics
        self.ec_encode_operations = Counter(
            "object_store_ec_encode_operations_total",
            "Total erasure coding encode operations",
            registry=registry,
        )
        self.ec_decode_operations = Counter(
            "object_store_ec_decode_operations_total",
            "Total erasure coding decode operations",
            registry=registry,
        )
        self.ec_reconstruction_operations = Counter(
            "object_store_ec_reconstruction_total",
            "Total shard reconstruction operations",
            ["reason"],
            registry=registry,
        )
        self.ec_encode_latency = Histogram(
            "object_store_ec_encode_latency_seconds",
            "Erasure coding encode latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=registry,
        )

        # Merkle Tree Metrics
        self.merkle_verifications = Counter(
            "object_store_merkle_verifications_total",
            "Total Merkle tree verifications",
            ["result"],
            registry=registry,
        )
        self.merkle_computation_time = Histogram(
            "object_store_merkle_computation_seconds",
            "Time to compute Merkle tree",
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=registry,
        )

        # API Latency
        self.put_object_latency = Histogram(
            "object_store_put_object_latency_seconds",
            "PUT object request latency",
            ["bucket"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=registry,
        )
        self.get_object_latency = Histogram(
            "object_store_get_object_latency_seconds",
            "GET object request latency",
            ["bucket"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=registry,
        )
        self.delete_object_latency = Histogram(
            "object_store_delete_object_latency_seconds",
            "DELETE object request latency",
            ["bucket"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5],
            registry=registry,
        )
        self.list_objects_latency = Histogram(
            "object_store_list_objects_latency_seconds",
            "LIST objects request latency",
            ["bucket"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
            registry=registry,
        )

        # Error Metrics
        self.request_errors = Counter(
            "object_store_request_errors_total",
            "Total request errors",
            ["operation", "error_type"],
            registry=registry,
        )
        self.integrity_errors = Counter(
            "object_store_integrity_errors_total",
            "Total data integrity errors",
            ["error_type"],
            registry=registry,
        )

        # Connection Metrics
        self.active_connections = Gauge(
            "object_store_active_connections",
            "Number of active client connections",
            registry=registry,
        )
        self.requests_in_flight = Gauge(
            "object_store_requests_in_flight",
            "Number of requests currently being processed",
            ["operation"],
            registry=registry,
        )

        # System Info
        self.system_info = Info(
            "object_store",
            "Object store system information",
            registry=registry,
        )


_metrics: ObjectStoreMetrics | None = None


def get_metrics() -> ObjectStoreMetrics:
    """Get the singleton metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = ObjectStoreMetrics()
    return _metrics
