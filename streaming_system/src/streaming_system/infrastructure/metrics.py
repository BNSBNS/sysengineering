"""Prometheus metrics for Distributed Streaming System."""

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry, REGISTRY


class StreamingSystemMetrics:
    """Metrics collector for the streaming system."""

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        # Partition Log Metrics
        self.records_appended = Counter(
            "streaming_records_appended_total",
            "Total number of records appended to partitions",
            ["topic", "partition"],
            registry=registry,
        )
        self.records_read = Counter(
            "streaming_records_read_total",
            "Total number of records read from partitions",
            ["topic", "partition", "consumer_group"],
            registry=registry,
        )
        self.bytes_written = Counter(
            "streaming_bytes_written_total",
            "Total bytes written to partitions",
            ["topic", "partition"],
            registry=registry,
        )
        self.bytes_read = Counter(
            "streaming_bytes_read_total",
            "Total bytes read from partitions",
            ["topic", "partition"],
            registry=registry,
        )

        # Segment Metrics
        self.active_segments = Gauge(
            "streaming_active_segments",
            "Number of active log segments",
            ["topic", "partition"],
            registry=registry,
        )
        self.segment_size_bytes = Gauge(
            "streaming_segment_size_bytes",
            "Current segment size in bytes",
            ["topic", "partition", "segment_id"],
            registry=registry,
        )
        self.segment_rolls = Counter(
            "streaming_segment_rolls_total",
            "Number of segment roll operations",
            ["topic", "partition"],
            registry=registry,
        )

        # Offset Metrics
        self.log_end_offset = Gauge(
            "streaming_log_end_offset",
            "Current log end offset (latest)",
            ["topic", "partition"],
            registry=registry,
        )
        self.log_start_offset = Gauge(
            "streaming_log_start_offset",
            "Current log start offset (earliest)",
            ["topic", "partition"],
            registry=registry,
        )
        self.high_watermark = Gauge(
            "streaming_high_watermark",
            "High watermark (last committed offset)",
            ["topic", "partition"],
            registry=registry,
        )
        self.consumer_lag = Gauge(
            "streaming_consumer_lag",
            "Consumer group lag behind log end",
            ["topic", "partition", "consumer_group"],
            registry=registry,
        )

        # Raft Consensus Metrics
        self.raft_term = Gauge(
            "streaming_raft_term",
            "Current Raft term",
            ["partition"],
            registry=registry,
        )
        self.raft_state = Gauge(
            "streaming_raft_state",
            "Raft state (0=follower, 1=candidate, 2=leader)",
            ["partition"],
            registry=registry,
        )
        self.raft_elections = Counter(
            "streaming_raft_elections_total",
            "Total number of Raft elections",
            ["partition", "result"],
            registry=registry,
        )
        self.raft_election_duration = Histogram(
            "streaming_raft_election_duration_seconds",
            "Time to complete leader election",
            ["partition"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=registry,
        )
        self.raft_heartbeats_sent = Counter(
            "streaming_raft_heartbeats_sent_total",
            "Total heartbeats sent by leader",
            ["partition"],
            registry=registry,
        )
        self.raft_heartbeats_received = Counter(
            "streaming_raft_heartbeats_received_total",
            "Total heartbeats received by followers",
            ["partition"],
            registry=registry,
        )

        # ISR Metrics
        self.isr_size = Gauge(
            "streaming_isr_size",
            "Number of replicas in ISR",
            ["topic", "partition"],
            registry=registry,
        )
        self.isr_shrinks = Counter(
            "streaming_isr_shrinks_total",
            "Total ISR shrink events",
            ["topic", "partition"],
            registry=registry,
        )
        self.isr_expands = Counter(
            "streaming_isr_expands_total",
            "Total ISR expand events",
            ["topic", "partition"],
            registry=registry,
        )
        self.under_replicated_partitions = Gauge(
            "streaming_under_replicated_partitions",
            "Number of under-replicated partitions",
            registry=registry,
        )

        # Replication Metrics
        self.replication_lag_bytes = Gauge(
            "streaming_replication_lag_bytes",
            "Replication lag in bytes",
            ["topic", "partition", "replica"],
            registry=registry,
        )
        self.replication_lag_ms = Gauge(
            "streaming_replication_lag_ms",
            "Replication lag in milliseconds",
            ["topic", "partition", "replica"],
            registry=registry,
        )

        # Consumer Group Metrics
        self.consumer_groups_active = Gauge(
            "streaming_consumer_groups_active",
            "Number of active consumer groups",
            registry=registry,
        )
        self.consumers_active = Gauge(
            "streaming_consumers_active",
            "Number of active consumers",
            ["consumer_group"],
            registry=registry,
        )
        self.rebalances = Counter(
            "streaming_rebalances_total",
            "Total consumer group rebalances",
            ["consumer_group", "reason"],
            registry=registry,
        )
        self.rebalance_duration = Histogram(
            "streaming_rebalance_duration_seconds",
            "Consumer group rebalance duration",
            ["consumer_group"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
            registry=registry,
        )
        self.commit_offset_latency = Histogram(
            "streaming_commit_offset_latency_seconds",
            "Offset commit latency",
            ["consumer_group"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=registry,
        )

        # Producer Metrics
        self.produce_requests = Counter(
            "streaming_produce_requests_total",
            "Total produce requests",
            ["topic", "ack_mode"],
            registry=registry,
        )
        self.produce_latency = Histogram(
            "streaming_produce_latency_seconds",
            "Produce request latency",
            ["topic", "ack_mode"],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
            registry=registry,
        )
        self.produce_errors = Counter(
            "streaming_produce_errors_total",
            "Total produce errors",
            ["topic", "error_type"],
            registry=registry,
        )

        # Fetch/Consume Metrics
        self.fetch_requests = Counter(
            "streaming_fetch_requests_total",
            "Total fetch requests",
            ["topic", "consumer_group"],
            registry=registry,
        )
        self.fetch_latency = Histogram(
            "streaming_fetch_latency_seconds",
            "Fetch request latency",
            ["topic"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=registry,
        )

        # Network Metrics
        self.network_requests = Counter(
            "streaming_network_requests_total",
            "Total network requests",
            ["request_type"],
            registry=registry,
        )
        self.network_errors = Counter(
            "streaming_network_errors_total",
            "Total network errors",
            ["error_type"],
            registry=registry,
        )

        # System Info
        self.system_info = Info(
            "streaming_system",
            "Streaming system information",
            registry=registry,
        )


_metrics: StreamingSystemMetrics | None = None


def get_metrics() -> StreamingSystemMetrics:
    """Get the singleton metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = StreamingSystemMetrics()
    return _metrics
