"""Prometheus metrics for Security Agent."""

from prometheus_client import Counter, Gauge, Histogram, Info, Summary, CollectorRegistry, REGISTRY


class SecurityAgentMetrics:
    """Metrics collector for the security agent."""

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        # eBPF Probe Metrics
        self.syscalls_traced = Counter(
            "security_syscalls_traced_total",
            "Total syscalls traced by eBPF",
            ["syscall_name", "process_name"],
            registry=registry,
        )
        self.network_events = Counter(
            "security_network_events_total",
            "Total network events captured",
            ["direction", "protocol", "action"],
            registry=registry,
        )
        self.file_events = Counter(
            "security_file_events_total",
            "Total file events captured",
            ["operation", "path_prefix"],
            registry=registry,
        )
        self.ebpf_buffer_usage = Gauge(
            "security_ebpf_buffer_usage_percent",
            "eBPF ring buffer usage percentage",
            ["probe_type"],
            registry=registry,
        )
        self.ebpf_dropped_events = Counter(
            "security_ebpf_dropped_events_total",
            "Total events dropped due to buffer overflow",
            ["probe_type"],
            registry=registry,
        )

        # Detection Metrics
        self.threats_detected = Counter(
            "security_threats_detected_total",
            "Total threats detected",
            ["severity", "category", "detection_method"],
            registry=registry,
        )
        self.anomalies_detected = Counter(
            "security_anomalies_detected_total",
            "Total anomalies detected by ML",
            ["anomaly_type"],
            registry=registry,
        )
        self.detection_latency = Histogram(
            "security_detection_latency_seconds",
            "Time to detect and classify threat",
            ["detection_method"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=registry,
        )
        self.rules_evaluated = Counter(
            "security_rules_evaluated_total",
            "Total detection rules evaluated",
            ["rule_category"],
            registry=registry,
        )
        self.rules_matched = Counter(
            "security_rules_matched_total",
            "Total detection rules that matched",
            ["rule_id", "severity"],
            registry=registry,
        )
        self.false_positives = Counter(
            "security_false_positives_total",
            "Total false positives reported",
            ["category"],
            registry=registry,
        )

        # ML Model Metrics
        self.ml_predictions = Counter(
            "security_ml_predictions_total",
            "Total ML model predictions",
            ["model_name", "prediction"],
            registry=registry,
        )
        self.ml_prediction_latency = Histogram(
            "security_ml_prediction_latency_seconds",
            "ML model prediction latency",
            ["model_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=registry,
        )
        self.ml_model_accuracy = Gauge(
            "security_ml_model_accuracy",
            "Current ML model accuracy estimate",
            ["model_name"],
            registry=registry,
        )
        self.baseline_deviation = Histogram(
            "security_baseline_deviation",
            "Deviation from learned baseline",
            ["metric_type"],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0],
            registry=registry,
        )

        # Response Metrics
        self.responses_triggered = Counter(
            "security_responses_triggered_total",
            "Total automated responses triggered",
            ["response_type", "threat_severity"],
            registry=registry,
        )
        self.processes_killed = Counter(
            "security_processes_killed_total",
            "Total processes terminated by agent",
            ["reason"],
            registry=registry,
        )
        self.files_quarantined = Counter(
            "security_files_quarantined_total",
            "Total files quarantined",
            ["file_type"],
            registry=registry,
        )
        self.network_connections_blocked = Counter(
            "security_network_connections_blocked_total",
            "Total network connections blocked",
            ["reason"],
            registry=registry,
        )
        self.response_latency = Histogram(
            "security_response_latency_seconds",
            "Time from detection to response",
            ["response_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=registry,
        )

        # Alert Metrics
        self.alerts_generated = Counter(
            "security_alerts_generated_total",
            "Total alerts generated",
            ["severity", "category"],
            registry=registry,
        )
        self.alerts_forwarded = Counter(
            "security_alerts_forwarded_total",
            "Total alerts forwarded to SIEM",
            ["destination"],
            registry=registry,
        )
        self.alerts_deduplicated = Counter(
            "security_alerts_deduplicated_total",
            "Total alerts deduplicated",
            ["category"],
            registry=registry,
        )

        # Agent Health Metrics
        self.agent_uptime = Gauge(
            "security_agent_uptime_seconds",
            "Agent uptime in seconds",
            registry=registry,
        )
        self.monitored_processes = Gauge(
            "security_monitored_processes",
            "Number of processes being monitored",
            registry=registry,
        )
        self.monitored_connections = Gauge(
            "security_monitored_connections",
            "Number of network connections being monitored",
            registry=registry,
        )
        self.events_per_second = Gauge(
            "security_events_per_second",
            "Current event processing rate",
            ["event_type"],
            registry=registry,
        )

        # System Info
        self.system_info = Info(
            "security_agent",
            "Security agent information",
            registry=registry,
        )


_metrics: SecurityAgentMetrics | None = None


def get_metrics() -> SecurityAgentMetrics:
    """Get the singleton metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = SecurityAgentMetrics()
    return _metrics
