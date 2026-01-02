"""Prometheus metrics export for GPU platform monitoring.

Exports GPU health metrics and scheduler statistics to Prometheus format
for time-series collection and analysis.

References:
    - design.md Section 7 (Observability)
    - Prometheus metrics best practices
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if TYPE_CHECKING:
    from gpu_platform.domain.entities.gpu_device import GPUDevice
    from gpu_platform.domain.entities.job import Job
    from gpu_platform.domain.services.scheduler import GPUScheduler
    from gpu_platform.domain.services.health_monitor import HealthMonitor


class PrometheusExporter:
    """Export GPU platform metrics to Prometheus."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus exporter.
        
        Args:
            registry: Prometheus collector registry. Creates default if None.
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus-client not installed. Install with: pip install prometheus-client")
        
        self.registry = registry or CollectorRegistry()
        
        # GPU Metrics
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU die temperature in Celsius',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_power = Gauge(
            'gpu_power_watts',
            'GPU power consumption in watts',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_mb',
            'GPU memory used in MB',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_mb',
            'GPU total memory in MB',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_ecc_errors_correctable = Counter(
            'gpu_ecc_errors_correctable_total',
            'Total correctable ECC errors',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_ecc_errors_uncorrectable = Counter(
            'gpu_ecc_errors_uncorrectable_total',
            'Total uncorrectable ECC errors',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        self.gpu_state = Gauge(
            'gpu_state',
            'GPU state (0=AVAILABLE, 1=ALLOCATED, 2=IN_USE, 3=UNHEALTHY, 4=RESET, 5=OFFLINE)',
            ['gpu_id', 'gpu_model'],
            registry=self.registry,
        )
        
        # Job Metrics
        self.jobs_submitted = Counter(
            'jobs_submitted_total',
            'Total jobs submitted',
            registry=self.registry,
        )
        
        self.jobs_scheduled = Counter(
            'jobs_scheduled_total',
            'Total jobs scheduled',
            registry=self.registry,
        )
        
        self.jobs_completed = Counter(
            'jobs_completed_total',
            'Total jobs completed',
            registry=self.registry,
        )
        
        self.jobs_failed = Counter(
            'jobs_failed_total',
            'Total jobs failed',
            registry=self.registry,
        )
        
        self.jobs_pending = Gauge(
            'jobs_pending',
            'Current pending jobs in queue',
            registry=self.registry,
        )
        
        self.job_duration = Histogram(
            'job_duration_seconds',
            'Job execution duration in seconds',
            registry=self.registry,
        )
        
        # Cluster Metrics
        self.cluster_gpus_total = Gauge(
            'cluster_gpus_total',
            'Total GPUs in cluster',
            registry=self.registry,
        )
        
        self.cluster_gpus_available = Gauge(
            'cluster_gpus_available',
            'Available GPUs in cluster',
            registry=self.registry,
        )
        
        self.cluster_gpus_allocated = Gauge(
            'cluster_gpus_allocated',
            'Allocated GPUs in cluster',
            registry=self.registry,
        )
        
        self.cluster_gpus_unhealthy = Gauge(
            'cluster_gpus_unhealthy',
            'Unhealthy GPUs in cluster',
            registry=self.registry,
        )
        
        # Alert Metrics
        self.alerts_triggered = Counter(
            'alerts_triggered_total',
            'Total alerts triggered',
            ['alert_type', 'severity'],
            registry=self.registry,
        )
        
        self.alerts_active = Gauge(
            'alerts_active',
            'Currently active alerts',
            ['alert_type', 'severity'],
            registry=self.registry,
        )
    
    def update_gpu_metrics(self, device: GPUDevice) -> None:
        """Update metrics for a GPU device.
        
        Args:
            device: GPU device with current metrics.
        """
        gpu_id = str(device.specs.gpu_id)
        gpu_model = device.specs.model
        labels = {'gpu_id': gpu_id, 'gpu_model': gpu_model}
        
        # Update gauge metrics
        self.gpu_memory_total.labels(**labels).set(device.specs.memory_mb)
        self.gpu_state.labels(**labels).set(device.state.value)
        
        # Update health metrics if available
        if device.health:
            self.gpu_temperature.labels(**labels).set(device.health.temperature_c)
            self.gpu_power.labels(**labels).set(device.health.power_w)
            self.gpu_utilization.labels(**labels).set(device.health.utilization_percent)
            self.gpu_memory_used.labels(**labels).set(device.health.memory_used_mb)
            
            # Update counters (cumulative)
            if device.health.ecc_errors_correctable > 0:
                self.gpu_ecc_errors_correctable.labels(**labels).inc(
                    device.health.ecc_errors_correctable
                )
            if device.health.ecc_errors_uncorrectable > 0:
                self.gpu_ecc_errors_uncorrectable.labels(**labels).inc(
                    device.health.ecc_errors_uncorrectable
                )
    
    def update_scheduler_metrics(self, scheduler: GPUScheduler) -> None:
        """Update scheduler statistics metrics.
        
        Args:
            scheduler: GPU scheduler with statistics.
        """
        stats = scheduler.get_stats()
        
        self.cluster_gpus_total.set(stats.get('total_gpus', 0))
        self.cluster_gpus_available.set(stats.get('available_gpus', 0))
        self.cluster_gpus_allocated.set(stats.get('allocated_gpus', 0))
        self.cluster_gpus_unhealthy.set(stats.get('unhealthy_gpus', 0))
        self.jobs_pending.set(stats.get('pending_jobs', 0))
    
    def record_job_submission(self) -> None:
        """Record a job submission."""
        self.jobs_submitted.inc()
    
    def record_job_scheduled(self) -> None:
        """Record a job being scheduled."""
        self.jobs_scheduled.inc()
    
    def record_job_completed(self, duration_seconds: float) -> None:
        """Record a job completion.
        
        Args:
            duration_seconds: Job execution duration.
        """
        self.jobs_completed.inc()
        self.job_duration.observe(duration_seconds)
    
    def record_job_failed(self) -> None:
        """Record a job failure."""
        self.jobs_failed.inc()
    
    def record_alert(self, alert_type: str, severity: str) -> None:
        """Record an alert being triggered.
        
        Args:
            alert_type: Type of alert (e.g., 'thermal_warning').
            severity: Severity level (e.g., 'critical').
        """
        labels = {'alert_type': alert_type, 'severity': severity}
        self.alerts_triggered.labels(**labels).inc()
        self.alerts_active.labels(**labels).inc()
    
    def clear_alert(self, alert_type: str, severity: str) -> None:
        """Record an alert being cleared.
        
        Args:
            alert_type: Type of alert.
            severity: Severity level.
        """
        labels = {'alert_type': alert_type, 'severity': severity}
        current = self.alerts_active.labels(**labels)._value.get()
        if current > 0:
            self.alerts_active.labels(**labels).set(current - 1)
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus text format.
        
        Returns:
            Prometheus text format metrics.
        """
        from prometheus_client import generate_latest, CollectorRegistry
        
        return generate_latest(self.registry).decode('utf-8')
