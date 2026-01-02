"""GPU Platform Application Coordinator.

Orchestrates domain services to provide the main GPU platform functionality.
Implements the GPUPlatformAPI by coordinating discovery, scheduling,
allocation, health monitoring, and observability.

References:
    - design.md Section 2 (Architecture)
    - design.md Section 7 (Observability)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from gpu_platform.adapters.outbound.metrics import PrometheusExporter
from gpu_platform.adapters.outbound.tracing import OpenTelemetryTracer
from gpu_platform.domain.entities.gpu_device import GPUDevice, GPUState
from gpu_platform.domain.entities.job import Job, JobState
from gpu_platform.domain.entities.topology import Topology
from gpu_platform.domain.services.gpu_discovery import GPUDiscoveryService
from gpu_platform.domain.services.health_monitor import AlertManager, HealthMonitor
from gpu_platform.domain.services.health_predictor import PredictiveHealthAnalyzer
from gpu_platform.domain.services.scheduler import GPUScheduler
from gpu_platform.domain.value_objects.gpu_identifiers import JobId
from gpu_platform.ports.inbound.api import GPUPlatformAPI

logger = logging.getLogger(__name__)


class GPUPlatformCoordinator:
    """Coordinates GPU platform operations with full observability."""

    def __init__(
        self,
        topology: Topology,
        discovery: GPUDiscoveryService,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
    ) -> None:
        """Initialize the coordinator.
        
        Args:
            topology: Cluster topology.
            discovery: GPU discovery service.
            enable_metrics: Enable Prometheus metrics export.
            enable_tracing: Enable OpenTelemetry tracing.
        """
        self._topology = topology
        self._discovery = discovery
        
        # Core services
        self._health_monitor = HealthMonitor()
        self._alert_manager = AlertManager(self._health_monitor)
        self._health_predictor = PredictiveHealthAnalyzer()
        self._scheduler = GPUScheduler(topology, self._health_monitor)
        
        # Observability
        self._metrics: Optional[PrometheusExporter] = None
        self._tracer: Optional[OpenTelemetryTracer] = None
        
        if enable_metrics:
            try:
                self._metrics = PrometheusExporter()
            except ImportError:
                logger.warning("Prometheus metrics not available")
        
        if enable_tracing:
            self._tracer = OpenTelemetryTracer()
        
        self._initialized = False
        self._last_health_check = 0
    
    def initialize(self) -> None:
        """Initialize the platform.
        
        Discovers GPUs, creates devices, registers with scheduler,
        and sets up health monitoring.
        """
        try:
            # Discover GPUs
            specs = self._discovery.scan_gpus()
            logger.info(f"Discovered {len(specs)} GPUs")
            
            # Create device objects and register with scheduler
            for spec in specs:
                health = self._discovery.get_health(spec.gpu_id)
                
                # Determine NUMA node
                node_id = self._topology.get_node_for_gpu(spec.gpu_id)
                if node_id is None:
                    node_id = 0  # Default to node 0
                
                device = GPUDevice(
                    specs=spec,
                    numa_node=node_id,
                    state=GPUState.AVAILABLE if health and health.is_healthy else GPUState.UNHEALTHY,
                    health=health,
                )
                
                self._scheduler.register_gpu(device)
                
                # Update metrics
                if self._metrics:
                    self._metrics.update_gpu_metrics(device)
            
            self._initialized = True
            logger.info("GPU Platform initialized successfully")
        
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def submit_job(self, job: Job) -> JobId:
        """Submit a job to the platform.
        
        Args:
            job: Job to submit.
            
        Returns:
            JobId assigned to the job.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")
        
        # Record submission
        if self._metrics:
            self._metrics.record_job_submission()
        
        if self._tracer:
            with self._tracer.trace_job_submit(str(job.job_id)):
                self._scheduler.submit_job(job)
        else:
            self._scheduler.submit_job(job)
        
        # Try to schedule it immediately
        decision = self._scheduler.schedule_pending()
        if not decision.success:
            logger.info(f"Job {job.job_id} queued (reason: {decision.reason})")
        else:
            if self._metrics:
                self._metrics.record_job_scheduled()
        
        return job.job_id
    
    def cancel_job(self, job_id: JobId) -> bool:
        """Cancel a job.

        Args:
            job_id: Job to cancel.

        Returns:
            True if cancelled, False if not found or already completed.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")

        result = self._scheduler.cancel_job(job_id)
        if result and self._metrics:
            self._metrics.record_job_cancelled()
        return result

    def get_job_status(self, job_id: JobId) -> Optional[Job]:
        """Get job status.

        Args:
            job_id: Job to query.

        Returns:
            Job with current state or None if not found.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")

        return self._scheduler.get_job(job_id)

    def start_job(self, job_id: JobId) -> bool:
        """Transition a scheduled job to running state.

        Should be called when the job actually starts executing.

        Args:
            job_id: Job to start.

        Returns:
            True if started, False if not found or invalid state.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")

        result = self._scheduler.start_job(job_id)
        if result and self._metrics:
            self._metrics.record_job_started()
        return result

    def complete_job(self, job_id: JobId) -> bool:
        """Complete a running job and release resources.

        Args:
            job_id: Job to complete.

        Returns:
            True if completed, False if not found.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")

        result = self._scheduler.release_job(job_id)
        if result and self._metrics:
            self._metrics.record_job_completed()
        return result

    def fail_job(self, job_id: JobId, reason: str = "") -> bool:
        """Mark a job as failed.

        Args:
            job_id: Job that failed.
            reason: Failure reason.

        Returns:
            True if marked failed, False if not found.
        """
        if not self._initialized:
            raise RuntimeError("Platform not initialized")

        result = self._scheduler.fail_job(job_id, reason)
        if result and self._metrics:
            self._metrics.record_job_failed()
        return result

    def get_cluster_topology(self) -> Topology:
        """Get cluster topology.
        
        Returns:
            Current topology.
        """
        return self._topology
    
    def get_cluster_stats(self) -> dict:
        """Get cluster statistics.
        
        Returns:
            Stats dictionary.
        """
        return self._scheduler.get_stats()
