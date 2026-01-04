"""Integration tests for GPU platform job scheduling."""

import pytest
import time

from gpu_platform.domain.entities.gpu_device import GPUDevice, GPUHealth, GPUSpecs, GPUState
from gpu_platform.domain.entities.job import Job, JobPriority, JobState, GPURequest, Placement
from gpu_platform.domain.entities.topology import NUMANode, NVLink, Topology
from gpu_platform.domain.services.gpu_discovery import GPUDiscoveryService
from gpu_platform.domain.services.scheduler import GPUScheduler, ScheduleDecision
from gpu_platform.domain.services.health_monitor import HealthMonitor
from gpu_platform.domain.value_objects.gpu_identifiers import (
    GPUId,
    JobId,
    NUMANodeId,
    PCIeBusId,
    create_gpu_id,
    create_job_id,
)


def create_mock_topology(num_gpus: int = 4, num_nodes: int = 2) -> Topology:
    """Create a mock topology for testing."""
    gpus_per_node = num_gpus // num_nodes
    nodes = {}

    for node_id in range(num_nodes):
        gpu_ids = [
            create_gpu_id(node_id * gpus_per_node + i)
            for i in range(gpus_per_node)
        ]
        nodes[node_id] = NUMANode(
            node_id=NUMANodeId(node_id),
            cpu_cores=list(range(node_id * 8, (node_id + 1) * 8)),
            memory_gb=128,
            gpus=gpu_ids,
        )

    return Topology(numa_nodes=nodes)


def create_mock_gpu_device(gpu_id: GPUId, numa_node: int, healthy: bool = True) -> GPUDevice:
    """Create a mock GPU device for testing."""
    specs = GPUSpecs(
        gpu_id=gpu_id,
        model="A100 (test)",
        compute_capability="8.0",
        memory_mb=40960,
        pcie_bus_id=PCIeBusId("0000:00:00.0"),
        supports_mig=True,
    )

    health = GPUHealth(
        gpu_id=gpu_id,
        temperature_c=45.0 if healthy else 95.0,
        power_w=250.0,
        utilization_percent=0.0,
        memory_used_mb=0,
        ecc_errors_correctable=0,
        ecc_errors_uncorrectable=0 if healthy else 5,
        throttled=not healthy,
        last_update_timestamp=time.time(),
    )

    return GPUDevice(
        specs=specs,
        numa_node=NUMANodeId(numa_node),
        state=GPUState.AVAILABLE if healthy else GPUState.UNHEALTHY,
        health=health,
    )


class TestGPUSchedulerIntegration:
    """Integration tests for GPU scheduler."""

    def test_single_gpu_job_scheduling(self):
        """Test scheduling a single-GPU job."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit job
        job = Job(
            job_id=create_job_id("test-user", 1),
            user_id="test-user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=8192),
        )
        scheduler.submit_job(job)

        # Schedule
        decision = scheduler.schedule_pending()
        assert decision.success
        assert decision.placement is not None
        assert len(decision.placement.gpu_ids) == 1

        # Check job state
        assert job.state == JobState.SCHEDULED
        assert len(job.allocated_gpus) == 1

    def test_multi_gpu_gang_scheduling(self):
        """Test gang scheduling for multi-GPU jobs."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit 2-GPU job
        job = Job(
            job_id=create_job_id("test-user", 1),
            user_id="test-user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job)

        # Schedule
        decision = scheduler.schedule_pending()
        assert decision.success
        assert len(decision.placement.gpu_ids) == 2

        # Both GPUs should be on same NUMA node (gang scheduling)
        assert not decision.placement.is_cross_numa

    def test_numa_aware_placement(self):
        """Test that scheduler prefers same-NUMA placement."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit job requiring 2 GPUs
        job = Job(
            job_id=create_job_id("user", 1),
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job)

        decision = scheduler.schedule_pending()
        assert decision.success

        # GPUs should be on same NUMA node
        placement = decision.placement
        node_for_gpus = set()
        for gpu_id in placement.gpu_ids:
            node = topology.get_node_for_gpu(gpu_id)
            if node is not None:
                node_for_gpus.add(node)

        assert len(node_for_gpus) == 1  # All GPUs on same node

    def test_cross_numa_fallback(self):
        """Test cross-NUMA scheduling when same-node not possible."""
        # Create topology with 1 GPU per node
        topology = create_mock_topology(num_gpus=2, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit 2-GPU job allowing cross-NUMA
        job = Job(
            job_id=create_job_id("user", 1),
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(
                gpu_count=2,
                min_memory_mb=8192,
                allow_cross_numa=True,
            ),
        )
        scheduler.submit_job(job)

        decision = scheduler.schedule_pending()
        assert decision.success
        assert decision.placement.is_cross_numa

    def test_insufficient_resources_queued(self):
        """Test that jobs are queued when resources insufficient."""
        topology = create_mock_topology(num_gpus=2, num_nodes=1)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit job requiring more GPUs than available
        job = Job(
            job_id=create_job_id("user", 1),
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job)

        # First job schedules successfully
        decision = scheduler.schedule_pending()
        assert decision.success

        # Submit second job
        job2 = Job(
            job_id=create_job_id("user", 2),
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job2)

        # Second job fails to schedule (no resources)
        decision2 = scheduler.schedule_pending()
        assert not decision2.success
        assert "Insufficient" in decision2.reason


class TestJobLifecycle:
    """Integration tests for complete job lifecycle."""

    def test_full_job_lifecycle(self):
        """Test complete job lifecycle: submit -> schedule -> start -> complete."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit job
        job_id = create_job_id("user", 1)
        job = Job(
            job_id=job_id,
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=8192),
        )
        scheduler.submit_job(job)
        assert job.state == JobState.QUEUED

        # Schedule
        scheduler.schedule_pending()
        assert job.state == JobState.SCHEDULED

        # Start
        result = scheduler.start_job(job_id)
        assert result
        assert job.state == JobState.RUNNING

        # Complete
        result = scheduler.release_job(job_id)
        assert result
        assert job.state == JobState.COMPLETED

    def test_job_cancellation(self):
        """Test job cancellation releases resources."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        initial_stats = scheduler.get_stats()

        # Submit and schedule job
        job_id = create_job_id("user", 1)
        job = Job(
            job_id=job_id,
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job)
        scheduler.schedule_pending()
        scheduler.start_job(job_id)

        # Verify GPUs allocated
        mid_stats = scheduler.get_stats()
        assert mid_stats["allocated_gpus"] == 2

        # Cancel job
        result = scheduler.cancel_job(job_id)
        assert result
        assert job.state == JobState.CANCELLED

        # Verify GPUs released
        final_stats = scheduler.get_stats()
        assert final_stats["allocated_gpus"] == 0

    def test_job_failure_releases_resources(self):
        """Test that failed jobs release GPU resources."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Submit and schedule job
        job_id = create_job_id("user", 1)
        job = Job(
            job_id=job_id,
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
        )
        scheduler.submit_job(job)
        scheduler.schedule_pending()
        scheduler.start_job(job_id)

        # Fail job
        result = scheduler.fail_job(job_id, "Out of memory")
        assert result
        assert job.state == JobState.FAILED

        # Verify GPUs released
        stats = scheduler.get_stats()
        assert stats["allocated_gpus"] == 0


class TestHealthAwareScheduling:
    """Integration tests for health-aware GPU scheduling."""

    def test_unhealthy_gpu_excluded(self):
        """Test that unhealthy GPUs are not scheduled."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        # Register GPUs - make one unhealthy
        gpu_count = 0
        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                is_healthy = gpu_count != 0  # First GPU is unhealthy
                device = create_mock_gpu_device(gpu_id, node_id, healthy=is_healthy)
                scheduler.register_gpu(device)
                gpu_count += 1

        # Submit multiple 1-GPU jobs
        for i in range(3):
            job = Job(
                job_id=create_job_id("user", i),
                user_id="user",
                priority=JobPriority.NORMAL,
                gpu_request=GPURequest(gpu_count=1, min_memory_mb=8192),
            )
            scheduler.submit_job(job)
            decision = scheduler.schedule_pending()

            # Should schedule on healthy GPUs only
            if decision.success:
                for gpu_id in decision.placement.gpu_ids:
                    device = scheduler._devices[gpu_id]
                    assert device.is_healthy


class TestGPUDiscoveryIntegration:
    """Integration tests for GPU discovery service."""

    def test_dev_mode_discovery(self):
        """Test GPU discovery in development mode."""
        discovery = GPUDiscoveryService(dev_mode=True)

        specs = discovery.scan_gpus()
        assert len(specs) == 2  # Mock returns 2 GPUs

        for spec in specs:
            assert "simulated" in spec.model
            assert spec.memory_mb > 0

            # Get health
            health = discovery.get_health(spec.gpu_id)
            assert health is not None
            assert health.temperature_c < 80  # Not overheating
            assert not health.throttled

    def test_discovery_with_scheduler(self):
        """Test using discovery to populate scheduler."""
        discovery = GPUDiscoveryService(dev_mode=True)
        specs = discovery.scan_gpus()

        # Create topology from discovered GPUs
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=list(range(8)),
                memory_gb=128,
                gpus=[spec.gpu_id for spec in specs],
            )
        }
        topology = Topology(numa_nodes=nodes)
        scheduler = GPUScheduler(topology)

        # Register discovered GPUs
        for spec in specs:
            health = discovery.get_health(spec.gpu_id)
            device = GPUDevice(
                specs=spec,
                numa_node=NUMANodeId(0),
                state=GPUState.AVAILABLE,
                health=health,
            )
            scheduler.register_gpu(device)

        # Submit job
        job = Job(
            job_id=create_job_id("user", 1),
            user_id="user",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=8192),
        )
        scheduler.submit_job(job)

        # Should schedule successfully
        decision = scheduler.schedule_pending()
        assert decision.success


class TestSchedulerStats:
    """Integration tests for scheduler statistics."""

    def test_stats_reflect_state(self):
        """Test that stats accurately reflect scheduler state."""
        topology = create_mock_topology(num_gpus=4, num_nodes=2)
        scheduler = GPUScheduler(topology)

        for node_id, node in topology.numa_nodes.items():
            for gpu_id in node.gpus:
                device = create_mock_gpu_device(gpu_id, node_id)
                scheduler.register_gpu(device)

        # Initial stats
        stats = scheduler.get_stats()
        assert stats["total_gpus"] == 4
        assert stats["allocated_gpus"] == 0
        assert stats["pending_jobs"] == 0

        # Submit jobs
        for i in range(2):
            job = Job(
                job_id=create_job_id("user", i),
                user_id="user",
                priority=JobPriority.NORMAL,
                gpu_request=GPURequest(gpu_count=2, min_memory_mb=8192),
            )
            scheduler.submit_job(job)

        stats = scheduler.get_stats()
        assert stats["pending_jobs"] == 2

        # Schedule first job
        scheduler.schedule_pending()
        stats = scheduler.get_stats()
        assert stats["allocated_gpus"] == 2
        assert stats["pending_jobs"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
