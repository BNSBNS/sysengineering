"""Unit tests for GPU platform components."""

import pytest
from gpu_platform.domain.entities.gpu_device import (
    GPUDevice,
    GPUHealth,
    GPUSpecs,
    GPUState,
)
from gpu_platform.domain.entities.job import (
    GPURequest,
    Job,
    JobPriority,
    JobState,
    Placement,
)
from gpu_platform.domain.entities.topology import (
    NUMANode,
    NVLink,
    Topology,
)
from gpu_platform.domain.services.scheduler import GPUScheduler, ScheduleDecision
from gpu_platform.domain.value_objects.gpu_identifiers import (
    GPUId,
    JobId,
    NUMANodeId,
    PCIeBusId,
    create_gpu_id,
    create_job_id,
)


class TestGPUIdentifiers:
    """Test GPU identifier creation and type safety."""
    
    def test_create_gpu_id(self):
        """Test GPU ID creation."""
        gpu_id = create_gpu_id(0)
        assert gpu_id == "GPU-0"
        
        gpu_id = create_gpu_id(7)
        assert gpu_id == "GPU-7"
    
    def test_create_job_id(self):
        """Test job ID creation."""
        job_id = create_job_id("train", 1)
        assert job_id == "train-00000001"
        
        job_id = create_job_id("infer", 42)
        assert job_id == "infer-00000042"


class TestGPUDevice:
    """Test GPU device entity."""
    
    def test_device_creation(self):
        """Test creating a GPU device."""
        specs = GPUSpecs(
            gpu_id=create_gpu_id(0),
            model="A100",
            compute_capability="8.0",
            memory_mb=40960,
            pcie_bus_id=PCIeBusId("0000:01:00.0"),
            supports_mig=True,
        )
        
        health = GPUHealth(
            gpu_id=specs.gpu_id,
            temperature_c=45.0,
            power_w=250.0,
            utilization_percent=75.0,
            memory_used_mb=0,
            ecc_errors_correctable=0,
            ecc_errors_uncorrectable=0,
            throttled=False,
            last_update_timestamp=0.0,
        )
        
        device = GPUDevice(
            specs=specs,
            numa_node=NUMANodeId(0),
            state=GPUState.AVAILABLE,
            health=health,
        )
        
        assert device.specs.model == "A100"
        assert device.numa_node == 0
        assert device.is_available
        assert device.available_memory_mb == 40960
    
    def test_device_health_check(self):
        """Test health status checks."""
        specs = GPUSpecs(
            gpu_id=create_gpu_id(0),
            model="A100",
            compute_capability="8.0",
            memory_mb=40960,
            pcie_bus_id=PCIeBusId("0000:01:00.0"),
        )
        
        health = GPUHealth(
            gpu_id=specs.gpu_id,
            temperature_c=45.0,
            power_w=250.0,
            utilization_percent=75.0,
            memory_used_mb=10240,
            ecc_errors_correctable=0,
            ecc_errors_uncorrectable=0,
            throttled=False,
            last_update_timestamp=0.0,
        )
        
        device = GPUDevice(
            specs=specs,
            numa_node=NUMANodeId(0),
            health=health,
        )
        
        assert device.is_healthy
        assert device.available_memory_mb == 30720


class TestJob:
    """Test job entity."""
    
    def test_job_creation(self):
        """Test creating a job."""
        job = Job(
            job_id=create_job_id("train", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )
        
        assert job.is_multi_gpu
        assert not job.is_allocated
        assert job.state == JobState.PENDING
    
    def test_job_allocation(self):
        """Test job GPU allocation."""
        job = Job(
            job_id=create_job_id("train", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )
        
        job.allocated_gpus = [create_gpu_id(0), create_gpu_id(1)]
        assert job.is_allocated
        assert len(job.allocated_gpus) == 2


class TestTopology:
    """Test cluster topology."""
    
    def test_topology_creation(self):
        """Test creating a topology."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1, 2, 3],
                memory_gb=128,
                gpus=[create_gpu_id(0), create_gpu_id(1)],
            ),
            1: NUMANode(
                node_id=NUMANodeId(1),
                cpu_cores=[4, 5, 6, 7],
                memory_gb=128,
                gpus=[create_gpu_id(2), create_gpu_id(3)],
            ),
        }
        
        topology = Topology(numa_nodes=nodes)
        
        assert topology.node_count == 2
        assert topology.total_gpus == 4
        assert topology.total_memory_gb == 256
    
    def test_get_node_for_gpu(self):
        """Test finding NUMA node for a GPU."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1],
                memory_gb=128,
                gpus=[create_gpu_id(0), create_gpu_id(1)],
            ),
        }
        
        topology = Topology(numa_nodes=nodes)
        
        node_id = topology.get_node_for_gpu(create_gpu_id(0))
        assert node_id == 0


class TestScheduler:
    """Test GPU scheduler."""
    
    def test_scheduler_creation(self):
        """Test creating a scheduler."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1],
                memory_gb=128,
                gpus=[create_gpu_id(0), create_gpu_id(1)],
            ),
        }
        
        topology = Topology(numa_nodes=nodes)
        scheduler = GPUScheduler(topology)
        
        # Register GPUs
        for i in range(2):
            specs = GPUSpecs(
                gpu_id=create_gpu_id(i),
                model="A100",
                compute_capability="8.0",
                memory_mb=40960,
                pcie_bus_id=PCIeBusId("0000:01:00.0"),
            )
            health = GPUHealth(
                gpu_id=specs.gpu_id,
                temperature_c=45.0,
                power_w=250.0,
                utilization_percent=75.0,
                memory_used_mb=0,
                ecc_errors_correctable=0,
                ecc_errors_uncorrectable=0,
                throttled=False,
                last_update_timestamp=0.0,
            )
            device = GPUDevice(specs=specs, numa_node=NUMANodeId(0), health=health)
            scheduler.register_gpu(device)
        
        assert len(scheduler._devices) == 2
    
    def test_job_submission(self):
        """Test submitting a job."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1],
                memory_gb=128,
                gpus=[create_gpu_id(0), create_gpu_id(1)],
            ),
        }
        
        topology = Topology(numa_nodes=nodes)
        scheduler = GPUScheduler(topology)
        
        # Register GPUs
        for i in range(2):
            specs = GPUSpecs(
                gpu_id=create_gpu_id(i),
                model="A100",
                compute_capability="8.0",
                memory_mb=40960,
                pcie_bus_id=PCIeBusId("0000:01:00.0"),
            )
            health = GPUHealth(
                gpu_id=specs.gpu_id,
                temperature_c=45.0,
                power_w=250.0,
                utilization_percent=75.0,
                memory_used_mb=0,
                ecc_errors_correctable=0,
                ecc_errors_uncorrectable=0,
                throttled=False,
                last_update_timestamp=0.0,
            )
            device = GPUDevice(specs=specs, numa_node=NUMANodeId(0), health=health)
            scheduler.register_gpu(device)
        
        # Submit a job
        job = Job(
            job_id=create_job_id("train", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )
        
        scheduler.submit_job(job)
        assert job.state == JobState.QUEUED
        assert len(scheduler._queue) == 1
    
    def test_job_scheduling(self):
        """Test scheduling a job."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1],
                memory_gb=128,
                gpus=[create_gpu_id(0), create_gpu_id(1)],
            ),
        }
        
        topology = Topology(numa_nodes=nodes)
        scheduler = GPUScheduler(topology)
        
        # Register GPUs
        for i in range(2):
            specs = GPUSpecs(
                gpu_id=create_gpu_id(i),
                model="A100",
                compute_capability="8.0",
                memory_mb=40960,
                pcie_bus_id=PCIeBusId("0000:01:00.0"),
            )
            health = GPUHealth(
                gpu_id=specs.gpu_id,
                temperature_c=45.0,
                power_w=250.0,
                utilization_percent=75.0,
                memory_used_mb=0,
                ecc_errors_correctable=0,
                ecc_errors_uncorrectable=0,
                throttled=False,
                last_update_timestamp=0.0,
            )
            device = GPUDevice(specs=specs, numa_node=NUMANodeId(0), health=health)
            scheduler.register_gpu(device)
        
        # Submit and schedule a job
        job = Job(
            job_id=create_job_id("train", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )
        
        scheduler.submit_job(job)
        decision = scheduler.schedule_pending()
        
        assert decision.success
        assert decision.placement is not None
        assert len(decision.placement.gpu_ids) == 2
        assert job.state == JobState.SCHEDULED


class TestJobLifecycle:
    """Test job lifecycle operations."""

    def _create_scheduler_with_gpus(self, num_gpus: int = 2) -> GPUScheduler:
        """Helper to create a scheduler with registered GPUs."""
        nodes = {
            0: NUMANode(
                node_id=NUMANodeId(0),
                cpu_cores=[0, 1, 2, 3],
                memory_gb=128,
                gpus=[create_gpu_id(i) for i in range(num_gpus)],
            ),
        }

        topology = Topology(numa_nodes=nodes)
        scheduler = GPUScheduler(topology)

        for i in range(num_gpus):
            specs = GPUSpecs(
                gpu_id=create_gpu_id(i),
                model="A100",
                compute_capability="8.0",
                memory_mb=40960,
                pcie_bus_id=PCIeBusId(f"0000:0{i}:00.0"),
            )
            health = GPUHealth(
                gpu_id=specs.gpu_id,
                temperature_c=45.0,
                power_w=250.0,
                utilization_percent=25.0,
                memory_used_mb=0,
                ecc_errors_correctable=0,
                ecc_errors_uncorrectable=0,
                throttled=False,
                last_update_timestamp=0.0,
            )
            device = GPUDevice(specs=specs, numa_node=NUMANodeId(0), health=health)
            scheduler.register_gpu(device)

        return scheduler

    def test_get_job(self):
        """Test retrieving a job by ID."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("test", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )

        scheduler.submit_job(job)

        # Get existing job
        retrieved = scheduler.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id
        assert retrieved.user_id == "alice"

        # Get non-existent job
        not_found = scheduler.get_job(create_job_id("nonexistent", 999))
        assert not_found is None

    def test_start_job(self):
        """Test starting a scheduled job."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("test", 2),
            user_id="bob",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )

        scheduler.submit_job(job)
        scheduler.schedule_pending()

        # Job should be SCHEDULED
        assert job.state == JobState.SCHEDULED

        # Start the job
        result = scheduler.start_job(job.job_id)
        assert result is True
        assert job.state == JobState.RUNNING

        # Can't start an already running job
        result = scheduler.start_job(job.job_id)
        assert result is False

    def test_cancel_queued_job(self):
        """Test canceling a queued job."""
        scheduler = self._create_scheduler_with_gpus(num_gpus=1)

        # Submit two jobs (second will be queued)
        job1 = Job(
            job_id=create_job_id("first", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )
        job2 = Job(
            job_id=create_job_id("second", 2),
            user_id="bob",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )

        scheduler.submit_job(job1)
        scheduler.schedule_pending()
        scheduler.submit_job(job2)  # Will remain queued

        assert job2.state == JobState.QUEUED

        # Cancel the queued job
        result = scheduler.cancel_job(job2.job_id)
        assert result is True
        assert job2.state == JobState.CANCELLED

    def test_cancel_running_job(self):
        """Test canceling a running job releases GPUs."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("test", 3),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )

        scheduler.submit_job(job)
        scheduler.schedule_pending()
        scheduler.start_job(job.job_id)

        assert job.state == JobState.RUNNING

        # Cancel running job
        result = scheduler.cancel_job(job.job_id)
        assert result is True
        assert job.state == JobState.CANCELLED

        # GPUs should be available again
        stats = scheduler.get_stats()
        assert stats["available_gpus"] == 2

    def test_fail_job(self):
        """Test marking a job as failed."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("test", 4),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )

        scheduler.submit_job(job)
        scheduler.schedule_pending()
        scheduler.start_job(job.job_id)

        # Fail the job
        result = scheduler.fail_job(job.job_id, reason="GPU memory error")
        assert result is True
        assert job.state == JobState.FAILED

        # GPUs should be available again
        stats = scheduler.get_stats()
        assert stats["available_gpus"] == 2

    def test_complete_job(self):
        """Test completing a running job."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("test", 5),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=2, min_memory_mb=20480),
        )

        scheduler.submit_job(job)
        scheduler.schedule_pending()
        scheduler.start_job(job.job_id)

        # Complete the job
        result = scheduler.release_job(job.job_id)
        assert result is True
        assert job.state == JobState.COMPLETED

        # GPUs should be available again
        stats = scheduler.get_stats()
        assert stats["available_gpus"] == 2

    def test_full_job_lifecycle(self):
        """Test complete job lifecycle: PENDING → QUEUED → SCHEDULED → RUNNING → COMPLETED."""
        scheduler = self._create_scheduler_with_gpus()

        job = Job(
            job_id=create_job_id("lifecycle", 1),
            user_id="alice",
            priority=JobPriority.NORMAL,
            gpu_request=GPURequest(gpu_count=1, min_memory_mb=20480),
        )

        # Initial state is PENDING
        assert job.state == JobState.PENDING

        # Submit → QUEUED
        scheduler.submit_job(job)
        assert job.state == JobState.QUEUED

        # Schedule → SCHEDULED
        decision = scheduler.schedule_pending()
        assert decision.success
        assert job.state == JobState.SCHEDULED

        # Start → RUNNING
        result = scheduler.start_job(job.job_id)
        assert result
        assert job.state == JobState.RUNNING

        # Complete → COMPLETED
        result = scheduler.release_job(job.job_id)
        assert result
        assert job.state == JobState.COMPLETED

        # Verify job completed timestamp is set
        assert job.completed_timestamp is not None


class TestGPUDiscovery:
    """Test GPU discovery service."""

    def test_discovery_dev_mode(self):
        """Test GPU discovery in development mode (no hardware)."""
        from gpu_platform.domain.services.gpu_discovery import GPUDiscoveryService
        
        # Create discovery service in dev mode
        discovery = GPUDiscoveryService(dev_mode=True)
        
        # Scan GPUs - should return mock GPUs
        gpus = discovery.scan_gpus()
        assert len(gpus) == 2
        assert gpus[0].model == "A100 (simulated)"
        assert gpus[0].compute_capability == "8.0"
        assert gpus[0].memory_mb == 40960
        
        discovery.shutdown()
    
    def test_discovery_dev_mode_health(self):
        """Test health metrics in development mode."""
        from gpu_platform.domain.services.gpu_discovery import GPUDiscoveryService
        
        discovery = GPUDiscoveryService(dev_mode=True)
        gpus = discovery.scan_gpus()
        
        # Get health for first GPU
        health = discovery.get_health(gpus[0].gpu_id)
        assert health is not None
        assert health.temperature_c == 45.0
        assert health.power_w == 250.0
        assert health.utilization_percent == 25.0
        assert health.throttled is False
        
        discovery.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
