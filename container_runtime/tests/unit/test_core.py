"""Unit tests for container_runtime domain layer."""

import pytest
from container_runtime.domain.entities.container import (
    Container,
    ContainerConfig,
    ContainerState,
    ResourceLimits,
    ContainerImage,
)
from container_runtime.domain.entities.job import (
    Job,
    JobState,
    Placement,
    ResourceAllocation,
)
from container_runtime.domain.services.container_manager import ContainerManager
from container_runtime.domain.services.scheduler import Scheduler
from container_runtime.domain.services.image_manager import ImageManager
from container_runtime.domain.value_objects.identifiers import (
    create_container_id,
    create_job_id,
    create_image_id,
    create_gpu_id,
)


class TestIdentifiers:
    """Test value object creation."""
    
    def test_create_container_id(self):
        """Test container ID creation."""
        cid = create_container_id("ubuntu", 0)
        assert "ubuntu" in cid
        assert "-0" in cid
    
    def test_create_job_id(self):
        """Test job ID creation."""
        jid = create_job_id("alice", 1704067200)
        assert "alice" in jid
        assert "1704067200" in jid
    
    def test_create_image_id(self):
        """Test image ID creation."""
        iid = create_image_id("ubuntu", "22.04")
        assert "ubuntu" in iid
        assert "22.04" in iid


class TestResourceLimits:
    """Test resource limits validation."""
    
    def test_valid_limits(self):
        """Test valid resource limits."""
        limits = ResourceLimits(
            cpu_shares=1024,
            memory_limit=512_000_000,
        )
        assert limits.is_valid()
    
    def test_invalid_memory(self):
        """Test invalid memory limit (too small)."""
        limits = ResourceLimits(memory_limit=1_000_000)  # 1MB
        assert not limits.is_valid()
    
    def test_invalid_cpu_period(self):
        """Test invalid CPU period."""
        limits = ResourceLimits(cpu_period=500)  # 500 microseconds
        assert not limits.is_valid()


class TestContainer:
    """Test container entity."""
    
    def test_container_creation(self):
        """Test creating a container."""
        container = Container(
            container_id="test-1",
            image_id="ubuntu:22.04",
        )
        assert container.state == ContainerState.CREATED
        assert not container.is_running()
    
    def test_container_lifecycle(self):
        """Test container state transitions."""
        container = Container(
            container_id="test-1",
            image_id="ubuntu:22.04",
        )
        
        container.start()
        assert container.is_running()
        
        container.stop(exit_code=0)
        assert container.state == ContainerState.STOPPED
        assert container.exit_code == 0
    
    def test_container_failure(self):
        """Test container failure."""
        container = Container(
            container_id="test-1",
            image_id="ubuntu:22.04",
        )
        
        container.fail("Out of memory")
        assert container.state == ContainerState.FAILED
        assert "Out of memory" in container.error_message


class TestContainerConfig:
    """Test container configuration."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = ContainerConfig(
            image_id="ubuntu:22.04",
            container_id="test-1",
            limits=ResourceLimits(memory_limit=512_000_000),
        )
        assert config.validate()
    
    def test_invalid_config(self):
        """Test invalid configuration."""
        config = ContainerConfig(
            image_id="",  # Empty image
            container_id="test-1",
        )
        assert not config.validate()


class TestJob:
    """Test job entity."""
    
    def test_job_creation(self):
        """Test creating a job."""
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
        )
        assert job.state == JobState.PENDING
        assert job.is_active()
    
    def test_job_lifecycle(self):
        """Test job state transitions."""
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
        )
        
        # Schedule
        placement = Placement(
            container_id="container-1",
            assigned=True,
        )
        job.schedule(placement)
        assert job.state == JobState.SCHEDULED
        
        # Run
        job.run()
        assert job.state == JobState.RUNNING
        
        # Complete
        job.complete()
        assert job.state == JobState.COMPLETED
        assert not job.is_active()


class TestResourceAllocation:
    """Test resource allocation."""
    
    def test_allocation_capacity(self):
        """Test capacity checking."""
        alloc = ResourceAllocation(
            total_cpu_shares=4096,
            total_memory_bytes=4_000_000_000,
            available_cpu_shares=4096,
            available_memory_bytes=4_000_000_000,
        )
        
        assert alloc.has_capacity(1024, 512_000_000, 0)
        assert not alloc.has_capacity(5000, 512_000_000, 0)
    
    def test_allocate_resources(self):
        """Test resource allocation."""
        alloc = ResourceAllocation(
            total_cpu_shares=4096,
            total_memory_bytes=4_000_000_000,
            available_cpu_shares=4096,
            available_memory_bytes=4_000_000_000,
            available_gpu_ids=["gpu-0", "gpu-1"],
        )
        
        alloc.allocate(1024, 512_000_000, 1, ["gpu-0"])
        
        assert alloc.available_cpu_shares == 3072
        assert alloc.available_memory_bytes == 3_488_000_000
        assert "gpu-0" not in alloc.available_gpu_ids


class TestContainerManager:
    """Test container manager service."""
    
    def test_create_container(self):
        """Test creating a container."""
        manager = ContainerManager()
        
        config = ContainerConfig(
            image_id="ubuntu:22.04",
            container_id="test-1",
        )
        
        container = manager.create(config)
        assert container.container_id == "test-1"
        assert container.state == ContainerState.CREATED
    
    def test_container_lifecycle(self):
        """Test container start/stop."""
        manager = ContainerManager()
        
        config = ContainerConfig(
            image_id="ubuntu:22.04",
            container_id="test-1",
        )
        
        container = manager.create(config)
        manager.start("test-1")
        
        container = manager.get("test-1")
        assert container.state == ContainerState.RUNNING
        assert container.pid is not None
        
        manager.stop("test-1")
        container = manager.get("test-1")
        assert container.state == ContainerState.STOPPED
    
    def test_list_containers(self):
        """Test listing containers."""
        manager = ContainerManager()
        
        for i in range(3):
            config = ContainerConfig(
                image_id="ubuntu:22.04",
                container_id=f"test-{i}",
            )
            manager.create(config)
        
        containers = manager.list()
        assert len(containers) == 3


class TestScheduler:
    """Test scheduler service."""
    
    def test_submit_job(self):
        """Test job submission."""
        scheduler = Scheduler()
        
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
            cpu_required=1024,
            memory_required=512_000_000,
        )
        
        jid = scheduler.submit(job)
        assert jid == "job-1"
    
    def test_schedule_job(self):
        """Test job scheduling."""
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=4096,
                total_memory_bytes=4_000_000_000,
                available_cpu_shares=4096,
                available_memory_bytes=4_000_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)
        
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
            cpu_required=1024,
            memory_required=512_000_000,
        )
        
        scheduler.submit(job)
        placements = scheduler.schedule_all()
        
        assert "job-1" in placements
        assert placements["job-1"].assigned
    
    def test_insufficient_resources(self):
        """Test scheduling with insufficient resources."""
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=1024,
                total_memory_bytes=256_000_000,
                available_cpu_shares=1024,
                available_memory_bytes=256_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)
        
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
            cpu_required=2048,  # More than available
            memory_required=512_000_000,
        )
        
        scheduler.submit(job)
        placements = scheduler.schedule_all()
        
        assert "job-1" in placements
        assert not placements["job-1"].assigned
    
    def test_cancel_job(self):
        """Test job cancellation."""
        scheduler = Scheduler()
        
        job = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
        )
        
        scheduler.submit(job)
        scheduler.cancel("job-1")
        
        cancelled_job = scheduler._jobs["job-1"]
        assert cancelled_job.state == JobState.CANCELLED
    
    def test_pending_queue_unchanged_if_job_not_scheduled(self):
        """Test that jobs stay in pending queue if not scheduled."""
        # Total resources: 2048 CPU, 512MB memory
        # job-1 (smaller): 512 CPU, 128MB - should schedule first after FFD
        # job-2 (larger): 512 CPU, 200MB - gets sorted first in FFD (descending)
        # After job-2: 1024 CPU, 184MB remaining - job-1 fits
        # Test: schedule both, they should both fit
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=2048,
                total_memory_bytes=512_000_000,
                available_cpu_shares=2048,
                available_memory_bytes=512_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)
        
        # Submit smaller job (scheduled second due to FFD sort)
        job1 = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
            cpu_required=512,
            memory_required=128_000_000,
        )
        scheduler.submit(job1)
        
        # Submit larger job (scheduled first due to FFD sort by memory descending)
        job2 = Job(
            job_id="job-2",
            image_id="ubuntu:22.04",
            container_id="container-2",
            cpu_required=512,
            memory_required=250_000_000,
        )
        scheduler.submit(job2)
        
        # Schedule all jobs
        placements = scheduler.schedule_all()
        
        # Both should be scheduled
        assert placements["job-2"].assigned  # Larger, scheduled first
        assert placements["job-1"].assigned  # Smaller, scheduled second
        assert "job-1" not in scheduler._pending_queue
        assert "job-2" not in scheduler._pending_queue
        
    def test_pending_queue_keeps_unschedulable_job(self):
        """Test that unschedulable jobs stay in pending queue."""
        # Total resources: 512 CPU, 256MB memory
        # job-1: 256 CPU, 128MB - fits
        # job-2: 256 CPU, 200MB - doesn't fit after job-1 allocated
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=512,
                total_memory_bytes=256_000_000,
                available_cpu_shares=512,
                available_memory_bytes=256_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)
        
        # Submit larger job
        job2 = Job(
            job_id="job-2",
            image_id="ubuntu:22.04",
            container_id="container-2",
            cpu_required=256,
            memory_required=200_000_000,
        )
        scheduler.submit(job2)
        
        # Submit smaller job
        job1 = Job(
            job_id="job-1",
            image_id="ubuntu:22.04",
            container_id="container-1",
            cpu_required=256,
            memory_required=128_000_000,
        )
        scheduler.submit(job1)
        
        # Schedule all jobs
        placements = scheduler.schedule_all()
        
        # job-2 scheduled first (FFD descending), job-1 doesn't fit after
        assert placements["job-2"].assigned
        assert not placements["job-1"].assigned
        assert "job-1" in scheduler._pending_queue


class TestImageManager:
    """Test image manager service."""
    
    def test_pull_image(self):
        """Test pulling an image."""
        manager = ImageManager()
        
        image = manager.pull("docker.io", "ubuntu", "22.04")
        assert image.name == "ubuntu"
        assert image.tag == "22.04"
    
    def test_cache_image(self):
        """Test image caching."""
        manager = ImageManager()
        
        image1 = manager.pull("docker.io", "ubuntu", "22.04")
        image2 = manager.pull("docker.io", "ubuntu", "22.04")
        
        assert image1.image_id == image2.image_id
    
    def test_list_images(self):
        """Test listing images."""
        manager = ImageManager()
        
        manager.pull("docker.io", "ubuntu", "22.04")
        manager.pull("docker.io", "python", "3.11")
        
        images = manager.list()
        assert len(images) == 2
    
    def test_delete_image(self):
        """Test deleting an image."""
        manager = ImageManager()
        
        image = manager.pull("docker.io", "ubuntu", "22.04")
        manager.delete(image.image_id)
        
        assert not manager.exists(image.image_id)
