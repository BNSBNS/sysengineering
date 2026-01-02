"""Integration tests for container_runtime lifecycle."""

import pytest

from container_runtime.domain.entities.container import (
    Container,
    ContainerConfig,
    ContainerState,
    ResourceLimits,
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


class TestContainerLifecycle:
    """Integration tests for full container lifecycle."""

    def test_full_container_lifecycle(self):
        """Test creating, starting, stopping, and deleting a container."""
        manager = ContainerManager()

        # Create container
        config = ContainerConfig(
            image_id="ubuntu:22.04",
            container_id="lifecycle-test-1",
            limits=ResourceLimits(memory_limit=512_000_000),
        )
        container = manager.create(config)
        assert container.state == ContainerState.CREATED

        # Start container
        manager.start("lifecycle-test-1")
        container = manager.get("lifecycle-test-1")
        assert container.state == ContainerState.RUNNING
        assert container.pid is not None

        # Get stats while running
        stats = manager.get_stats("lifecycle-test-1")
        assert stats.cpu_percent > 0
        assert stats.uptime_seconds >= 0

        # Stop container
        manager.stop("lifecycle-test-1")
        container = manager.get("lifecycle-test-1")
        assert container.state == ContainerState.STOPPED

        # Delete container
        manager.delete("lifecycle-test-1")
        assert manager.get("lifecycle-test-1") is None


class TestSchedulerIntegration:
    """Integration tests for scheduler with container manager."""

    def test_schedule_and_run_job(self):
        """Test scheduling a job and running it through the container manager."""
        # Setup scheduler with resources
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=4096,
                total_memory_bytes=4_000_000_000,
                available_cpu_shares=4096,
                available_memory_bytes=4_000_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)
        manager = ContainerManager()

        # Submit job
        job = Job(
            job_id="integration-job-1",
            image_id="ubuntu:22.04",
            container_id="integration-container-1",
            cpu_required=1024,
            memory_required=512_000_000,
        )
        scheduler.submit(job)

        # Schedule
        placements = scheduler.schedule_all()
        assert placements["integration-job-1"].assigned
        placement = placements["integration-job-1"]

        # Create and start container based on placement
        config = ContainerConfig(
            image_id=job.image_id,
            container_id=placement.container_id,
            limits=ResourceLimits(
                cpu_shares=placement.cpu_allocation,
                memory_limit=placement.memory_allocation,
            ),
        )
        container = manager.create(config)
        manager.start(container.container_id)

        # Verify running
        container = manager.get(container.container_id)
        assert container.state == ContainerState.RUNNING

        # Update job state
        job.run()
        assert job.state == JobState.RUNNING

        # Complete
        manager.stop(container.container_id)
        job.complete()
        assert job.state == JobState.COMPLETED


class TestMultipleContainers:
    """Integration tests for managing multiple containers."""

    def test_multiple_container_lifecycle(self):
        """Test managing multiple containers concurrently."""
        manager = ContainerManager()

        # Create multiple containers
        container_ids = []
        for i in range(5):
            config = ContainerConfig(
                image_id="ubuntu:22.04",
                container_id=f"multi-test-{i}",
            )
            manager.create(config)
            container_ids.append(f"multi-test-{i}")

        assert len(manager.list()) == 5

        # Start all
        for cid in container_ids:
            manager.start(cid)

        running = [c for c in manager.list() if c.state == ContainerState.RUNNING]
        assert len(running) == 5

        # Stop all
        for cid in container_ids:
            manager.stop(cid)

        stopped = [c for c in manager.list() if c.state == ContainerState.STOPPED]
        assert len(stopped) == 5

        # Delete all
        for cid in container_ids:
            manager.delete(cid)

        assert len(manager.list()) == 0


class TestImageManagerIntegration:
    """Integration tests for image manager with container creation."""

    def test_pull_and_create_container(self):
        """Test pulling an image and using it to create a container."""
        image_manager = ImageManager()
        container_manager = ContainerManager()

        # Pull image
        image = image_manager.pull("docker.io", "python", "3.11")
        assert image_manager.exists(image.image_id)

        # Create container using the image
        config = ContainerConfig(
            image_id=image.image_id,
            container_id="python-test-1",
        )
        container = container_manager.create(config)
        assert container.image_id == image.image_id

        # Start and run
        container_manager.start("python-test-1")
        container = container_manager.get("python-test-1")
        assert container.state == ContainerState.RUNNING


class TestBinPackingScheduler:
    """Integration tests for bin-packing scheduler behavior."""

    def test_bin_packing_fills_nodes_efficiently(self):
        """Test that bin-packing algorithm efficiently uses resources."""
        # Setup: 4GB memory node
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=4096,
                total_memory_bytes=4_000_000_000,
                available_cpu_shares=4096,
                available_memory_bytes=4_000_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)

        # Submit jobs of varying sizes
        # Total: 500MB + 1GB + 1.5GB + 750MB = 3.75GB (should all fit in 4GB)
        jobs = [
            Job(job_id="small", image_id="img", container_id="c1",
                cpu_required=256, memory_required=500_000_000),
            Job(job_id="medium", image_id="img", container_id="c2",
                cpu_required=512, memory_required=1_000_000_000),
            Job(job_id="large", image_id="img", container_id="c3",
                cpu_required=1024, memory_required=1_500_000_000),
            Job(job_id="medium2", image_id="img", container_id="c4",
                cpu_required=512, memory_required=750_000_000),
        ]

        for job in jobs:
            scheduler.submit(job)

        # Schedule all
        placements = scheduler.schedule_all()

        # All should be scheduled
        for job in jobs:
            assert placements[job.job_id].assigned, f"Job {job.job_id} should be scheduled"

        # Check FFD ordering (largest first)
        stats = scheduler.get_stats()
        assert stats["scheduled"] == 4

    def test_resource_exhaustion(self):
        """Test scheduler behavior when resources are exhausted."""
        # Setup: Limited resources
        nodes = {
            "node-0": ResourceAllocation(
                total_cpu_shares=1024,
                total_memory_bytes=1_000_000_000,  # 1GB
                available_cpu_shares=1024,
                available_memory_bytes=1_000_000_000,
            )
        }
        scheduler = Scheduler(nodes=nodes)

        # Submit jobs that exceed capacity
        job1 = Job(job_id="job1", image_id="img", container_id="c1",
                   cpu_required=512, memory_required=600_000_000)  # 600MB
        job2 = Job(job_id="job2", image_id="img", container_id="c2",
                   cpu_required=512, memory_required=600_000_000)  # 600MB - won't fit

        scheduler.submit(job1)
        scheduler.submit(job2)

        placements = scheduler.schedule_all()

        # First job should be scheduled (FFD: job2 is equal size, sorted by submission?)
        # Actually FFD sorts by memory descending, both are equal
        # So job1 goes first (first submitted)
        assigned_count = sum(1 for p in placements.values() if p.assigned)
        assert assigned_count == 1

        # One job should remain pending
        stats = scheduler.get_stats()
        assert len(scheduler._pending_queue) == 1


class TestContainerStats:
    """Integration tests for container statistics."""

    def test_stats_across_lifecycle(self):
        """Test that stats reflect container state throughout lifecycle."""
        manager = ContainerManager()

        config = ContainerConfig(
            image_id="ubuntu:22.04",
            container_id="stats-test",
            limits=ResourceLimits(memory_limit=1_000_000_000),
        )
        manager.create(config)

        # Stats when created (not running)
        stats = manager.get_stats("stats-test")
        assert stats.state == ContainerState.CREATED
        assert stats.cpu_percent == 0
        assert stats.pids_count == 0

        # Stats when running
        manager.start("stats-test")
        stats = manager.get_stats("stats-test")
        assert stats.state == ContainerState.RUNNING
        assert stats.cpu_percent > 0
        assert stats.pids_count > 0
        assert stats.memory_limit_bytes == 1_000_000_000

        # Stats when stopped
        manager.stop("stats-test")
        stats = manager.get_stats("stats-test")
        assert stats.state == ContainerState.STOPPED
        assert stats.cpu_percent == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
