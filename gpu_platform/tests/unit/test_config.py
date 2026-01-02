"""Unit tests for GPU Platform configuration."""

import pytest
from gpu_platform.infrastructure.config import (
    AllocationConfig,
    DiscoveryConfig,
    SchedulerConfig,
    ServerConfig,
)


@pytest.mark.unit
class TestConfig:
    """Test configuration loading and validation."""

    def test_discovery_config_defaults(self):
        """Test discovery configuration defaults."""
        config = DiscoveryConfig()
        assert config.enable_health_checks is True
        assert config.detect_ecc_errors is True
        assert config.poll_interval_seconds == 30

    def test_allocation_config_defaults(self):
        """Test allocation configuration defaults."""
        allocation_config = AllocationConfig()
        assert allocation_config.enable_mps is False
        assert allocation_config.enable_mig is False

    def test_scheduler_config_defaults(self):
        """Test scheduler configuration defaults."""
        scheduler_config = SchedulerConfig()
        assert scheduler_config.algorithm in ["numa_aware", "gang", "binpack"]
        assert scheduler_config.enable_preemption is False

    def test_server_config_defaults(self):
        """Test server configuration defaults."""
        server_config = ServerConfig()
        assert server_config.host == "0.0.0.0"
        assert server_config.grpc_port == 50054
        assert server_config.metrics_port == 8004
