"""Unit tests for container runtime configuration."""

from pathlib import Path
import pytest

from container_runtime.infrastructure.config import (
    Config,
    RuntimeConfig,
    CgroupConfig,
    NetworkConfig,
    SchedulerConfig,
    GPUConfig,
    ServerConfig,
    ObservabilityConfig,
)


@pytest.mark.unit
class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RuntimeConfig()
        assert config.root_dir == Path("/var/lib/containers")
        assert config.state_dir == Path("/run/containers")
        assert config.image_dir == Path("/var/lib/containers/images")

    def test_custom_values(self, temp_dir: Path):
        """Test custom configuration values."""
        config = RuntimeConfig(
            root_dir=temp_dir / "containers",
            state_dir=temp_dir / "run",
            image_dir=temp_dir / "images",
        )
        assert config.root_dir == temp_dir / "containers"


@pytest.mark.unit
class TestCgroupConfig:
    """Tests for CgroupConfig."""

    def test_default_values(self):
        """Test default cgroup configuration."""
        config = CgroupConfig()
        assert config.version == "v2"
        assert config.default_cpu_shares == 1024
        assert config.default_memory_limit == 536870912  # 512MB


@pytest.mark.unit
class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_values(self):
        """Test default network configuration."""
        config = NetworkConfig()
        assert config.bridge_name == "ctr0"
        assert config.bridge_subnet == "172.20.0.0/16"
        assert config.enable_nat is True


@pytest.mark.unit
class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_values(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        assert config.algorithm == "binpack"
        assert config.max_pending_jobs == 1000
        assert config.placement_timeout_seconds == 30


@pytest.mark.unit
class TestGPUConfig:
    """Tests for GPUConfig."""

    def test_default_values(self):
        """Test default GPU configuration."""
        config = GPUConfig()
        assert config.enabled is True
        assert config.allow_oversubscription is False
        assert config.default_gpu_memory_fraction == 1.0


@pytest.mark.unit
class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.grpc_port == 50052
        assert config.metrics_port == 8002


@pytest.mark.unit
class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.otel_endpoint is None


@pytest.mark.unit
class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.cgroup, CgroupConfig)
        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.gpu, GPUConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.observability, ObservabilityConfig)

    def test_ensure_directories(self, temp_dir: Path):
        """Test directory creation."""
        config = Config(
            runtime=RuntimeConfig(
                root_dir=temp_dir / "containers",
                state_dir=temp_dir / "run",
                image_dir=temp_dir / "images",
            )
        )
        config.ensure_directories()
        assert config.runtime.root_dir.exists()
        assert config.runtime.state_dir.exists()
        assert config.runtime.image_dir.exists()
