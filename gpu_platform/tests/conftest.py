"""Pytest configuration and shared fixtures for GPU Platform tests."""

import pytest
from unittest.mock import MagicMock, patch

from gpu_platform.infrastructure.config import Config, get_config
from gpu_platform.infrastructure.container import Container


@pytest.fixture(autouse=True)
def reset_container():
    """Reset the DI container before each test."""
    Container.reset()
    yield
    Container.reset()


@pytest.fixture
def test_config() -> Config:
    """Provide a test configuration."""
    return Config()


@pytest.fixture
def mock_gpu_device():
    """Provide a mock GPU device for testing."""
    device = MagicMock()
    device.uuid = "GPU-test-uuid-1234"
    device.name = "NVIDIA Test GPU"
    device.memory_total = 16 * 1024 * 1024 * 1024  # 16 GB
    device.memory_free = 12 * 1024 * 1024 * 1024  # 12 GB
    device.utilization = 25.0
    device.temperature = 45
    device.power_usage = 150.0
    device.power_limit = 300.0
    return device


@pytest.fixture
def mock_nvml():
    """Mock NVML library for GPU operations."""
    with patch("pynvml.nvmlInit") as mock_init:
        with patch("pynvml.nvmlDeviceGetCount") as mock_count:
            with patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_handle:
                mock_count.return_value = 1
                yield {
                    "init": mock_init,
                    "count": mock_count,
                    "handle": mock_handle,
                }


@pytest.fixture
def container(test_config: Config) -> Container:
    """Provide a configured container for testing."""
    with patch("gpu_platform.infrastructure.container.get_config", return_value=test_config):
        return Container.create()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "chaos: mark test as chaos test")
    config.addinivalue_line("markers", "property: mark test as property-based test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
