"""Pytest configuration and shared fixtures for Streaming System tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from streaming_system.infrastructure.config import Config, get_config
from streaming_system.infrastructure.container import Container


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
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="streaming_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_record():
    """Provide a mock record for testing."""
    return {
        "offset": 0,
        "timestamp": 1234567890,
        "key": b"test-key",
        "value": b"test-value",
        "headers": {},
    }


@pytest.fixture
def mock_raft_node():
    """Provide a mock Raft node for testing."""
    node = MagicMock()
    node.term = 1
    node.state = "follower"
    node.leader_id = None
    node.voted_for = None
    return node


@pytest.fixture
def container(test_config: Config) -> Container:
    """Provide a configured container for testing."""
    with patch("streaming_system.infrastructure.container.get_config", return_value=test_config):
        return Container.create()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "chaos: mark test as chaos test")
    config.addinivalue_line("markers", "property: mark test as property-based test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "raft: mark test as Raft consensus test")
