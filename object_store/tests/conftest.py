"""Pytest configuration and shared fixtures for Object Store tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from object_store.infrastructure.config import Config, get_config
from object_store.infrastructure.container import Container


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
    temp_dir = tempfile.mkdtemp(prefix="object_store_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_object_data():
    """Provide sample object data for testing."""
    return b"Hello, World! This is test data for the object store."


@pytest.fixture
def large_object_data():
    """Provide larger object data for multipart testing."""
    return b"X" * (10 * 1024 * 1024)  # 10 MB


@pytest.fixture
def mock_chunk_ref():
    """Provide a mock chunk reference."""
    return {
        "hash": "sha256:abcd1234...",
        "size": 65536,
        "shards": [
            {"node": "node1", "path": "/data/shard0"},
            {"node": "node2", "path": "/data/shard1"},
        ],
    }


@pytest.fixture
def container(test_config: Config) -> Container:
    """Provide a configured container for testing."""
    with patch("object_store.infrastructure.container.get_config", return_value=test_config):
        return Container.create()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "chaos: mark test as chaos test")
    config.addinivalue_line("markers", "property: mark test as property-based test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "s3: mark test as S3 compatibility test")
