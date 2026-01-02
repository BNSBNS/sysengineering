"""Pytest configuration and fixtures for db_engine tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from prometheus_client import CollectorRegistry

from db_engine.infrastructure.config import Config, StorageConfig, WALConfig
from db_engine.infrastructure.container import Container, reset_container
from db_engine.infrastructure.metrics import MetricsRegistry


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Provide a test configuration with temporary directories."""
    return Config(
        storage=StorageConfig(
            data_dir=temp_dir / "data",
            wal_dir=temp_dir / "wal",
            page_size=4096,
            buffer_pool_size=1048576,  # 1MB for tests
        ),
        wal=WALConfig(
            segment_size=1048576,  # 1MB for tests
            sync_mode="none",  # Faster for tests
        ),
    )


@pytest.fixture
def container() -> Generator[Container, None, None]:
    """Provide a fresh DI container for each test."""
    reset_container()
    c = Container()
    yield c
    c.clear()


@pytest.fixture
def metrics_registry() -> MetricsRegistry:
    """Provide a fresh metrics registry for each test."""
    # Use a separate registry to avoid conflicts between tests
    registry = CollectorRegistry(auto_describe=True)
    return MetricsRegistry(registry=registry)


# Markers for test categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "chaos: Chaos/fault injection tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")
    config.addinivalue_line("markers", "slow: Slow tests")
