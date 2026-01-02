"""Pytest configuration and fixtures for container_runtime tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from prometheus_client import CollectorRegistry

from container_runtime.infrastructure.config import Config, RuntimeConfig
from container_runtime.infrastructure.container import Container, reset_container
from container_runtime.infrastructure.metrics import MetricsRegistry


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Provide a test configuration."""
    return Config(
        runtime=RuntimeConfig(
            root_dir=temp_dir / "containers",
            state_dir=temp_dir / "run",
            image_dir=temp_dir / "images",
        )
    )


@pytest.fixture
def container() -> Generator[Container, None, None]:
    """Provide a fresh DI container."""
    reset_container()
    c = Container()
    yield c
    c.clear()


@pytest.fixture
def metrics_registry() -> MetricsRegistry:
    """Provide a fresh metrics registry."""
    registry = CollectorRegistry(auto_describe=True)
    return MetricsRegistry(registry=registry)


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "chaos: Chaos tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "benchmark: Benchmarks")
    config.addinivalue_line("markers", "requires_root: Requires root privileges")
