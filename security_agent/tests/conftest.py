"""Pytest configuration and shared fixtures for Security Agent tests."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from security_agent.infrastructure.config import Config, get_config
from security_agent.infrastructure.container import Container


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
def mock_event():
    """Provide a mock security event for testing."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "syscall",
        "process": {
            "pid": 1234,
            "ppid": 1,
            "name": "test_process",
            "exe_path": "/usr/bin/test",
            "cmdline": "test --arg value",
        },
        "payload": {
            "syscall": "execve",
            "args": ["/bin/bash", "-c", "echo test"],
        },
        "metadata": {},
    }


@pytest.fixture
def mock_detection():
    """Provide a mock detection result."""
    return {
        "detection_id": "det-1234",
        "timestamp": datetime.utcnow().isoformat(),
        "severity": "high",
        "category": "execution",
        "rule_id": "suspicious_exec",
        "event": {},
        "confidence": 0.95,
    }


@pytest.fixture
def mock_ebpf_probe():
    """Mock eBPF probe for testing without kernel access."""
    probe = MagicMock()
    probe.is_loaded = True
    probe.read_events = MagicMock(return_value=[])
    return probe


@pytest.fixture
def container(test_config: Config) -> Container:
    """Provide a configured container for testing."""
    with patch("security_agent.infrastructure.container.get_config", return_value=test_config):
        return Container.create()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "chaos: mark test as chaos test")
    config.addinivalue_line("markers", "property: mark test as property-based test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "ebpf: mark test as requiring eBPF (root/capabilities)")
