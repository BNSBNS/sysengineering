"""Unit tests for Security Agent configuration."""

import pytest
from security_agent.infrastructure.config import Config, EBPFConfig, DetectionConfig, ResponseConfig


@pytest.mark.unit
class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.ebpf.enable_syscall_tracing is True
        assert config.detection.enable_ml_detection is True
        assert config.response.enable_auto_response is False

    def test_ebpf_config_defaults(self):
        """Test eBPF configuration defaults."""
        ebpf_config = EBPFConfig()
        assert ebpf_config.ring_buffer_size == 16384
        assert ebpf_config.enable_network_tracing is True

    def test_detection_config_defaults(self):
        """Test detection configuration defaults."""
        detection_config = DetectionConfig()
        assert detection_config.anomaly_threshold == 0.85
        assert detection_config.rule_engine_enabled is True

    def test_response_config_defaults(self):
        """Test response configuration defaults."""
        response_config = ResponseConfig()
        assert response_config.enable_process_kill is False
        assert response_config.enable_file_quarantine is True
