"""Unit tests for Streaming System configuration."""

import pytest
from streaming_system.infrastructure.config import Config, BrokerConfig, StorageConfig, RaftConfig


@pytest.mark.unit
class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.broker.broker_id == 1
        assert config.broker.port == 9092

    def test_broker_config_defaults(self):
        """Test broker configuration defaults."""
        broker_config = BrokerConfig()
        assert broker_config.host == "localhost"
        assert broker_config.raft_port == 9093

    def test_storage_config_defaults(self):
        """Test storage configuration defaults."""
        storage_config = StorageConfig()
        assert storage_config.segment_size == "1GB"
        assert storage_config.retention_hours == 168

    def test_raft_config_defaults(self):
        """Test Raft configuration defaults."""
        raft_config = RaftConfig()
        assert raft_config.election_timeout_min_ms == 150
        assert raft_config.election_timeout_max_ms == 300
        assert raft_config.heartbeat_interval_ms == 50
