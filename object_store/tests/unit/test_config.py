"""Unit tests for Object Store configuration."""

import pytest
from object_store.infrastructure.config import Config, StorageConfig, ErasureCodingConfig


@pytest.mark.unit
class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.storage.enable_deduplication is True
        assert config.erasure_coding.enabled is True
        assert config.server.port == 9000

    def test_storage_config_defaults(self):
        """Test storage configuration defaults."""
        storage_config = StorageConfig()
        assert storage_config.chunk_size == "64MB"
        assert storage_config.compression_algorithm == "zstd"

    def test_erasure_coding_defaults(self):
        """Test erasure coding configuration defaults."""
        ec_config = ErasureCodingConfig()
        assert ec_config.data_shards == 4
        assert ec_config.parity_shards == 2
        assert ec_config.storage_class == "standard"
