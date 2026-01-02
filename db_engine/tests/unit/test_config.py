"""Unit tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from db_engine.infrastructure.config import Config, StorageConfig, WALConfig, get_config


@pytest.mark.unit
class TestConfig:
    """Tests for Config class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()

        assert config.storage.page_size == 4096
        assert config.storage.buffer_pool_size == 1073741824  # 1GB
        assert config.wal.segment_size == 67108864  # 64MB
        assert config.wal.sync_mode == "fsync"
        assert config.server.port == 5432
        assert config.server.max_connections == 100

    def test_custom_storage_config(self, temp_dir: Path) -> None:
        """Test custom storage configuration."""
        storage = StorageConfig(
            data_dir=temp_dir / "data",
            wal_dir=temp_dir / "wal",
            page_size=8192,
            buffer_pool_size=2147483648,  # 2GB
        )

        assert storage.page_size == 8192
        assert storage.buffer_pool_size == 2147483648

    def test_ensure_directories(self, temp_dir: Path) -> None:
        """Test that ensure_directories creates required directories."""
        config = Config(
            storage=StorageConfig(
                data_dir=temp_dir / "data",
                wal_dir=temp_dir / "wal",
            )
        )

        config.ensure_directories()

        assert config.storage.data_dir.exists()
        assert config.storage.wal_dir.exists()

    def test_invalid_page_size(self) -> None:
        """Test that invalid page size raises validation error."""
        with pytest.raises(ValueError):
            StorageConfig(page_size=1024)  # Too small

    def test_wal_sync_modes(self) -> None:
        """Test valid WAL sync modes."""
        for mode in ["fsync", "fdatasync", "none"]:
            wal = WALConfig(sync_mode=mode)  # type: ignore
            assert wal.sync_mode == mode


@pytest.mark.unit
class TestConfigSingleton:
    """Tests for get_config singleton."""

    def test_get_config_returns_same_instance(self) -> None:
        """Test that get_config returns the same instance."""
        # Note: This test may interfere with other tests due to caching
        # In production, use dependency injection instead
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
