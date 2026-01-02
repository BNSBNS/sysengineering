"""Configuration management for the database engine."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseModel):
    """Storage configuration."""

    data_dir: Path = Field(default=Path("/data"), description="Data directory path")
    wal_dir: Path = Field(default=Path("/wal"), description="WAL directory path")
    page_size: int = Field(default=4096, ge=4096, le=65536, description="Page size in bytes")
    buffer_pool_size: int = Field(
        default=1073741824, ge=1048576, description="Buffer pool size in bytes (default 1GB)"
    )


class WALConfig(BaseModel):
    """Write-Ahead Log configuration."""

    segment_size: int = Field(
        default=67108864, ge=1048576, description="WAL segment size in bytes (default 64MB)"
    )
    sync_mode: Literal["fsync", "fdatasync", "none"] = Field(
        default="fsync", description="WAL sync mode"
    )
    checkpoint_interval_seconds: int = Field(
        default=300, ge=10, description="Checkpoint interval in seconds"
    )


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5432, ge=1, le=65535, description="Server port")
    grpc_port: int = Field(default=50051, ge=1, le=65535, description="gRPC port")
    metrics_port: int = Field(default=8001, ge=1, le=65535, description="Prometheus metrics port")
    max_connections: int = Field(default=100, ge=1, le=10000, description="Max connections")
    query_timeout_seconds: int = Field(default=300, ge=1, description="Query timeout in seconds")


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_format: Literal["json", "console"] = Field(default="json", description="Log format")
    otel_endpoint: str | None = Field(
        default=None, description="OpenTelemetry collector endpoint"
    )
    otel_service_name: str = Field(default="db_engine", description="Service name for tracing")


class Config(BaseSettings):
    """Main configuration for the database engine."""

    model_config = SettingsConfigDict(
        env_prefix="DB_ENGINE_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    storage: StorageConfig = Field(default_factory=StorageConfig)
    wal: WALConfig = Field(default_factory=WALConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def ensure_directories(self) -> None:
        """Ensure data and WAL directories exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.wal_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    """Get the global configuration instance."""
    config = Config()
    config.ensure_directories()
    return config
