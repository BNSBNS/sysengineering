"""Configuration for streaming system."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrokerConfig(BaseModel):
    """Broker configuration."""

    broker_id: int = Field(default=1)
    host: str = Field(default="localhost")
    port: int = Field(default=9092)
    raft_port: int = Field(default=9093)


class StorageConfig(BaseModel):
    """Log storage configuration."""

    segment_size: str = Field(default="1GB")
    retention_hours: int = Field(default=168)  # 7 days
    compaction_enabled: bool = Field(default=True)


class RaftConfig(BaseModel):
    """Raft consensus configuration."""

    election_timeout_min_ms: int = Field(default=150)
    election_timeout_max_ms: int = Field(default=300)
    heartbeat_interval_ms: int = Field(default=50)
    snapshot_threshold: int = Field(default=10000)


class ReplicationConfig(BaseModel):
    """Replication configuration."""

    min_isr: int = Field(default=2)
    replication_factor: int = Field(default=3)
    ack_mode: Literal["leader", "all"] = Field(default="all")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0")
    client_port: int = Field(default=9092)
    grpc_port: int = Field(default=50055)
    metrics_port: int = Field(default=8005)


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")
    otel_endpoint: str | None = Field(default=None)


class Config(BaseSettings):
    """Main configuration."""

    model_config = SettingsConfigDict(env_prefix="STREAMING_", env_nested_delimiter="__")

    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    raft: RaftConfig = Field(default_factory=RaftConfig)
    replication: ReplicationConfig = Field(default_factory=ReplicationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def ensure_directories(self) -> None:
        self.log.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    config = Config()
    config.ensure_directories()
    return config
