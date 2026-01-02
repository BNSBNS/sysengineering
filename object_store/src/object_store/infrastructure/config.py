"""Configuration management for Object Store using Pydantic Settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    """Storage backend configuration."""

    model_config = SettingsConfigDict(env_prefix="OBJECT_STORE_STORAGE_")

    data_dir: str = "/var/lib/object_store/data"
    chunk_size: str = "64MB"
    max_object_size: str = "5TB"
    enable_deduplication: bool = True
    enable_compression: bool = True
    compression_algorithm: str = "zstd"


class ErasureCodingConfig(BaseSettings):
    """Erasure coding configuration."""

    model_config = SettingsConfigDict(env_prefix="OBJECT_STORE_EC_")

    enabled: bool = True
    data_shards: int = 4
    parity_shards: int = 2
    storage_class: str = "standard"  # standard, reduced_redundancy


class MetadataConfig(BaseSettings):
    """Metadata store configuration."""

    model_config = SettingsConfigDict(env_prefix="OBJECT_STORE_METADATA_")

    backend: str = "sqlite"  # sqlite, postgres
    db_path: str = "/var/lib/object_store/metadata.db"
    connection_pool_size: int = 10


class ServerConfig(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="OBJECT_STORE_SERVER_")

    host: str = "0.0.0.0"
    port: int = 9000
    workers: int = 4
    max_connections: int = 1000


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(env_prefix="OBJECT_STORE_OBSERVABILITY_")

    log_level: str = "info"
    log_format: str = "json"
    metrics_port: int = 8006
    otlp_endpoint: str = ""
    environment: str = "development"


class Config(BaseSettings):
    """Root configuration for Object Store."""

    model_config = SettingsConfigDict(
        env_prefix="OBJECT_STORE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    storage: StorageConfig = Field(default_factory=StorageConfig)
    erasure_coding: ErasureCodingConfig = Field(default_factory=ErasureCodingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config()
