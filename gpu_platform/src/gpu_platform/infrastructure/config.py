"""Configuration for GPU platform."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DiscoveryConfig(BaseModel):
    """GPU discovery configuration."""

    poll_interval_seconds: int = Field(default=30)
    enable_health_checks: bool = Field(default=True)
    detect_ecc_errors: bool = Field(default=True)


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    algorithm: Literal["numa_aware", "gang", "binpack"] = Field(default="numa_aware")
    enable_preemption: bool = Field(default=False)
    max_queue_size: int = Field(default=1000)
    placement_timeout_seconds: int = Field(default=60)


class AllocationConfig(BaseModel):
    """Allocation configuration."""

    allow_oversubscription: bool = Field(default=False)
    enable_mps: bool = Field(default=False)
    enable_mig: bool = Field(default=False)
    default_memory_fraction: float = Field(default=1.0)


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0")
    grpc_port: int = Field(default=50054)
    metrics_port: int = Field(default=8004)


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")
    otel_endpoint: str | None = Field(default=None)


class Config(BaseSettings):
    """Main configuration."""

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_", env_nested_delimiter="__")

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


@lru_cache
def get_config() -> Config:
    return Config()
