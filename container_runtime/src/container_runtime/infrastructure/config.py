"""Configuration management for the container runtime."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeConfig(BaseModel):
    """Container runtime configuration."""

    root_dir: Path = Field(default=Path("/var/lib/containers"), description="Container root dir")
    state_dir: Path = Field(default=Path("/run/containers"), description="Runtime state dir")
    image_dir: Path = Field(default=Path("/var/lib/containers/images"), description="Image storage")


class CgroupConfig(BaseModel):
    """Cgroups configuration."""

    version: Literal["v1", "v2"] = Field(default="v2", description="Cgroup version")
    root_path: Path = Field(default=Path("/sys/fs/cgroup"), description="Cgroup mount point")
    default_cpu_shares: int = Field(default=1024, description="Default CPU shares")
    default_memory_limit: int = Field(default=536870912, description="Default memory limit (512MB)")


class NetworkConfig(BaseModel):
    """Network configuration."""

    bridge_name: str = Field(default="ctr0", description="Bridge network name")
    bridge_subnet: str = Field(default="172.20.0.0/16", description="Bridge subnet")
    enable_nat: bool = Field(default=True, description="Enable NAT for containers")


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    algorithm: Literal["binpack", "spread", "random"] = Field(
        default="binpack", description="Scheduling algorithm"
    )
    max_pending_jobs: int = Field(default=1000, description="Max pending jobs in queue")
    placement_timeout_seconds: int = Field(default=30, description="Placement timeout")


class GPUConfig(BaseModel):
    """GPU configuration."""

    enabled: bool = Field(default=True, description="Enable GPU support")
    allow_oversubscription: bool = Field(default=False, description="Allow GPU oversubscription")
    default_gpu_memory_fraction: float = Field(default=1.0, description="Default GPU memory fraction")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    grpc_port: int = Field(default=50052, ge=1, le=65535, description="gRPC port")
    metrics_port: int = Field(default=8002, ge=1, le=65535, description="Prometheus metrics port")


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")
    otel_endpoint: str | None = Field(default=None)
    otel_service_name: str = Field(default="container_runtime")


class Config(BaseSettings):
    """Main configuration for the container runtime."""

    model_config = SettingsConfigDict(
        env_prefix="CONTAINER_RUNTIME_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cgroup: CgroupConfig = Field(default_factory=CgroupConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.runtime.root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.state_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.image_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    """Get the global configuration instance."""
    config = Config()
    config.ensure_directories()
    return config
