"""Configuration management for the secure platform."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PKIConfig(BaseModel):
    """PKI configuration."""

    ca_cert_path: Path = Field(default=Path("/certs/ca.crt"))
    ca_key_path: Path = Field(default=Path("/certs/ca.key"))
    cert_validity_days: int = Field(default=365)
    key_size: int = Field(default=4096)
    rotation_threshold_days: int = Field(default=30)


class AuthzConfig(BaseModel):
    """Authorization configuration."""

    policy_path: Path = Field(default=Path("/policies"))
    default_effect: Literal["allow", "deny"] = Field(default="deny")
    cache_ttl_seconds: int = Field(default=60)
    max_cache_size: int = Field(default=10000)


class AuditConfig(BaseModel):
    """Audit log configuration."""

    log_path: Path = Field(default=Path("/audit/audit.log"))
    merkle_tree_path: Path = Field(default=Path("/audit/merkle.db"))
    max_log_size_mb: int = Field(default=100)
    retention_days: int = Field(default=365)


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0")
    grpc_port: int = Field(default=8443)
    metrics_port: int = Field(default=8003)
    require_mtls: bool = Field(default=True)


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")
    otel_endpoint: str | None = Field(default=None)
    otel_service_name: str = Field(default="secure_platform")


class Config(BaseSettings):
    """Main configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SECURE_PLATFORM_",
        env_nested_delimiter="__",
    )

    pki: PKIConfig = Field(default_factory=PKIConfig)
    authz: AuthzConfig = Field(default_factory=AuthzConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.pki.ca_cert_path.parent.mkdir(parents=True, exist_ok=True)
        self.authz.policy_path.mkdir(parents=True, exist_ok=True)
        self.audit.log_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    """Get the global configuration."""
    config = Config()
    config.ensure_directories()
    return config
