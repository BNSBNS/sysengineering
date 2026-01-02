"""Configuration management for Security Agent using Pydantic Settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EBPFConfig(BaseSettings):
    """eBPF probe configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_EBPF_")

    enable_syscall_tracing: bool = True
    enable_network_tracing: bool = True
    enable_file_tracing: bool = True
    ring_buffer_size: int = Field(default=16384, description="Ring buffer size in pages")
    perf_buffer_size: int = Field(default=64, description="Perf buffer size in pages")


class DetectionConfig(BaseSettings):
    """Threat detection configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_DETECTION_")

    enable_ml_detection: bool = True
    anomaly_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    rule_engine_enabled: bool = True
    rules_path: str = "/etc/security_agent/rules"
    baseline_learning_period_hours: int = 24


class ResponseConfig(BaseSettings):
    """Automated response configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_RESPONSE_")

    enable_auto_response: bool = False
    enable_process_kill: bool = False
    enable_network_isolation: bool = False
    enable_file_quarantine: bool = True
    quarantine_path: str = "/var/lib/security_agent/quarantine"
    response_cooldown_seconds: int = 60


class AlertConfig(BaseSettings):
    """Alert and notification configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_ALERT_")

    enable_siem_forwarding: bool = True
    siem_endpoint: str = ""
    enable_slack_notifications: bool = False
    slack_webhook_url: str = ""
    alert_dedup_window_seconds: int = 300


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_OBSERVABILITY_")

    log_level: str = "info"
    log_format: str = "json"
    metrics_port: int = 8005
    otlp_endpoint: str = ""
    environment: str = "development"


class Config(BaseSettings):
    """Root configuration for Security Agent."""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    ebpf: EBPFConfig = Field(default_factory=EBPFConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config()
