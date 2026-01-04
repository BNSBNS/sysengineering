"""Configuration management for the vector database.

Reference:
    - HNSW parameters: Malkov & Yashunin (2018), arXiv:1603.09320
    - IVF parameters: FAISS, arXiv:1702.08734
    - PQ parameters: Jégou et al. (2011), IEEE TPAMI
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HNSWConfig(BaseModel):
    """HNSW index configuration.

    Parameters follow recommendations from Malkov & Yashunin (2018).
    Reference: arXiv:1603.09320
    """

    M: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Max connections per node (paper recommends 12-48)",
    )
    M_max_0: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Max connections at layer 0 (typically 2*M)",
    )
    ef_construction: int = Field(
        default=200,
        ge=10,
        le=2000,
        description="Beam width during index construction",
    )
    ef_search: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Beam width during search (higher = better recall, slower)",
    )
    mL: float = Field(
        default=0.0,  # Computed from M if not set
        ge=0.0,
        description="Level multiplier (default: 1/ln(M))",
    )

    @field_validator("mL", mode="before")
    @classmethod
    def compute_ml(cls, v: float, info) -> float:
        """Compute mL from M if not explicitly set."""
        if v == 0.0 or v is None:
            M = info.data.get("M", 16)
            return 1.0 / math.log(M)
        return v


class IVFConfig(BaseModel):
    """IVF index configuration.

    Parameters follow recommendations from FAISS paper.
    Reference: arXiv:1702.08734
    """

    nlist: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Number of Voronoi cells (clusters)",
    )
    nprobe: int = Field(
        default=10,
        ge=1,
        description="Number of cells to search (higher = better recall, slower)",
    )
    training_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="K-means iterations for training",
    )
    spherical: bool = Field(
        default=False,
        description="Normalize vectors to unit sphere",
    )

    @classmethod
    def for_dataset_size(cls, n: int, target_recall: float = 0.95) -> "IVFConfig":
        """Compute optimal parameters for dataset size.

        Rule of thumb from FAISS: nlist ≈ sqrt(n)
        """
        nlist = max(1, int(math.sqrt(n)))
        # 5% of clusters for ~95% recall
        nprobe_ratio = 0.05 if target_recall >= 0.95 else 0.01
        nprobe = max(1, int(nlist * nprobe_ratio))
        return cls(nlist=nlist, nprobe=nprobe)


class PQConfig(BaseModel):
    """Product Quantization configuration.

    Parameters follow recommendations from Jégou et al. (2011).
    Reference: IEEE TPAMI, DOI: 10.1109/TPAMI.2010.57
    """

    M: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of subquantizers (dimension must be divisible by M)",
    )
    Ks: int = Field(
        default=256,
        ge=2,
        le=65536,
        description="Centroids per subquantizer (256 = 8-bit codes)",
    )
    training_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="K-means iterations for codebook training",
    )

    @field_validator("Ks")
    @classmethod
    def validate_ks_power_of_two(cls, v: int) -> int:
        """Validate Ks is a power of 2 for efficient encoding."""
        if v & (v - 1) != 0:
            raise ValueError(f"Ks must be a power of 2, got {v}")
        return v


class StorageConfig(BaseModel):
    """Vector storage configuration."""

    data_dir: Path = Field(
        default=Path("/var/lib/vector_db/data"),
        description="Data directory path",
    )
    index_dir: Path = Field(
        default=Path("/var/lib/vector_db/index"),
        description="Index directory path",
    )
    use_mmap: bool = Field(
        default=True,
        description="Use memory-mapped files for vector storage",
    )
    max_vectors: int = Field(
        default=10_000_000,
        ge=1,
        description="Maximum vectors per index",
    )


class GPUConfig(BaseModel):
    """GPU acceleration configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable GPU acceleration",
    )
    device_id: int = Field(
        default=0,
        ge=0,
        description="CUDA device ID",
    )
    batch_size: int = Field(
        default=1024,
        ge=1,
        description="Batch size for GPU operations",
    )
    memory_limit_mb: int = Field(
        default=4096,
        ge=256,
        description="GPU memory limit in MB",
    )


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8008, ge=1, le=65535, description="REST API port")
    grpc_port: int = Field(default=50058, ge=1, le=65535, description="gRPC port")
    metrics_port: int = Field(
        default=8009, ge=1, le=65535, description="Prometheus metrics port"
    )


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log format"
    )
    otel_endpoint: str | None = Field(
        default=None, description="OpenTelemetry collector endpoint"
    )
    otel_service_name: str = Field(
        default="vector_db", description="Service name for tracing"
    )


class Config(BaseSettings):
    """Main configuration for the vector database."""

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_DB_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    ivf: IVFConfig = Field(default_factory=IVFConfig)
    pq: PQConfig = Field(default_factory=PQConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def ensure_directories(self) -> None:
        """Ensure data and index directories exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    """Get the global configuration instance."""
    config = Config()
    config.ensure_directories()
    return config
