"""Dependency injection container for GPU Platform."""

from dataclasses import dataclass
from typing import Any

import structlog
from opentelemetry import trace

from gpu_platform.infrastructure.config import Config, get_config
from gpu_platform.infrastructure.logging import setup_logging, get_logger
from gpu_platform.infrastructure.metrics import get_metrics
from gpu_platform.infrastructure.tracing import setup_tracing, get_tracer


@dataclass
class Container:
    """Dependency injection container for GPU platform components."""

    config: Config
    logger: structlog.stdlib.BoundLogger
    tracer: trace.Tracer
    metrics: Any  # GPUPlatformMetrics

    _instance: "Container | None" = None

    @classmethod
    def create(cls) -> "Container":
        """Create and initialize the container with all dependencies."""
        if cls._instance is not None:
            return cls._instance

        config = get_config()
        logger = setup_logging()
        tracer = setup_tracing()
        metrics = get_metrics()

        cls._instance = cls(
            config=config,
            logger=logger,
            tracer=tracer,
            metrics=metrics,
        )

        logger.info(
            "gpu_platform_container_initialized",
            environment=config.observability.environment,
        )

        return cls._instance

    @classmethod
    def get(cls) -> "Container":
        """Get the singleton container instance."""
        if cls._instance is None:
            return cls.create()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the container (useful for testing)."""
        cls._instance = None


def get_container() -> Container:
    """Get the dependency injection container."""
    return Container.get()
