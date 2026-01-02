"""Dependency injection container for Object Store."""

from dataclasses import dataclass
from typing import Any

import structlog
from opentelemetry import trace

from object_store.infrastructure.config import Config, get_config
from object_store.infrastructure.logging import setup_logging, get_logger
from object_store.infrastructure.metrics import get_metrics
from object_store.infrastructure.tracing import setup_tracing, get_tracer


@dataclass
class Container:
    """Dependency injection container for object store components."""

    config: Config
    logger: structlog.stdlib.BoundLogger
    tracer: trace.Tracer
    metrics: Any  # ObjectStoreMetrics

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
            "object_store_container_initialized",
            environment=config.observability.environment,
            dedup_enabled=config.storage.enable_deduplication,
            ec_enabled=config.erasure_coding.enabled,
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
