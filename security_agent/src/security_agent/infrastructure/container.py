"""Dependency injection container for Security Agent."""

from dataclasses import dataclass
from typing import Any

import structlog
from opentelemetry import trace

from security_agent.infrastructure.config import Config, get_config
from security_agent.infrastructure.logging import setup_logging, get_logger
from security_agent.infrastructure.metrics import get_metrics
from security_agent.infrastructure.tracing import setup_tracing, get_tracer


@dataclass
class Container:
    """Dependency injection container for security agent components."""

    config: Config
    logger: structlog.stdlib.BoundLogger
    tracer: trace.Tracer
    metrics: Any  # SecurityAgentMetrics

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
            "security_agent_container_initialized",
            environment=config.observability.environment,
            ml_enabled=config.detection.enable_ml_detection,
            auto_response=config.response.enable_auto_response,
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
