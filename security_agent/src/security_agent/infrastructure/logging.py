"""Structured logging configuration for Security Agent."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from security_agent.infrastructure.config import get_config


def add_security_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add security-specific context to log entries."""
    event_dict["service"] = "security_agent"
    return event_dict


def setup_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging for the security agent."""
    config = get_config()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_security_context,
    ]

    if config.observability.log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(config.observability.log_level.upper())

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("bcc").setLevel(logging.WARNING)

    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with optional name binding."""
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    return logger
