"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO", log_format: str = "json") -> None:
    """Set up structured logging."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper()))

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **context: Any) -> structlog.BoundLogger:
    """Get a bound logger."""
    logger = structlog.get_logger(name)
    return logger.bind(**context) if context else logger
