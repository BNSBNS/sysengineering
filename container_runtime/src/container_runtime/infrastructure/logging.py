"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(level: str = "INFO", log_format: str = "json") -> None:
    """Set up structured logging with structlog."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        processors: list[Processor] = [*shared_processors, structlog.processors.JSONRenderer()]
    else:
        processors = [*shared_processors, structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """Get a bound logger instance."""
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
