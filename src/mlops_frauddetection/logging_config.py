"""Centralized logging configuration."""

from __future__ import annotations

import logging
from typing import Literal

from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# _DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_DATEFMT = "%X"  # Rich uses time-only by convention


# def setup_logging(level: LogLevel = "INFO", fmt: str = _DEFAULT_FORMAT) -> None:
#     """Configure the root logger for the application.

#     Idempotent: safe to call multiple times; re-applies the handler config.
#     """
#     root = logging.getLogger()
#     root.setLevel(level)

#     for handler in list(root.handlers):
#         root.removeHandler(handler)

#     handler = logging.StreamHandler(stream=sys.stdout)
#     handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=_DEFAULT_DATEFMT))
#     root.addHandler(handler)


# def get_logger(name: str) -> logging.Logger:
#     """Return a module-level logger."""
#     return logging.getLogger(name)


def setup_logging(level: LogLevel = "INFO") -> None:
    """Configure the root logger with Rich handler.

    Idempotent: safe to call multiple times; re-applies the handler config.

    Args:
        level: Minimum logging level (default: INFO).
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = RichHandler(
        rich_tracebacks=True,  # Syntax-highlighted exception tracebacks
        tracebacks_show_locals=True,  # Show local variables on exception
        show_path=True,  # Show file:line next to each log entry
        markup=True,  # Allow [bold]/[red] etc. in log messages
        log_time_format=_DEFAULT_DATEFMT,
    )

    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
