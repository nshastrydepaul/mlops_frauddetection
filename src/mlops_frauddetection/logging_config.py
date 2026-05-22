"""Centralized logging configuration for the ML pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Literal, TypeVar

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mlops_frauddetection.config import LOGS_DIR

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_DATEFMT = "%X"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

_APP_LOG_MAX_BYTES = 5_000_000
_APP_LOG_BACKUP_COUNT = 5
_ERROR_LOG_MAX_BYTES = 2_000_000
_ERROR_LOG_BACKUP_COUNT = 3

T = TypeVar("T")


def setup_logging(level: LogLevel = "INFO") -> None:
    """Configure root logging with Rich console output and rotating log files.

    This function is idempotent. It can be called multiple times safely because
    existing handlers are removed before the new logging configuration is applied.

    Handlers:
        - RichHandler: colorized terminal logs with readable tracebacks.
        - logs/app.log: rotating file log for all messages at the configured level.
        - logs/error.log: rotating file log for ERROR and CRITICAL messages only.

    Args:
        level: Minimum logging level for console and application logs.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    rich_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_path=True,
        markup=True,
        log_time_format=_DEFAULT_DATEFMT,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(
        logging.Formatter(fmt="%(message)s", datefmt=_DEFAULT_DATEFMT)
    )

    app_handler = RotatingFileHandler(
        LOGS_DIR / "app.log",
        maxBytes=_APP_LOG_MAX_BYTES,
        backupCount=_APP_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_handler.setLevel(level)
    app_handler.setFormatter(logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATEFMT))

    error_handler = RotatingFileHandler(
        LOGS_DIR / "error.log",
        maxBytes=_ERROR_LOG_MAX_BYTES,
        backupCount=_ERROR_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATEFMT)
    )

    root.addHandler(rich_handler)
    root.addHandler(app_handler)
    root.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def get_progress() -> Progress:
    """Return a reusable Rich progress bar for long-running pipeline steps."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


@contextmanager
def timer(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log elapsed runtime for a code block.

    Args:
        logger: Logger used to write the timing message.
        label: Description of the timed operation.

    Example:
        with timer(logger, "Feature engineering"):
            df = build_features(df)

        Logs:
            Feature engineering completed in 3.42s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", label, elapsed)


def track(
    iterable: Iterable[T],
    description: str = "Processing",
    total: int | None = None,
) -> Iterable[T]:
    """Wrap an iterable with a Rich progress bar.

    Args:
        iterable: Iterable object to track.
        description: Label shown next to the progress bar.
        total: Total item count if not inferable from the iterable.

    Yields:
        Items from the original iterable.
    """
    with get_progress() as progress:
        yield from progress.track(iterable, total=total, description=description)
