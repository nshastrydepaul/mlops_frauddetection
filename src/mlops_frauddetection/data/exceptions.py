# exceptions.py
"""Custom exceptions for the data pipeline."""

from __future__ import annotations


class RawDataNotFoundError(Exception):
    """Raised when the expected raw data file is not found."""

    pass


class DataLoaderError(Exception):
    """Raised when a data loading or saving operation fails."""

    pass


class VisualizationError(Exception):
    """Raised when a visualization cannot be created or saved."""

    pass
