"""Dataset loading utilities.

Wrappers around pandas/numpy I/O that resolve paths against the
project's configured data directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mlops_frauddetection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_frauddetection.data.exceptions import DataLoaderError
from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/raw` directory."""
    path = os.path.join(RAW_DATA_DIR, filename)
    try:
        logger.info(f"Loading raw data: {path}")
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded raw data: {path} — shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Raw data file not found: {path}")
        raise DataLoaderError(f"Raw data file not found: {path}") from None
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse raw data file {path}: {e}")
        raise DataLoaderError(f"Failed to parse raw data file: {path}") from e
    except OSError as e:
        logger.error(f"OS error while reading raw data file {path}: {e}")
        raise DataLoaderError(f"OS error while reading: {path}") from e


def load_processed(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/processed` directory."""
    path = PROCESSED_DATA_DIR / filename
    try:
        logger.info(f"Loading processed data: {path}")
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded processed data: {path} — shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Processed data file not found: {path}")
        raise DataLoaderError(f"Processed data file not found: {path}") from None
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse processed data file {path}: {e}")
        raise DataLoaderError(f"Failed to parse processed data file: {path}") from e
    except OSError as e:
        logger.error(f"OS error while reading processed data file {path}: {e}")
        raise DataLoaderError(f"OS error while reading: {path}") from e


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a dataframe to `data/processed`, creating the directory if needed."""
    path = PROCESSED_DATA_DIR / filename
    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved processed data: {path} — shape: {df.shape}")
        return path
    except OSError as e:
        logger.error(f"Failed to save processed data to {path}: {e}")
        raise DataLoaderError(f"Failed to save processed data: {path}") from e
