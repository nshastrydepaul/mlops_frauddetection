"""Raw-to-processed data pipeline entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_frauddetection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_frauddetection.data.exceptions import RawDataNotFoundError
from mlops_frauddetection.data.loaders import load_raw, save_processed
from mlops_frauddetection.features.build_features import build_features
from mlops_frauddetection.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def raw_data_exists(input_file: str) -> bool:
    """Check if raw data file exists.

    Args:
        input_file: The filename to search for in RAW_DATA_DIR.

    Returns:
        True if the file exists.

    Raises:
        RawDataNotFoundError: If the directory or expected file is not found.
    """
    try:
        if not os.path.exists(RAW_DATA_DIR):
            raise RawDataNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

        logger.info(f"Raw data directory exists: {RAW_DATA_DIR}")

        for file in os.listdir(RAW_DATA_DIR):
            if os.path.splitext(file)[1] == ".csv":
                logger.info(f"Found raw data file: {file}")
                if file == input_file:
                    logger.info(f"Sourced raw data file found: {file}")
                    return True

        available = os.listdir(RAW_DATA_DIR)
        raise RawDataNotFoundError(
            f"Sourced raw data file '{input_file}' not found in '{RAW_DATA_DIR}'. "
            f"Available: {available}"
        )

    except RawDataNotFoundError:
        logger.error(f"Raw data file not found: {input_file}")
        raise
    except OSError as e:
        logger.error(f"Failed to access raw data directory {RAW_DATA_DIR}: {e}")
        raise RawDataNotFoundError(f"Directory access error: {RAW_DATA_DIR}") from e


def process_data(input_dir: Path, input_file: str, output_dir: Path) -> None:
    """Transform raw data into processed datasets (train/test)."""

    if input_file is None:
        logger.error("No input file specified for raw data processing")
        raise RawDataNotFoundError("Input file must be specified")

    raw_data_exists(input_file)  # Raises RawDataNotFoundError if missing

    logger.info(f"Loading raw data from path:{input_dir}")

    df = load_raw(input_file)

    logger.info(f"Initial raw dataset load shape: {df.shape}")

    # DROP ONLY agreed columns
    drop_cols = [
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "unix_time",
        "merch_lat",
        "merch_long",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    logger.info(f"Dropped agreed columns: {drop_cols}")
    logger.info(f"After dropping columns: {df.shape}")

    # FEATURE ENGINEERING
    df = build_features(df)
    logger.info(
        "Built features using build_features function - see features/build_features.py"
    )
    logger.info(f"Data shape after feature engineering: {df.shape}")
    logger.info("Feature engineering complete")

    # DROP ID / LEAKAGE COLUMNS
    drop_id_cols = [
        "ssn",
        "cc_num",
        "acct_num",
        "first",
        "last",
        "trans_num",
        "profile",
    ]
    df = df.drop(columns=[c for c in drop_id_cols if c in df.columns], errors="ignore")
    logger.info(f"Data shape after dropping ID columns: {df.shape}")

    # HOT-ENCODING
    high_card_cols = ["merchant", "job"]
    for col in high_card_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if df[col].nunique() < 50]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.info(f"Data shape after hot-encoding:  {df.shape}")

    # SPLITTING DATA INTO TRAIN/TEST
    x = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    logger.info("Train/Test split completed — shapes")
    logger.info(f" x_train: {x_train.shape}, x_test: {x_test.shape}")
    logger.info(f" y_train: {y_train.shape}, y_test: {y_test.shape}")

    # SAVE FILE TO PROCESSED_DATA_DIR
    logging.info(f"Saving processed datasets to output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_processed(x_train, "X_train.csv")
    save_processed(x_test, "X_test.csv")
    save_processed(y_train.to_frame(), "y_train.csv")
    save_processed(y_test.to_frame(), "y_test.csv")
    logger.info(f"Processed data saved successfully to output directory: {output_dir}")


# CLI ENTRYPOINT
def main() -> None:
    """CLI entrypoint for data processing."""
    parser = argparse.ArgumentParser(
        description="Process raw data into model inputs and split into train/test sets."
    )
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--input-file", type=str, default="data_100k.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    setup_logging()

    try:
        process_data(args.input, args.input_file, args.output)
        logger.info(
            "Data processing pipeline complete — processed datasets ready for modeling"
        )
    except RawDataNotFoundError as e:
        logger.critical(f"Pipeline aborted — raw data unavailable: {e}")
        raise SystemExit(1) from e


# RUN THE PIPELINE FROM THE COMMAND LINE + ALLOQ IMPORT AS MODULE
if __name__ == "__main__":
    main()
