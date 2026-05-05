"""Raw-to-processed data pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_frauddetection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_frauddetection.data.loaders import load_raw, save_processed
from mlops_frauddetection.features.build_features import build_features
from mlops_frauddetection.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def process_data(input_dir: Path, output_dir: Path) -> None:
    """Transform raw data into processed datasets (train/test)."""

    logger.info("Loading raw data from %s", input_dir)

    df = load_raw("data_100k.csv")

    logger.info("Initial shape: %s", df.shape)

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

    logger.info("After dropping columns: %s", df.shape)

    # FEATURE ENGINEERING
    df = build_features(df)

    logger.info("Feature engineering complete")

    # DROP ID / LEAKAGE COLUMNS (IMPORTANT)

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

    logger.info("After dropping ID columns: %s", df.shape)

    # ENCODING (STRICT CONTROL)
    # Explicitly define high-cardinality columns
    high_card_cols = ["merchant", "job"]

    # Label encode them FIRST
    for col in high_card_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Identify categorical columns AFTER encoding
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # SAFETY CHECK (VERY IMPORTANT)
    # Remove anything unexpected (extra safety)
    categorical_cols = [
        col
        for col in categorical_cols
        if df[col].nunique() < 50  # threshold
    ]

    # One-hot encode ONLY safe columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logger.info("After encoding: %s", df.shape)

    # SPLIT

    x = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    logger.info("Train/Test split completed")

    # SAVE

    output_dir.mkdir(parents=True, exist_ok=True)

    save_processed(x_train, "X_train.csv")
    save_processed(x_test, "X_test.csv")
    save_processed(y_train.to_frame(), "y_train.csv")
    save_processed(y_test.to_frame(), "y_test.csv")

    logger.info("Processed data saved successfully")


def main() -> None:
    """CLI entrypoint for data processing."""
    parser = argparse.ArgumentParser(description="Process raw data into model inputs")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    setup_logging()
    process_data(args.input, args.output)
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
