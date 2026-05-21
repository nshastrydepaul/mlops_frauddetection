"""Data loading and cleaning utilities.

Handles all I/O and cleaning for both pipelines:
  - Pipeline A (Musaddiq): LR with 4-class labels
  - Pipeline B (Israail):  Ensemble binary models
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops_frauddetection.config import PROCESSED_DATA_DIR
from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)

#cleaning for Pipeline 
clean_num_col = [
    "merchant_risk_30_day", "merchant_risk_7_day",
    "trans_time_is_night", "trans_date_is_weekend",
    "customer_num_trans_1_day", "amt", "avg_amt_per_customer",
]

bool_col = [
    "gender_M", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos",
    "category_health_fitness", "category_home", "category_kids_pets",
    "category_misc_net", "category_misc_pos", "category_personal_care",
    "category_shopping_net", "category_shopping_pos", "category_travel",
]

dropped = ["dob", "trans_date", "trans_time"]


def load_data(
    data_path: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load preprocessed train/test CSVs from data_path.

    Args:
        data_path: Directory containing X_train.csv, y_train.csv,
                   X_test.csv, y_test.csv.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    logger.info("Loading data from %s", data_path)
    X_train = pd.read_csv(data_path / "X_train.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    X_test  = pd.read_csv(data_path / "X_test.csv")
    y_test  = pd.read_csv(data_path / "y_test.csv").squeeze()
    logger.info("X_train %s  X_test %s", X_train.shape, X_test.shape)
    logger.info(
        "Train fraud: %d (%.2f%%)  Test fraud: %d (%.2f%%)",
        int(y_train.sum()), float(y_train.mean()) * 100,
        int(y_test.sum()),  float(y_test.mean()) * 100,
    )
    return X_train, y_train, X_test, y_test


def clean_data_lr(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean data for Pipeline A (LR 4-class pipeline).

    Steps:
      1. Striping whitespace from column names
      2. filling NaN with training median
      3. Droping columns not needed for LR
      4. Mapping boolean string columns ('True'/'False') to 1/0

    Args:
        X_train: Raw training features.
        X_test:  Raw test features.

    Returns:
        Cleaned (X_train, X_test).
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train.columns = X_train.columns.str.strip()
    X_test.columns  = X_test.columns.str.strip()

    for col in clean_num_col:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")
        median_val   = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

    X_train = X_train.drop(columns=dropped, errors="ignore")
    X_test  = X_test.drop(columns=dropped,  errors="ignore")

    for col in bool_col:
        if col in X_train.columns:
            if X_train[col].dtype == object:
                X_train[col] = X_train[col].str.strip().map({"True": 1, "False": 0})
                X_test[col]  = X_test[col].str.strip().map({"True": 1, "False": 0})
            else:
                X_train[col] = X_train[col].fillna(0).astype(int)
                X_test[col]  = X_test[col].fillna(0).astype(int)

    logger.info(
        "Cleaning done — NaN train: %d  NaN test: %d",
        int(X_train.isna().sum().sum()),
        int(X_test.isna().sum().sum()),
    )
    return X_train, X_test


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/raw` directory."""
    path = RAW_DATA_DIR / filename
    logger.info("Loading raw data: %s", path)
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/processed` directory."""
    path = PROCESSED_DATA_DIR / filename
    logger.info("Loading processed data: %s", path)
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a dataframe to `data/processed`, creating the directory if needed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed data: %s", path)
    return path