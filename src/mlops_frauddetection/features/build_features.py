"""Feature engineering for both fraud detection.

Pipeline A (Musaddiq) — 4-class LR:
  - create_4class_labels(): TT / TF / FT / FF label system
  - add_features_lr(): 5 ratio-based engineered features

Pipeline B (Israail) — Binary ensemble:
  - engineer_features_ensemble(): log_amt, velocity, cyclic hour features
  - build_preprocessor(): StandardScaler + passthrough ColumnTransformer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)

# 4-class label names
lbls: dict[int, str] = {
    0: "TT-Legit-Normal",
    1: "TF-Legit-Suspicious",
    2: "FT-Fraud-LowRisk",
    3: "FF-Fraud-HighRisk",
}

# Pipeline numeric features
num_feat = [
    "log_amt",
    "city_pop",
    "customer_num_trans_1_day",
    "customer_num_trans_7_day",
    "customer_num_trans_30_day",
    "customer_avg_amout_1_day",
    "customer_avg_amount_7_day",
    "customer_avg_amount_30_day",
    "merchant_num_trans_1_day",
    "merchant_num_trans_7_day",
    "merchant_num_trans_30_day",
    "merchant_risk_1_day",
    "merchant_risk_7_day",
    "merchant_risk_30_day",
    "merchant_risk_90_day",
    "trans_time_hrs",
    "trans_hour_sin",
    "trans_hour_cos",
    "trans_time_is_night",
    "trans_date_is_weekend",
    "trans_time_day",
    "category_is_online",
    "online_x_log_amt",
    "velocity_ratio",
]


# Pipeline A


def create_4class_labels(features: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Convert binary is_fraud into 4-class fraud risk labels.

    Classes
    0 — TT: Legit Normal      (not fraud, no suspicious signals)
    1 — TF: Legit Suspicious  (not fraud but warning signals present)
    2 — FT: Fraud Low Risk    (fraud at low-risk merchant, risk_30 < 17)
    3 — FF: Fraud High Risk   (fraud at high-risk merchant, risk_30 >= 17)

    Args:
        features: Cleaned feature matrix (amt, avg_amt_per_customer,
            merchant_risk_30_day, trans_time_is_night).
        y: Binary fraud series (0 = legit, 1 = fraud).

    Returns:
        pd.Series of integer labels 0-3 with name 'risk_label'.
    """
    logger.info("Creating 4-class labels for %d rows", len(features))

    required_cols = [
        "amt",
        "avg_amt_per_customer",
        "merchant_risk_30_day",
        "trans_time_is_night",
    ]
    missing = [c for c in required_cols if c not in features.columns]
    if missing:
        logger.warning("Missing expected columns for label creation: %s", missing)

    amt = features["amt"]
    cus_avg_amt = features["avg_amt_per_customer"]
    risk_lvl30 = features["merchant_risk_30_day"]
    is_night = features["trans_time_is_night"]
    is_fraud = y.values

    suspicious = (
        (risk_lvl30 >= 21) | (amt > cus_avg_amt * 2) | ((is_night == 1) & (amt > 200))
    )
    high_risk_merchant = risk_lvl30 >= 17

    labels = pd.Series(0, index=features.index, name="risk_label")
    labels[(is_fraud == 0) & suspicious] = 1
    labels[(is_fraud == 1) & ~high_risk_merchant] = 2
    labels[(is_fraud == 1) & high_risk_merchant] = 3

    logger.info("4-class label distribution:")
    for cls, name in lbls.items():
        cnt = int((labels == cls).sum())
        logger.info(
            "  Class %d (%s): %d rows (%.2f%%)",
            cls,
            name,
            cnt,
            cnt / len(labels) * 100,
        )

    logger.info("4-class label creation complete — output shape: %s", labels.shape)
    return labels


def add_features_lr(x_features: pd.DataFrame) -> pd.DataFrame:
    """Add 5 ratio-based engineered features for the LR pipeline.

    New columns
    amt_ratio      — Transaction amount / (customer average spend + 1)
    combined_risk  — Weighted blend: risk_30 * 0.6 + risk_7 * 0.4
    amt_risk_score — amt_ratio * merchant_risk_30_day interaction
    is_high_spend  — 1 if amt > 1.5x customer average, else 0
    night_high_amt — 1 if nighttime transaction AND amt > $100, else 0

    Args:
        x_features (DataFrame Attribute): Feature matrix with required columns.

    Returns:
        Copy of x_features with 5 additional columns.
    """
    logger.info("Adding LR ratio features — input shape: %s", x_features.shape)

    df = x_features.copy()
    df["amt_ratio"] = df["amt"] / (df["avg_amt_per_customer"] + 1)
    df["combined_risk"] = (
        df["merchant_risk_30_day"] * 0.6 + df["merchant_risk_7_day"] * 0.4
    )
    df["amt_risk_score"] = df["amt_ratio"] * df["merchant_risk_30_day"]
    df["is_high_spend"] = (df["amt"] > df["avg_amt_per_customer"] * 1.5).astype(int)
    df["night_high_amt"] = (
        (df["trans_time_is_night"] == 1) & (df["amt"] > 100)
    ).astype(int)
    logger.info(
        "LR feature stats — amt_ratio mean: %.2f, combined_risk mean: %.2f",
        df["amt_ratio"].mean(),
        df["combined_risk"].mean(),
    )
    logger.info("Added 5 LR features — output shape: %s", df.shape)
    return df


# Pipeline B


def engineer_features_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraud-related features for the ensemble pipeline (Israail).

    New columns
    log_amt            — Log-transformed transaction amount (log1p)
    category_is_online — 1 if any online category column is True
    online_x_log_amt   — Interaction: category_is_online * log_amt
    velocity_ratio     — customer_num_trans_1_day / (trans_30_day + 1)
    trans_hour_sin     — sine encoding of transaction hour
    trans_hour_cos     — cosine encoding of transaction hour

    Args:
        df: Raw feature DataFrame (X_train or X_test).

    Returns:
        Copy of df with 6 additional columns.
    """
    logger.info("Engineering ensemble features — input shape: %s", df.shape)

    df = df.copy()
    df["log_amt"] = np.log1p(df["amt"])
    logger.info(
        "Added log_amt — mean: %.2f, max: %.2f",
        df["log_amt"].mean(),
        df["log_amt"].max(),
    )

    online_cols = [
        "category_shopping_net",
        "category_misc_net",
        "category_grocery_net",
    ]
    missing_online = [c for c in online_cols if c not in df.columns]
    if missing_online:
        logger.warning("Missing online category columns: %s", missing_online)

    for col in online_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip().map({"True": 1, "False": 0})

    df["category_is_online"] = df[online_cols].any(axis=1).astype(int)
    logger.info(
        "Online transactions: %d (%.2f%%)",
        df["category_is_online"].sum(),
        df["category_is_online"].mean() * 100,
    )

    df["online_x_log_amt"] = df["category_is_online"] * df["log_amt"]
    df["velocity_ratio"] = df["customer_num_trans_1_day"] / (
        df["customer_num_trans_30_day"] + 1
    )
    logger.info(
        "velocity_ratio — mean: %.2f, max: %.2f",
        df["velocity_ratio"].mean(),
        df["velocity_ratio"].max(),
    )

    df["trans_hour_sin"] = np.sin(2 * np.pi * df["trans_time_hrs"] / 24)
    df["trans_hour_cos"] = np.cos(2 * np.pi * df["trans_time_hrs"] / 24)

    logger.info("Added 6 ensemble features — output shape: %s", df.shape)
    return df


def build_preprocessor(
    x_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build sklearn ColumnTransformer for the ensemble pipeline.

    Apply StandardScaler to numeric features and passes
    one-hot encoded features through unchanged.

    Args:
        X_train: Training feature matrix used to identify columns.

    Returns:
        Tuple of (preprocessor, numeric_features, categorical_features).
    """
    logger.info("Building preprocessor — input shape: %s", x_train.shape)

    numeric_features = [f for f in num_feat if f in x_train.columns]
    categorical_features = [
        col
        for col in x_train.columns
        if (col.startswith("category_") or col.startswith("gender_"))
        and col not in numeric_features
    ]

    # Remove duplicates
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))
    categorical_features = [
        c for c in categorical_features if c not in numeric_features
    ]

    missing_num = [f for f in num_feat if f not in x_train.columns]
    if missing_num:
        logger.warning(
            "Expected numeric features not found in X_train: %s", missing_num
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    logger.info(
        "Preprocessor built — %d numeric features, %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)
    return preprocessor, numeric_features, categorical_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Args:
        df: Raw dataframe to transform.

    Returns:
        DataFrame with additional engineered features.
    """
    logger.info("Starting feature engineering for %d rows", len(df))

    df = df.copy()

    # CUSTOMER BEHAVIOR FEATURES
    if "cc_num" in df.columns:
        df["customer_txn_count"] = df.groupby("cc_num")["cc_num"].transform("count")
        logger.info(
            "Added customer_txn_count — unique customers: %d", df["cc_num"].nunique()
        )

        if "amt" in df.columns:
            df["avg_amt_per_customer"] = df.groupby("cc_num")["amt"].transform("mean")
            logger.info(
                "Added avg_amt_per_customer — mean: %.2f, min: %.2f, max: %.2f",
                df["avg_amt_per_customer"].mean(),
                df["avg_amt_per_customer"].min(),
                df["avg_amt_per_customer"].max(),
            )
    else:
        logger.warning(
            "Column 'cc_num' not found — skipping customer behavior features"
        )

    # MERCHANT BEHAVIOR FEATURES
    if "merchant" in df.columns:
        df["merchant_txn_count"] = df.groupby("merchant")["merchant"].transform("count")
        logger.info(
            "Added merchant_txn_count — unique merchants: %d", df["merchant"].nunique()
        )
    else:
        logger.warning(
            "Column 'merchant' not found — skipping merchant behavior features"
        )

    added = [
        f
        for f in ["customer_txn_count", "avg_amt_per_customer", "merchant_txn_count"]
        if f in df.columns
    ]
    logger.info(
        "Feature engineering complete — added %d features: %s", len(added), added
    )
    logger.info("Output shape: %s", df.shape)

    return df
