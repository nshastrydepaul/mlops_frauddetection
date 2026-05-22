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


def create_4class_labels(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Convert binary is_fraud into 4-class fraud risk labels.

    Classes
    0 — TT: Legit Normal      (not fraud, no suspicious signals)
    1 — TF: Legit Suspicious  (not fraud but warning signals present)
    2 — FT: Fraud Low Risk    (fraud at low-risk merchant, risk_30 < 17)
    3 — FF: Fraud High Risk   (fraud at high-risk merchant, risk_30 >= 17)

    The FT/FF split uses merchant_risk_30_day >= 17, giving a near-50/50
    split (428 FT vs 459 FF on the 80k training set).

    Args:
        X: Cleaned feature matrix (must have amt, avg_amt_per_customer,
           merchant_risk_30_day, trans_time_is_night columns).
        y: Binary fraud series (0 = legit, 1 = fraud).

    Returns:
        pd.Series of integer labels 0-3 with name 'risk_label'.
    """
    amt = X["amt"]
    cus_avg_amt = X["avg_amt_per_customer"]
    risk_lvl30 = X["merchant_risk_30_day"]
    is_night = X["trans_time_is_night"]
    is_fraud = y.values

    # TF: legit but any one suspicious signal present
    suspicious = (
        (risk_lvl30 >= 21) | (amt > cus_avg_amt * 2) | ((is_night == 1) & (amt > 200))
    )

    # FT vs FF: split fraud by merchant risk (balanced ~48/52 split)
    high_risk_merchant = risk_lvl30 >= 17

    labels = pd.Series(0, index=X.index, name="risk_label")
    labels[(is_fraud == 0) & suspicious] = 1
    labels[(is_fraud == 1) & ~high_risk_merchant] = 2
    labels[(is_fraud == 1) & high_risk_merchant] = 3

    logger.info("4-class label distribution:")
    for cls, name in lbls.items():
        cnt = int((labels == cls).sum())
        logger.info(
            "  Class %d (%s): %d rows (%.2f%%)", cls, name, cnt, cnt / len(labels) * 100
        )
    return labels


def add_features_lr(X: pd.DataFrame) -> pd.DataFrame:
    """Add 5 ratio-based engineered features for the LR pipeline.

    These features improve class separation, especially between TF and FF:
      - TF has amt/avg ratio ~1.74 vs TT ~0.61
      - FF has high merchant_risk AND high amt_ratio

    New columns
    amt_ratio      — Transaction amount / (customer average spend + 1)
    combined_risk  — Weighted blend: risk_30 * 0.6 + risk_7 * 0.4
    amt_risk_score — amt_ratio * merchant_risk_30_day interaction
    is_high_spend  — 1 if amt > 1.5x customer average, else 0
    night_high_amt — 1 if nighttime transaction AND amt > $100, else 0


    Returning:
        Copy of X with 5 additional columns.
    """
    X = X.copy()
    X["amt_ratio"] = X["amt"] / (X["avg_amt_per_customer"] + 1)
    X["combined_risk"] = (
        X["merchant_risk_30_day"] * 0.6 + X["merchant_risk_7_day"] * 0.4
    )
    X["amt_risk_score"] = X["amt_ratio"] * X["merchant_risk_30_day"]
    X["is_high_spend"] = (X["amt"] > X["avg_amt_per_customer"] * 1.5).astype(int)
    X["night_high_amt"] = ((X["trans_time_is_night"] == 1) & (X["amt"] > 100)).astype(
        int
    )

    logger.info("Added 5 LR features — new shape: %s", X.shape)
    return X


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

    Returning:
        Copy of df with 6 additional columns.
    """
    df = df.copy()
    df["log_amt"] = np.log1p(df["amt"])

    online_cols = [
        "category_shopping_net",
        "category_misc_net",
        "category_grocery_net",
    ]
    for col in online_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip().map({"True": 1, "False": 0})

    df["category_is_online"] = df[online_cols].any(axis=1).astype(int)
    df["online_x_log_amt"] = df["category_is_online"] * df["log_amt"]
    df["velocity_ratio"] = df["customer_num_trans_1_day"] / (
        df["customer_num_trans_30_day"] + 1
    )
    df["trans_hour_sin"] = np.sin(2 * np.pi * df["trans_time_hrs"] / 24)
    df["trans_hour_cos"] = np.cos(2 * np.pi * df["trans_time_hrs"] / 24)

    logger.info("Added 6 ensemble features — new shape: %s", df.shape)
    return df


def build_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Building sklearn ColumnTransformer for the ensemble pipeline.

    Apply StandardScaler to numeric features and passes
    one-hot encoded features through unchanged.
    """
    numeric_features = [f for f in num_feat if f in X_train.columns]
    categorical_features = [
        col
        for col in X_train.columns
        if (col.startswith("category_") or col.startswith("gender_"))
        and col not in numeric_features
    ]
    # removing dups
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))
    categorical_features = [
        c for c in categorical_features if c not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    logger.info(
        "Preprocessor built — %d numeric, %d categorical",
        len(numeric_features),
        len(categorical_features),
    )
    return preprocessor, numeric_features, categorical_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe."""

    logger.info("Building features for %d rows", len(df))

    df = df.copy()

    # Customer behavior

    if "cc_num" in df.columns:
        df["customer_txn_count"] = df.groupby("cc_num")["cc_num"].transform("count")

        if "amt" in df.columns:
            df["avg_amt_per_customer"] = df.groupby("cc_num")["amt"].transform("mean")

    # Merchant behavior

    if "merchant" in df.columns:
        df["merchant_txn_count"] = df.groupby("merchant")["merchant"].transform("count")

    logger.info("Feature engineering complete")

    return df
