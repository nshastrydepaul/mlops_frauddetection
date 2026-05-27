"""
Shared pytest fixtures for mlops_frauddetection.

Imports from your real modules:
  - mlops_frauddetection.features.build_features  (num_feat)
  - mlops_frauddetection.models.fraud_models       (get_lr_balanced)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Synthetic DataFrame matching your real processed schema ───────────────


@pytest.fixture(scope="session")
def raw_transaction_df() -> pd.DataFrame:
    """2000-row synthetic DataFrame matching your real transaction schema."""
    np.random.seed(42)
    n = 2000
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range("2023-01-01", periods=n, freq="h"),
            "cc_num": np.random.randint(1_000_000, 9_999_999, n),
            "trans_num": [f"txn_{i:06d}" for i in range(n)],
            "amt": np.abs(np.random.normal(50, 30, n)).clip(min=0.01),
            "merchant": np.random.choice(["merchant_A", "merchant_B", "merchant_C"], n),
            "category": np.random.choice(
                [
                    "grocery_pos",
                    "entertainment",
                    "gas_transport",
                    "shopping_net",
                    "food_dining",
                    "misc_net",
                ],
                n,
            ),
            "lat": np.random.uniform(25.0, 48.0, n),
            "long": np.random.uniform(-125.0, -65.0, n),
            "city": np.random.choice(["Springfield", "Chicago", "Austin"], n),
            "state": np.random.choice(["CA", "TX", "NY", "IL"], n),
            "zip": np.random.randint(10000, 99999, n),
            "city_pop": np.random.randint(1000, 500_000, n),
            "merch_lat": np.random.uniform(25.0, 48.0, n),
            "merch_long": np.random.uniform(-125.0, -65.0, n),
            "first": ["John"] * n,
            "last": ["Doe"] * n,
            "gender": np.random.choice(["M", "F"], n),
            "street": ["123 Main St"] * n,
            "job": np.random.choice(["engineer", "teacher", "doctor"], n),
            "dob": pd.date_range("1960-01-01", periods=n, freq="30D"),
            "unix_time": np.arange(1_672_531_200, 1_672_531_200 + n * 3600, 3600),
            "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )


@pytest.fixture(scope="session")
def engineered_feature_df(raw_transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features your build_features pipeline expects."""
    df = raw_transaction_df.copy()
    n = len(df)

    df["log_amt"] = np.log1p(df["amt"])
    df["trans_time_hrs"] = df["trans_date_trans_time"].dt.hour
    df["trans_hour_sin"] = np.sin(2 * np.pi * df["trans_time_hrs"] / 24)
    df["trans_hour_cos"] = np.cos(2 * np.pi * df["trans_time_hrs"] / 24)
    df["trans_time_is_night"] = (
        (df["trans_time_hrs"] < 6) | (df["trans_time_hrs"] >= 22)
    ).astype(int)
    df["trans_date_is_weekend"] = (
        df["trans_date_trans_time"].dt.dayofweek.isin([5, 6])
    ).astype(int)
    df["trans_time_day"] = df["trans_date_trans_time"].dt.dayofweek
    df["category_is_online"] = (
        df["category"].isin(["shopping_net", "misc_net", "grocery_net"]).astype(int)
    )

    for w in [1, 7, 30]:
        df[f"customer_num_trans_{w}_day"] = np.random.randint(1, 20, n)
        df[f"customer_avg_amout_{w}_day"] = np.abs(np.random.normal(50, 20, n))
        df[f"merchant_num_trans_{w}_day"] = np.random.randint(5, 100, n)
        df[f"merchant_risk_{w}_day"] = np.random.uniform(0, 0.3, n)

    # aliases used in num_feat
    df["customer_avg_amount_7_day"] = df["customer_avg_amout_7_day"]
    df["customer_avg_amount_30_day"] = df["customer_avg_amout_30_day"]
    df["merchant_risk_90_day"] = np.random.uniform(0, 0.3, n)
    df["online_x_log_amt"] = df["category_is_online"] * df["log_amt"]
    df["velocity_ratio"] = df["customer_num_trans_1_day"] / (
        df["customer_num_trans_7_day"] + 1e-9
    )
    return df


@pytest.fixture(scope="session")
def binary_x_y(engineered_feature_df: pd.DataFrame) -> tuple:
    """Feature matrix X (using your real num_feat list) and binary target y."""
    from mlops_frauddetection.features.build_features import num_feat

    available = [c for c in num_feat if c in engineered_feature_df.columns]
    x = engineered_feature_df[available].values
    y = engineered_feature_df["is_fraud"].values
    return x, y


@pytest.fixture(scope="session")
def train_test_data(binary_x_y: tuple) -> tuple:  # type: ignore[return]
    """Stratified 80/20 split ready for model tests."""
    from sklearn.model_selection import train_test_split

    x, y = binary_x_y
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def trained_lr_balanced(train_test_data: tuple) -> Pipeline:
    """Fitted Pipeline(StandardScaler + get_lr_balanced()) — your real factory."""
    from mlops_frauddetection.models.fraud_models import get_lr_balanced

    X_train, _, y_train, _ = train_test_data
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", get_lr_balanced())])
    pipe.fit(X_train, y_train)
    return pipe


@pytest.fixture
def tmp_model_path(tmp_path: Path) -> Path:
    return tmp_path / "model.joblib"
