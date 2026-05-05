"""Feature engineering transformations."""

from __future__ import annotations

import pandas as pd

from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)


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
