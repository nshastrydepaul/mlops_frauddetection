"""Model inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlops_frauddetection.config import MODELS_DIR, PROCESSED_DATA_DIR
from mlops_frauddetection.logging_config import (
    get_logger,
    get_progress,
    setup_logging,
    timer,
)

logger = get_logger(__name__)


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Load a trained model and write predictions for input_path to output_path.

    Args:
        model_path:  Path to the saved model file.
        input_path:  Path to the input CSV to score.
        output_path: Path to write predictions CSV.
    """
    stages = [
        "Load model",
        "Load input data",
        "Preprocess",
        "Feature engineering",
        "Select features",
        "Inference",
        "Save predictions",
    ]

    with get_progress() as progress:
        task = progress.add_task("Prediction pipeline", total=len(stages))

        # Stage 1 — Load model
        progress.update(task, description=stages[0])
        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info("Loading model from %s", model_path)
        with timer(logger, "Model load"):
            model = joblib.load(model_path)
        logger.info("Model loaded — type: %s", type(model).__name__)
        progress.advance(task)

        # Stage 2 — Load input data
        progress.update(task, description=stages[1])
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            raise FileNotFoundError(f"Input file not found: {input_path}")
        logger.info("Loading input data from %s", input_path)
        with timer(logger, "Input data load"):
            df = pd.read_csv(input_path)
        logger.info("Input shape: %s", df.shape)
        progress.advance(task)

        # Stage 3 — Preprocess
        progress.update(task, description=stages[2])
        raw_cols = [
            "dob",
            "trans_date",
            "trans_time",
            "trans_num",
            "first",
            "last",
            "street",
            "city",
            "state",
            "zip",
            "job",
            "merchant",
            "trans_time_secs",
            "customer_txn_count",
            "merchant_txn_count",
        ]
        df = df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")
        for col in df.select_dtypes(include="object").columns:
            logger.warning("Dropping unconverted string column: %s", col)
            df = df.drop(columns=[col])
        logger.info("Cleaned input shape: %s", df.shape)
        progress.advance(task)

        # Stage 4 — Feature engineering
        progress.update(task, description=stages[3])
        logger.info("Applying ensemble feature engineering ...")
        with timer(logger, "Feature engineering"):
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
        logger.info("Feature engineering complete — shape: %s", df.shape)
        progress.advance(task)

        # Stage 5 — Select features
        progress.update(task, description=stages[4])
        num_features = [
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
        cat_features = [
            col
            for col in df.columns
            if (col.startswith("category_") or col.startswith("gender_"))
            and col not in num_features
        ]
        all_feats = num_features + cat_features
        missing = [f for f in all_feats if f not in df.columns]
        if missing:
            logger.warning("Missing expected features: %s", missing)
        df = df[[f for f in all_feats if f in df.columns]]
        logger.info("Final input shape for model: %s", df.shape)
        progress.advance(task)

        # Stage 6 — Inference
        progress.update(task, description=stages[5])
        logger.info("Running inference on %d rows ...", len(df))
        with timer(logger, "Inference"):
            preds = model.predict(df)
        fraud_count = int(preds.sum())
        fraud_rate = float(preds.mean()) * 100
        logger.info(
            "Predictions complete — fraud: %d / %d (%.2f%%)",
            fraud_count,
            len(df),
            fraud_rate,
        )
        progress.advance(task)

        # Stage 7 — Save predictions
        progress.update(task, description=stages[6])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing predictions to %s", output_path)
        with timer(logger, "Save predictions"):
            pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
        logger.info("Predictions saved — %d rows written", len(preds))
        progress.advance(task)


def main() -> None:
    """CLI entrypoint for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Generate predictions from a trained model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "logistic_regression_20260507.joblib",
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "X_test.csv")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    setup_logging()

    try:
        predict(args.model_path, args.input, args.output)
        logger.info("Prediction pipeline complete")
    except FileNotFoundError as e:
        logger.critical("Prediction aborted — file not found: %s", e)
        raise SystemExit(1) from e
    except Exception as e:
        logger.critical("Prediction aborted — unexpected error: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
