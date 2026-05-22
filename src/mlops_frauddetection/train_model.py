"""model training entrypoint

Wires both notebook pipelines into a single CLI so ``make train`` runs
without Jupyter:

  Pipeline A (Musaddiq) — LR + SMOTE, 4-class labels
  Pipeline B (Israail)  — LR / Random Forest / LightGBM / XGBoost, binary

Usage
-----
    python -m mlops_frauddetection.train_model  # train all
    python -m mlops_frauddetection.train_model.py \\
        experiment=lr_only		    # Runs logistic Regression + SMOTE Only
    python -m mlops_frauddetection.train_model.py \\
        experiment=ensemble_only		# Runs ensemble models only
    python -m mlops_frauddetection.train_model --pipeline lr
    python -m mlops_frauddetection.train_model --pipeline ensemble
    python -m mlops_frauddetection.train_model --no-smote
    python -m mlops_frauddetection.train_model --n-estimators-rf 100
"""

from __future__ import annotations

import json
import logging
import time
import warnings

# Imports and logging setup
# import argparse - replaced by hydra
from datetime import datetime
from pathlib import Path

# Enable logging
import hydra
import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.core.hydra_config import HydraConfig
from imblearn.over_sampling import SMOTE
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_frauddetection.config import CONFIG_DIR, MODELS_DIR
from mlops_frauddetection.logging_config import (
    get_logger,
    get_progress,
    setup_logging,
    timer,
)
from mlops_frauddetection.utils.seed import set_seed

# Suppress MLflow deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress MLflow verbose logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

Mlflow_track = "sqlite:///mlflow.db"
Mlflow_name = "fraud-anomaly-detection"

logger = get_logger(__name__)

RANDOM_STATE = 42

# Pipeline A (Musaddiq) constants
lr_label_names_m = {
    0: "TT-Legit-Normal",
    1: "TF-Legit-Suspicious",
    2: "FT-Fraud-LowRisk",
    3: "FF-Fraud-HighRisk",
}

clean_numeric_columns = [
    "merchant_risk_30_day",
    "merchant_risk_7_day",
    "trans_time_is_night",
    "trans_date_is_weekend",
    "customer_num_trans_1_day",
    "amt",
    "avg_amt_per_customer",
]

bool_cols = [
    "gender_M",
    "category_food_dining",
    "category_gas_transport",
    "category_grocery_net",
    "category_grocery_pos",
    "category_health_fitness",
    "category_home",
    "category_kids_pets",
    "category_misc_net",
    "category_misc_pos",
    "category_personal_care",
    "category_shopping_net",
    "category_shopping_pos",
    "category_travel",
]

dropping_cols = ["dob", "trans_date", "trans_time"]

# Pipeline B (Israail) constants
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


# data loading


def load_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load preprocessed train/test CSVs.

    Args:
        data_path: Directory containing x_train.csv, y_train.csv,
                   x_test.csv, y_test.csv.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    logger.info("Loading data from %s", data_path)
    x_train = pd.read_csv(data_path / "x_train.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    x_test = pd.read_csv(data_path / "x_test.csv")
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()
    logger.info("x_train %s  x_test %s", x_train.shape, x_test.shape)
    logger.info(
        "Train fraud: %d (%.2f%%)  Test fraud: %d (%.2f%%)",
        int(y_train.sum()),
        float(y_train.mean()) * 100,
        int(y_test.sum()),
        float(y_test.mean()) * 100,
    )
    return x_train, y_train, x_test, y_test


# cleaning


def _clean_data_lr(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip spaces, coerce numerics, map booleans, drop date cols.

    Args:
        x_train: Raw training features.
        x_test:  Raw test features.

    Returns:
        Cleaned (x_train, x_test).
    """
    x_train.columns = x_train.columns.str.strip()
    x_test.columns = x_test.columns.str.strip()

    for col in clean_numeric_columns:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
        median_val = x_train[col].median()
        x_train[col] = x_train[col].fillna(median_val)
        x_test[col] = x_test[col].fillna(median_val)

    x_train = x_train.drop(columns=dropping_cols, errors="ignore")
    x_test = x_test.drop(columns=dropping_cols, errors="ignore")

    for col in bool_cols:
        if col in x_train.columns:
            if x_train[col].dtype == object:
                x_train[col] = x_train[col].str.strip().map({"True": 1, "False": 0})
                x_test[col] = x_test[col].str.strip().map({"True": 1, "False": 0})
            else:
                x_train[col] = x_train[col].fillna(0).astype(int)
                x_test[col] = x_test[col].fillna(0).astype(int)

    logger.info(
        "LR cleaning — NaN train: %d  NaN test: %d",
        int(x_train.isna().sum().sum()),
        int(x_test.isna().sum().sum()),
    )
    return x_train, x_test


def _create_4class_labels(features: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Convert binary is_fraud into 4-class fraud risk labels.

    Classes
    -------
    0 — TT: Legit Normal      (not fraud, no suspicious signals)
    1 — TF: Legit Suspicious  (not fraud but warning signals present)
    2 — FT: Fraud Low Risk    (fraud at merchant with risk_30 < 17)
    3 — FF: Fraud High Risk   (fraud at merchant with risk_30 >= 17)

    Args:
        X: Cleaned feature matrix.
        y: Binary fraud series (0=legit, 1=fraud).

    Returns:
        pd.Series of integer labels 0-3.
    """
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
    return labels


def _add_features_lr(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 ratio-based features for the LR pipeline.

    New columns: amt_ratio, combined_risk, amt_risk_score,
                 is_high_spend, night_high_amt.

    Args:
        X: Cleaned feature matrix.

    Returns:
        Copy of X with 5 additional columns.
    """
    df = df.copy()
    df["amt_ratio"] = df["amt"] / (df["avg_amt_per_customer"] + 1)
    df["combined_risk"] = (
        df["merchant_risk_30_day"] * 0.6 + df["merchant_risk_7_day"] * 0.4
    )
    df["amt_risk_score"] = df["amt_ratio"] * df["merchant_risk_30_day"]
    df["is_high_spend"] = (df["amt"] > df["avg_amt_per_customer"] * 1.5).astype(int)
    df["night_high_amt"] = (
        (df["trans_time_is_night"] == 1) & (df["amt"] > 100)
    ).astype(int)
    return df


def _run_stratified_cv(
    model: object,
    features: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Stratified k-fold CV returning (mean_weighted_f1, std_weighted_f1).

    Args:
        model:    Unfitted estimator.
        X:        Feature matrix.
        y:        Label series.
        n_splits: Number of folds.

    Returns:
        (mean_f1, std_f1).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, features, y, cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    for i, s in enumerate(scores, 1):
        logger.info("  Fold %d: %.4f", i, s)
    logger.info("  Mean F1: %.4f  Std: %.4f", scores.mean(), scores.std())
    return float(scores.mean()), float(scores.std())


def train_lr_pipeline(
    data_path: Path,
    model_dir: Path,
    max_iter: int = 1000,
    seed: int = 42,
    run_smote: bool = True,
) -> None:
    """Pipeline A — LR + SMOTE, 4-class labels (Musaddiq).

    Steps: load → clean → 4-class labels → feature engineering →
    Model 1 (LR balanced) → Model 2 (SMOTE+LR) → evaluate → save → MLflow.

    Args:
        data_path: Directory with processed CSVs.
        model_dir: Output directory for artefacts.
        max_iter:  Max iterations for LogisticRegression solver.
        seed:      Random seed.
        run_smote: If True also train the SMOTE variant.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE A — Logistic Regression (Musaddiq)")
    logger.info("=" * 60)

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    mlflow.set_tracking_uri(Mlflow_track)
    mlflow.set_experiment(Mlflow_name)

    stages_a = [
        "Load data",
        "LR data cleaning",
        "4-class label creation",
        "LR feature engineering",
        "Model 1: LR balanced",
        "Model 2: SMOTE + LR",
        "Save metadata",
    ]

    with get_progress() as progress:
        task = progress.add_task("Pipeline A", total=len(stages_a))

        # Stage 1 — Load data
        progress.update(task, description=stages_a[0])
        x_train, y_train, x_test, y_test = load_data(data_path)
        progress.advance(task)

        # Stage 2 — Clean data
        progress.update(task, description=stages_a[1])
        with timer(logger, "LR data cleaning"):
            x_train, x_test = _clean_data_lr(x_train.copy(), x_test.copy())
        progress.advance(task)

        # Stage 3 — 4-class labels
        progress.update(task, description=stages_a[2])
        logger.info("Building 4-class labels ...")
        with timer(logger, "4-class label creation"):
            y_train_4 = _create_4class_labels(x_train, y_train)
            y_test_4 = _create_4class_labels(x_test, y_test)
        for cls, name in lr_label_names_m.items():
            cnt = int((y_train_4 == cls).sum())
            logger.info(
                "  Class %d (%s): %d rows (%.2f%%)",
                cls,
                name,
                cnt,
                cnt / len(y_train_4) * 100,
            )
        progress.advance(task)

        # Stage 4 — Feature engineering
        progress.update(task, description=stages_a[3])
        logger.info("Feature engineering ...")
        with timer(logger, "LR feature engineering"):
            x_tr = _add_features_lr(x_train)
            x_ts = _add_features_lr(x_test)
        logger.info("Shape: %s -> %s", x_train.shape, x_tr.shape)
        progress.advance(task)

        # Stage 5 — Model 1: LR balanced
        progress.update(task, description=stages_a[4])
        logger.info("Model 1: LR balanced")
        model_lr = LogisticRegression(
            max_iter=max_iter, class_weight="balanced", random_state=seed, n_jobs=-1
        )
        t0 = time.time()
        m1_cv_mean, m1_cv_std = _run_stratified_cv(model_lr, x_tr, y_train_4)
        logger.info("CV in %.1fs", time.time() - t0)
        model_lr.fit(x_tr, y_train_4)
        y_pred_m1 = model_lr.predict(x_ts)
        m1_train_acc = accuracy_score(y_train_4, model_lr.predict(x_tr))
        m1_test_acc = accuracy_score(y_test_4, y_pred_m1)
        m1_test_f1 = f1_score(y_test_4, y_pred_m1, average="weighted")
        logger.info(
            "Model 1 — Train: %.4f  Test acc: %.4f  F1: %.4f",
            m1_train_acc,
            m1_test_acc,
            m1_test_f1,
        )
        logger.info(
            "\n%s",
            classification_report(
                y_test_4, y_pred_m1, target_names=list(lr_label_names_m.values())
            ),
        )
        m1_path = model_dir / f"lr_balanced_{timestamp}.joblib"
        joblib.dump(model_lr, m1_path)
        with mlflow.start_run(run_name=f"LR_balanced_4class_{timestamp}"):
            mlflow.log_params(
                {
                    "model": "LR_balanced",
                    "class_weight": "balanced",
                    "max_iter": max_iter,
                    "seed": seed,
                    "smote": False,
                    "cv_folds": 5,
                    "pipeline": "A",
                }
            )
            mlflow.log_metrics(
                {
                    "cv_mean_f1": m1_cv_mean,
                    "cv_std_f1": m1_cv_std,
                    "train_acc": m1_train_acc,
                    "test_acc": m1_test_acc,
                    "test_f1": m1_test_f1,
                }
            )
            mlflow.sklearn.log_model(model_lr, "lr_balanced")
            mlflow.log_artifact(str(m1_path))
            cm = confusion_matrix(y_test_4, y_pred_m1)
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(
                cm, display_labels=list(lr_label_names_m.values())
            ).plot(ax=ax)
            plt.tight_layout()
            plt.savefig("reports/figures/mlflow_cm_lr_balanced.png")
            mlflow.log_artifact("reports/figures/mlflow_cm_lr_balanced.png")
            plt.close(fig)
        logger.info("Model 1 saved and logged -> %s", m1_path)
        progress.advance(task)

        # Stage 6 — Model 2: SMOTE + LR
        m2_cv_mean = m2_cv_std = m2_train_acc = m2_test_acc = m2_test_f1 = None
        m2_path = None
        progress.update(task, description=stages_a[5])
        if run_smote:
            logger.info("Model 2: SMOTE + LR")
            smote = SMOTE(random_state=seed, k_neighbors=5)
            t0 = time.time()
            x_smote, y_smote = smote.fit_resample(x_tr, y_train_4)
            logger.info("SMOTE in %.1fs — size: %s", time.time() - t0, x_smote.shape)
            model_smote = LogisticRegression(
                max_iter=max_iter, class_weight="balanced", random_state=seed, n_jobs=-1
            )
            t0 = time.time()
            m2_cv_mean, m2_cv_std = _run_stratified_cv(model_smote, x_smote, y_smote)
            logger.info("CV in %.1fs", time.time() - t0)
            model_smote.fit(x_smote, y_smote)
            y_pred_m2 = model_smote.predict(x_ts)
            m2_train_acc = accuracy_score(y_smote, model_smote.predict(x_smote))
            m2_test_acc = accuracy_score(y_test_4, y_pred_m2)
            m2_test_f1 = f1_score(y_test_4, y_pred_m2, average="weighted")
            logger.info(
                "Model 2 — Train: %.4f  Test acc: %.4f  F1: %.4f",
                m2_train_acc,
                m2_test_acc,
                m2_test_f1,
            )
            logger.info(
                "\n%s",
                classification_report(
                    y_test_4, y_pred_m2, target_names=list(lr_label_names_m.values())
                ),
            )
            m2_path = model_dir / f"lr_smote_{timestamp}.joblib"
            joblib.dump(model_smote, m2_path)
            with mlflow.start_run(run_name=f"LR_SMOTE_4class_{timestamp}"):
                mlflow.log_params(
                    {
                        "model": "LR_SMOTE",
                        "class_weight": "balanced",
                        "max_iter": max_iter,
                        "seed": seed,
                        "smote": True,
                        "smote_strategy": "auto",
                        "cv_folds": 5,
                        "pipeline": "A",
                    }
                )
                mlflow.log_metrics(
                    {
                        "cv_mean_f1": m2_cv_mean,
                        "cv_std_f1": m2_cv_std,
                        "train_acc": m2_train_acc,
                        "test_acc": m2_test_acc,
                        "test_f1": m2_test_f1,
                    }
                )
                mlflow.sklearn.log_model(model_smote, "lr_smote")
                mlflow.log_artifact(str(m2_path))
            logger.info("Model 2 saved and logged -> %s", m2_path)
        progress.advance(task)

        # Stage 7 — Save metadata
        progress.update(task, description=stages_a[6])
        meta: dict = {
            "dataset": "Credit Card Fraud Detection - Kaggle",
            "dataset_version": "DVC tracked - data/processed/",
            "training_date": timestamp,
            "train_samples": int(x_tr.shape[0]),
            "test_samples": int(x_ts.shape[0]),
            "features": int(x_tr.shape[1]),
            "classes": lr_label_names_m,
            "reproducibility": {
                "random_seed": seed,
                "cv_strategy": "StratifiedKFold(n_splits=5)",
            },
            "model_1": {
                "name": "LR_balanced",
                "max_iter": max_iter,
                "cv_f1_mean": round(m1_cv_mean, 4),
                "cv_f1_std": round(m1_cv_std, 4),
                "train_acc": round(m1_train_acc, 4),
                "test_acc": round(m1_test_acc, 4),
                "test_f1": round(m1_test_f1, 4),
                "file": str(m1_path),
            },
        }
        if (
            run_smote
            and m2_path
            and all(
                v is not None
                for v in [m2_cv_mean, m2_cv_std, m2_train_acc, m2_test_acc, m2_test_f1]
            )
        ):
            meta["model_2"] = {
                "name": "LR_SMOTE",
                "max_iter": max_iter,
                "cv_f1_mean": round(m2_cv_mean, 4),  # type: ignore[arg-type]
                "cv_f1_std": round(m2_cv_std, 4),  # type: ignore[arg-type]
                "train_acc": round(m2_train_acc, 4),  # type: ignore[arg-type]
                "test_acc": round(m2_test_acc, 4),  # type: ignore[arg-type]
                "test_f1": round(m2_test_f1, 4),  # type: ignore[arg-type]
                "file": str(m2_path),
            }
        with open(model_dir / "LR_SMOTE_model_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        progress.advance(task)

    logger.info("Pipeline A complete")


# feature engineering


def _engineer_features_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraud-related features for the ensemble pipeline (Israail).

    New columns: log_amt, category_is_online, online_x_log_amt,
                 velocity_ratio, trans_hour_sin, trans_hour_cos.

    Args:
        df: Raw feature DataFrame.

    Returns:
        Copy of df with 6 additional columns.
    """
    df = df.copy()
    df["log_amt"] = np.log1p(df["amt"])

    online_cols = ["category_shopping_net", "category_misc_net", "category_grocery_net"]
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
    return df


def _build_preprocessor(
    x_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build ColumnTransformer (StandardScaler for numeric, passthrough for cat).

    Args:
        x_train: Training features after feature engineering.

    Returns:
        (unfitted_preprocessor, numeric_features, categorical_features).
    """
    numeric_features = [f for f in num_features if f in x_train.columns]
    categorical_features = [
        col
        for col in x_train.columns
        if (col.startswith("category_") or col.startswith("gender_"))
        and col not in numeric_features
    ]
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
    return preprocessor, numeric_features, categorical_features


def train_ensemble_pipeline(
    data_path: Path,
    model_dir: Path,
    seed: int = 42,
    n_estimators_rf: int = 200,
    n_estimators_lgb: int = 500,
    n_estimators_xgb: int = 500,
) -> None:
    """Pipeline B — RF / LightGBM / XGBoost binary fraud detection (Israail).

    Steps: load -> feature engineering -> preprocessing -> SMOTE(0.3) ->
    train 4 models -> 5-fold TimeSeriesSplit CV -> evaluate -> save -> MLflow.

    Args:
        data_path:        Directory with processed CSVs.
        model_dir:        Output directory for artefacts.
        seed:             Random seed.
        n_estimators_rf:  Trees for Random Forest.
        n_estimators_lgb: Estimators for LightGBM.
        n_estimators_xgb: Estimators for XGBoost.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE B — Ensemble Models (Israail)")
    logger.info("=" * 60)

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    mlflow.set_tracking_uri(Mlflow_track)
    mlflow.set_experiment(Mlflow_name)

    stages_b = [
        "Load data",
        "Ensemble feature engineering",
        "Preprocessing",
        "SMOTE resampling",
        "Train ensemble models",
        "Save metadata",
    ]

    with get_progress() as progress:
        task = progress.add_task("Pipeline B", total=len(stages_b))

        # Stage 1 — Load data
        progress.update(task, description=stages_b[0])
        x_train, y_train, x_test, y_test = load_data(data_path)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train["is_fraud"]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test["is_fraud"]
        progress.advance(task)

        # Stage 2 — Feature engineering
        progress.update(task, description=stages_b[1])
        logger.info("Feature engineering ...")
        with timer(logger, "Ensemble feature engineering"):
            x_train = _engineer_features_ensemble(x_train)
            x_test = _engineer_features_ensemble(x_test)
        preprocessor, num_feats, cat_feats = _build_preprocessor(x_train)
        all_feats = num_feats + cat_feats
        x_train = x_train[all_feats].copy()
        x_test = x_test[all_feats].copy()
        progress.advance(task)

        # Stage 3 — Preprocessing
        progress.update(task, description=stages_b[2])
        logger.info("Preprocessing %d features ...", len(all_feats))
        with timer(logger, "Preprocessing"):
            x_train_prep = preprocessor.fit_transform(x_train)
            x_test_prep = preprocessor.transform(x_test)
        progress.advance(task)

        # Stage 4 — SMOTE
        progress.update(task, description=stages_b[3])
        logger.info("Applying SMOTE (sampling_strategy=0.3) ...")
        class_counts_before = {
            int(k): int(v)
            for k, v in pd.Series(y_train).value_counts().sort_index().items()
        }
        logger.info("Before SMOTE class counts: %s", class_counts_before)
        with timer(logger, "SMOTE resampling"):
            smote = SMOTE(random_state=seed, sampling_strategy=0.3)
            x_train_res, y_train_res = smote.fit_resample(x_train_prep, y_train)
        class_counts_after = {
            int(k): int(v)
            for k, v in pd.Series(y_train_res).value_counts().sort_index().items()
        }
        logger.info("After SMOTE class counts: %s", class_counts_after)
        scale_pos = int((y_train_res == 0).sum() / (y_train_res == 1).sum())
        progress.advance(task)

        # Stage 5 — Train ensemble models
        progress.update(task, description=stages_b[4])
        ensemble_models = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=n_estimators_rf,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=n_estimators_lgb,
                class_weight="balanced",
                learning_rate=0.05,
                num_leaves=63,
                random_state=seed,
                verbose=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=n_estimators_xgb,
                scale_pos_weight=scale_pos,
                learning_rate=0.05,
                max_depth=6,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            ),
        }
        tscv = TimeSeriesSplit(n_splits=5)
        all_results: dict = {}
        model_task = progress.add_task("Training models", total=len(ensemble_models))
        for name, model in ensemble_models.items():
            progress.update(model_task, description=f"Training {name}")
            logger.info("Training %s ...", name)
            t0 = time.time()
            model.fit(x_train_res, y_train_res)
            logger.info("  Done in %.1fs", time.time() - t0)
            y_pred = model.predict(x_test_prep)
            y_pred_prob = model.predict_proba(x_test_prep)[:, 1]
            train_acc = accuracy_score(y_train_res, model.predict(x_train_res))
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            avg_prec = average_precision_score(y_test, y_pred_prob)
            logger.info(
                "  %s — Train: %.4f  F1: %.4f  ROC: %.4f  AP: %.4f",
                name,
                train_acc,
                test_f1,
                roc_auc,
                avg_prec,
            )
            cv_scores = cross_val_score(
                model, x_train_res, y_train_res, cv=tscv, scoring="f1", n_jobs=-1
            )
            logger.info(
                "  CV F1 mean=%.4f  std=%.4f", cv_scores.mean(), cv_scores.std()
            )
            all_results[name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "f1": test_f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc_auc,
                "avg_prec": avg_prec,
                "cv_mean_f1": float(cv_scores.mean()),
                "cv_std_f1": float(cv_scores.std()),
            }
            fname = name.lower() + f"_{timestamp}.joblib"
            mpath = model_dir / fname
            joblib.dump(model, mpath)
            with mlflow.start_run(run_name=f"{name}_{timestamp}"):
                mlflow.log_params(
                    {
                        "model": name,
                        "seed": seed,
                        "smote_strategy": 0.3,
                        "cv_folds": 5,
                        "pipeline": "B",
                    }
                )
                mlflow.log_metrics(
                    {
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "f1": test_f1,
                        "precision": prec,
                        "recall": rec,
                        "roc_auc": roc_auc,
                        "avg_prec": avg_prec,
                        "cv_mean_f1": float(cv_scores.mean()),
                        "cv_std_f1": float(cv_scores.std()),
                    }
                )
                mlflow.sklearn.log_model(model, name.lower())
                mlflow.log_artifact(str(mpath))
            logger.info("  %s saved and logged -> %s", name, mpath)
            progress.advance(model_task)
        progress.advance(task)

        # Stage 6 — Save metadata
        progress.update(task, description=stages_b[5])
        for name, res in all_results.items():
            model_key = name.lower()
            meta_path = model_dir / f"RGhazzal_{model_key}_metadata.json"
            existing: dict = {}
            if meta_path.exists():
                with open(meta_path) as fh:
                    existing = json.load(fh)
            existing[timestamp] = {
                "dataset": "Credit Card Fraud Detection - Kaggle",
                "training_date": timestamp,
                "train_samples": int(x_train_res.shape[0]),
                "test_samples": int(x_test_prep.shape[0]),
                "features": int(x_train_res.shape[1]),
                "smote_strategy": 0.3,
                "seed": seed,
                **res,
            }
            with open(meta_path, "w") as fh:
                json.dump(existing, fh, indent=2)
        progress.advance(task)

    logger.info("Pipeline B results:")
    logger.info("  %-22s  F1      ROC-AUC  AvgPrec", "Model")
    for name, res in all_results.items():
        logger.info(
            "  %-22s  %.4f  %.4f   %.4f",
            name,
            res["f1"],
            res["roc_auc"],
            res["avg_prec"],
        )
    logger.info("Pipeline B complete")


def _validate_config(cfg: DictConfig) -> None:
    """Validate Hydra config values before training begins."""
    errors = []

    if cfg.project.seed < 0:
        errors.append("project.seed must be non-negative")
    if not Path(cfg.data.processed_path).exists():
        errors.append(f"data.processed_path not found: {cfg.data.processed_path}")
    if not (0 < cfg.data.test_size < 1):
        errors.append("data.test_size must be between 0 and 1")
    if cfg.training.pipeline not in ("all", "lr", "ensemble"):
        errors.append(
            f"training.pipeline must be all/lr/ensemble, got: {cfg.training.pipeline}"
        )
    if cfg.training.cv_folds < 2:
        errors.append("training.cv_folds must be >= 2")
    if cfg.model.lr.max_iter < 100:
        errors.append("model.lr.max_iter must be >= 100")
    if cfg.model.ensemble.n_estimators_rf < 10:
        errors.append("model.ensemble.n_estimators_rf must be >= 10")
    if not (0 < cfg.model.ensemble.smote_strategy <= 1):
        errors.append("model.ensemble.smote_strategy must be between 0 and 1")
    if cfg.model.ensemble.learning_rate <= 0:
        errors.append("model.ensemble.learning_rate must be positive")

    if errors:
        for err in errors:
            logger.error("Config validation error: %s", err)
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s). "
            "Set HYDRA_FULL_ERROR=1 for full details."
        )
    logger.info("Config validation passed")


# CLI


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """CLI entry point for training pipelines."""
    setup_logging()
    _validate_config(cfg)
    set_seed(cfg.project.seed)

    hydra_cfg = HydraConfig.get()
    experiment_name = hydra_cfg.runtime.choices.get("experiment", "none")

    logger.info("=" * 72)
    logger.info("Starting fraud detection training run")
    logger.info("Hydra config loaded from: %s", CONFIG_DIR / "config.yaml")
    logger.info("Hydra experiment loaded: %s", experiment_name)
    logger.info("Project: %s", cfg.project.name)
    logger.info("Training pipeline selected: %s", cfg.training.pipeline)
    logger.info("SMOTE enabled: %s", cfg.training.smote)
    logger.info("Random seed: %s", cfg.project.seed)
    logger.debug("Full Hydra config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("=" * 72)

    if cfg.training.pipeline in ("all", "lr"):
        train_lr_pipeline(
            data_path=Path(cfg.data.processed_path),
            model_dir=MODELS_DIR,
            max_iter=cfg.model.lr.max_iter,
            seed=cfg.project.seed,
            run_smote=cfg.training.smote,
        )

    if cfg.training.pipeline in ("all", "ensemble"):
        train_ensemble_pipeline(
            data_path=Path(cfg.data.processed_path),
            model_dir=MODELS_DIR,
            seed=cfg.project.seed,
            n_estimators_rf=cfg.model.ensemble.n_estimators_rf,
            n_estimators_lgb=cfg.model.ensemble.n_estimators_lgb,
            n_estimators_xgb=cfg.model.ensemble.n_estimators_xgb,
        )

    logger.info("=" * 72)
    logger.info("Training run complete")
    logger.info("Models saved to: %s", MODELS_DIR)
    logger.info("MLflow tracking URI: %s", Mlflow_track)
    logger.info("MLflow experiment: %s", Mlflow_name)
    logger.info("=" * 72)


# Run the CLI
if __name__ == "__main__":
    main()
