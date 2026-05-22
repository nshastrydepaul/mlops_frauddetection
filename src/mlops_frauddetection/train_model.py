"""model training entrypoint

Wires both notebook pipelines into a single CLI so ``make train`` runs
without Jupyter:

  Pipeline A (Musaddiq) — LR + SMOTE, 4-class labels
  Pipeline B (Israail)  — LR / Random Forest / LightGBM / XGBoost, binary

Usage
-----
    python -m mlops_frauddetection.train_model              # train all
    python -m mlops_frauddetection.train_model --pipeline lr
    python -m mlops_frauddetection.train_model --pipeline ensemble
    python -m mlops_frauddetection.train_model --no-smote
    python -m mlops_frauddetection.train_model --n-estimators-rf 100
"""

from __future__ import annotations

import argparse
import json
import pdb
import time
from datetime import datetime
from pathlib import Path

from gto import ext
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from seaborn import cm
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
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

from mlops_frauddetection.config import MODELS_DIR, PROCESSED_DATA_DIR
from mlops_frauddetection.logging_config import get_logger, setup_logging
from mlops_frauddetection.utils.seed import set_seed
from mlops_frauddetection.monitoring.system_monitoring import ResourceMonitor

import warnings
import logging

# Suppress MLflow deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress MLflow verbose logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report as skl_report

Mlflow_track = "sqlite:///mlflow.db"
Mlflow_name = "fraud-anomaly-detection-v4-rf300-lgb100"

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
    "merchant_risk_30_day", "merchant_risk_7_day",
    "trans_time_is_night", "trans_date_is_weekend",
    "customer_num_trans_1_day", "amt", "avg_amt_per_customer",
]

bool_cols = [
    "gender_M", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos",
    "category_health_fitness", "category_home", "category_kids_pets",
    "category_misc_net", "category_misc_pos", "category_personal_care",
    "category_shopping_net", "category_shopping_pos", "category_travel",
]

dropping_cols = ["dob", "trans_date", "trans_time"]

# Pipeline B (Israail) constants
num_features = [
    "log_amt", "city_pop",
    "customer_num_trans_1_day", "customer_num_trans_7_day",
    "customer_num_trans_30_day",
    "customer_avg_amout_1_day", "customer_avg_amount_7_day",
    "customer_avg_amount_30_day",
    "merchant_num_trans_1_day", "merchant_num_trans_7_day",
    "merchant_num_trans_30_day",
    "merchant_risk_1_day", "merchant_risk_7_day",
    "merchant_risk_30_day", "merchant_risk_90_day",
    "trans_time_hrs", "trans_hour_sin", "trans_hour_cos",
    "trans_time_is_night", "trans_date_is_weekend", "trans_time_day",
    "category_is_online", "online_x_log_amt", "velocity_ratio",
]


# data loading

def load_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load preprocessed train/test CSVs.

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


# cleaning

def _clean_data_lr(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip spaces, coerce numerics, map booleans, drop date cols.

    Args:
        X_train: Raw training features.
        X_test:  Raw test features.

    Returns:
        Cleaned (X_train, X_test).
    """
    X_train.columns = X_train.columns.str.strip()
    X_test.columns  = X_test.columns.str.strip()

    for col in clean_numeric_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")
        median_val   = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

    X_train = X_train.drop(columns=dropping_cols, errors="ignore")
    X_test  = X_test.drop(columns=dropping_cols,  errors="ignore")

    for col in bool_cols:
        if col in X_train.columns:
            if X_train[col].dtype == object:
                X_train[col] = X_train[col].str.strip().map({"True": 1, "False": 0})
                X_test[col]  = X_test[col].str.strip().map({"True": 1, "False": 0})
            else:
                X_train[col] = X_train[col].fillna(0).astype(int)  
                X_test[col]  = X_test[col].fillna(0).astype(int)

    logger.info("LR cleaning — NaN train: %d  NaN test: %d",
                int(X_train.isna().sum().sum()),
                int(X_test.isna().sum().sum()))
    return X_train, X_test


def _create_4class_labels(X: pd.DataFrame, y: pd.Series) -> pd.Series:
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
    amt         = X["amt"]
    cus_avg_amt = X["avg_amt_per_customer"]
    risk_lvl30  = X["merchant_risk_30_day"]
    is_night    = X["trans_time_is_night"]
    is_fraud    = y.values

    suspicious = (
        (risk_lvl30 >= 21)
        | (amt > cus_avg_amt * 2)
        | ((is_night == 1) & (amt > 200))
    )
    high_risk_merchant = risk_lvl30 >= 17

    labels = pd.Series(0, index=X.index, name="risk_label")
    labels[(is_fraud == 0) & suspicious]          = 1
    labels[(is_fraud == 1) & ~high_risk_merchant] = 2
    labels[(is_fraud == 1) & high_risk_merchant]  = 3
    return labels


def _add_features_lr(X: pd.DataFrame) -> pd.DataFrame:
    """Add 5 ratio-based features for the LR pipeline.

    New columns: amt_ratio, combined_risk, amt_risk_score,
                 is_high_spend, night_high_amt.

    Args:
        X: Cleaned feature matrix.

    Returns:
        Copy of X with 5 additional columns.
    """
    X = X.copy()
    X["amt_ratio"]      = X["amt"] / (X["avg_amt_per_customer"] + 1)
    X["combined_risk"]  = (X["merchant_risk_30_day"] * 0.6
                           + X["merchant_risk_7_day"] * 0.4)
    X["amt_risk_score"] = X["amt_ratio"] * X["merchant_risk_30_day"]
    X["is_high_spend"]  = (X["amt"] > X["avg_amt_per_customer"] * 1.5).astype(int)
    X["night_high_amt"] = (
        (X["trans_time_is_night"] == 1) & (X["amt"] > 100)
    ).astype(int)
    return X


def _run_stratified_cv(
    model: object, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="f1_weighted", n_jobs=-1)
    for i, s in enumerate(scores, 1):
        logger.info("  Fold %d: %.4f", i, s)
    logger.info("  Mean F1: %.4f  Std: %.4f", scores.mean(), scores.std())
    return float(scores.mean()), float(scores.std())

def run_debug_checks(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        enable_breakpoint: bool = True
    ) -> None:

    """Run debug checks on the data and optionally set a breakpoint
    This function is used only when --debug is passed from CLI.
    This function is used to verify that the training and testing data are loaded correctly
    and testing data is valid before model training starts."""

    logger.debug("Debugging mode enabled......")

    assert not X_train.empty, "X_train is empty"
    assert not y_train.empty, "y_train is empty"
    assert not X_test.empty, "X_test is empty"
    assert not y_test.empty, "y_test is empty"

    assert len(X_train) == len(y_train) , (f"Training feature/label mismatch: X_train={len(X_train)}, "
        f"y_train={len(y_train)}"
    )
    assert len(X_test) == len(y_test), (f"Testing feature/label mismatch: X_test={len(X_test)}, "
        f"y_test={len(y_test)}"
    )

    logger.info("DEBUG X_train shape: %s", X_train.shape)
    logger.info("DEBUG X_test shape: %s", X_test.shape)
    logger.info("DEBUG y_train shape: %s", y_train.shape)
    logger.info("DEBUG y_test shape: %s", y_test.shape)

    logger.info("DEBUG y_train class distribution:\n%s", y_train.value_counts())
    logger.info("DEBUG y_test class distribution:\n%s", y_test.value_counts())

    if enable_breakpoint:
        logger.debug("Entering debugger. Use commands like: p X_train.shape, n, c, q")
        pdb.set_trace()


def train_lr_pipeline(
    data_path: Path,
    model_dir: Path,
    max_iter: int = 1000,
    seed: int = 42,
    run_smote: bool = True,
    debug:bool = False,
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
    monitor_path = Path("reports/monitoring/lr_pipeline_resource_usage.csv")
    monitor = ResourceMonitor(monitor_path)
    monitor.start()

    pipeline_start_time = time.time()

    try:
        X_train, y_train, X_test, y_test = load_data(data_path)

        if debug:
            run_debug_checks(X_train, y_train, X_test, y_test)

        X_train, X_test = _clean_data_lr(X_train.copy(), X_test.copy())

        logger.info("Building 4-class labels ...")
        y_train_4 = _create_4class_labels(X_train, y_train)
        y_test_4  = _create_4class_labels(X_test,  y_test)
        for cls, name in lr_label_names_m.items():
            cnt = int((y_train_4 == cls).sum())
            logger.info("  Class %d (%s): %d rows (%.2f%%)",
                        cls, name, cnt, cnt / len(y_train_4) * 100)

        logger.info("Feature engineering ...")
        X_tr = _add_features_lr(X_train)
        X_ts = _add_features_lr(X_test)
        logger.info("Shape: %s -> %s", X_train.shape, X_tr.shape)

        # Model 1
        logger.info("Model 1: LR balanced")
        model_lr = LogisticRegression(max_iter=max_iter, class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
        t0 = time.time()
        m1_cv_mean, m1_cv_std = _run_stratified_cv(model_lr, X_tr, y_train_4)
        logger.info("CV in %.1fs", time.time() - t0)

        model_lr.fit(X_tr, y_train_4)
        y_pred_m1    = model_lr.predict(X_ts)
        m1_train_acc = accuracy_score(y_train_4, model_lr.predict(X_tr))
        m1_test_acc  = accuracy_score(y_test_4,  y_pred_m1)
        m1_test_f1   = f1_score(y_test_4, y_pred_m1, average="weighted")
        logger.info("Model 1 — Train: %.4f  Test acc: %.4f  F1: %.4f",
                    m1_train_acc, m1_test_acc, m1_test_f1)
        logger.info("\n%s", classification_report(
            y_test_4, y_pred_m1, target_names=list(lr_label_names_m.values())))

        m1_path = model_dir / f"lr_balanced_{timestamp}.joblib"
        joblib.dump(model_lr, m1_path)

        with mlflow.start_run(run_name=f"LR_balanced_maxiter{max_iter}_{timestamp}"):
            mlflow.log_params({"model": "LR_balanced", "class_weight": "balanced",
                            "max_iter": max_iter, "seed": seed,
                            "smote": False, "cv_folds": 5, "pipeline": "A"})
            mlflow.log_metrics({"cv_mean_f1": m1_cv_mean, "cv_std_f1": m1_cv_std,
                                "train_acc": m1_train_acc, "test_acc": m1_test_acc,
                                "test_f1": m1_test_f1})
            mlflow.sklearn.log_model(model_lr, "lr_balanced")
            mlflow.log_artifact(str(m1_path))
            mlflow.log_artifact(str(monitor_path))

            cm = confusion_matrix(y_test_4, y_pred_m1)
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(cm, display_labels=list(lr_label_names_m.values()))
            disp.plot(ax=ax, colorbar=True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
            for text in ax.texts:
                try:
                    val = float(text.get_text())
                    if val >= 1000:
                        text.set_text(f"{int(val):,}")
                    text.set_fontsize(9)
                except ValueError:
                    pass
            plt.tight_layout()
            plt.savefig("reports/figures/mlflow_cm_lr_balanced.png", dpi=150, bbox_inches='tight')
            mlflow.log_artifact("reports/figures/mlflow_cm_lr_balanced.png")
            plt.close(fig)
        logger.info("Model 1 saved and logged -> %s", m1_path)

        # Model 2 — SMOTE
        m2_cv_mean = m2_cv_std = m2_train_acc = m2_test_acc = m2_test_f1 = None
        m2_path = None

        if run_smote:
            logger.info("Model 2: SMOTE + LR")
            smote = SMOTE(random_state=seed, k_neighbors=5)
            t0 = time.time()
            X_smote, y_smote = smote.fit_resample(X_tr, y_train_4)
            logger.info("SMOTE in %.1fs — size: %s", time.time() - t0, X_smote.shape)

            model_smote = LogisticRegression(max_iter=max_iter,
                                            class_weight="balanced",
                                            random_state=seed, n_jobs=-1)
            t0 = time.time()
            m2_cv_mean, m2_cv_std = _run_stratified_cv(model_smote, X_smote, y_smote)
            logger.info("CV in %.1fs", time.time() - t0)

            model_smote.fit(X_smote, y_smote)
            y_pred_m2    = model_smote.predict(X_ts)
            m2_train_acc = accuracy_score(y_smote, model_smote.predict(X_smote))
            m2_test_acc  = accuracy_score(y_test_4, y_pred_m2)
            m2_test_f1   = f1_score(y_test_4, y_pred_m2, average="weighted")
            logger.info("Model 2 — Train: %.4f  Test acc: %.4f  F1: %.4f",
                        m2_train_acc, m2_test_acc, m2_test_f1)
            logger.info("\n%s", classification_report(
                y_test_4, y_pred_m2, target_names=list(lr_label_names_m.values())))

            m2_path = model_dir / f"lr_smote_{timestamp}.joblib"
            joblib.dump(model_smote, m2_path)

            with mlflow.start_run(run_name=f"LR_SMOTE_maxiter{max_iter}_{timestamp}"):
                mlflow.log_params({"model": "LR_SMOTE", "class_weight": "balanced",
                                "max_iter": max_iter, "seed": seed,
                                "smote": True, "smote_strategy": "auto",
                                "cv_folds": 5, "pipeline": "A"})
                mlflow.log_metrics({"cv_mean_f1": m2_cv_mean, "cv_std_f1": m2_cv_std,
                                    "train_acc": m2_train_acc, "test_acc": m2_test_acc,
                                    "test_f1": m2_test_f1})
                mlflow.sklearn.log_model(model_smote, "lr_smote")
                mlflow.log_artifact(str(m2_path))
                mlflow.log_artifact(str(monitor_path))

                cm2 = confusion_matrix(y_test_4, y_pred_m2)
                fig2, ax2 = plt.subplots(figsize=(10, 8))  # bigger figure
                disp = ConfusionMatrixDisplay(cm2, display_labels=list(lr_label_names_m.values()))
                disp.plot(ax=ax2, colorbar=True)
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=9)
                ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)
                # Format large numbers in cells
                for text in ax2.texts:
                    try:
                        val = float(text.get_text())
                        if val >= 1000:
                            text.set_text(f"{int(val):,}")
                        text.set_fontsize(9)
                    except ValueError:
                        pass
                plt.tight_layout()
                plt.savefig("reports/figures/mlflow_cm_lr_smote.png", dpi=150, bbox_inches='tight')
                mlflow.log_artifact("reports/figures/mlflow_cm_lr_smote.png")
                plt.close(fig2)
            logger.info("Model 2 saved and logged -> %s", m2_path)

        # Metadata
        meta: dict = {
            "dataset": "Credit Card Fraud Detection - Kaggle",
            "dataset_version": "DVC tracked - data/processed/",
            "training_date": timestamp,
            "train_samples": int(X_tr.shape[0]),
            "test_samples":  int(X_ts.shape[0]),
            "features":      int(X_tr.shape[1]),
            "classes":       lr_label_names_m,
            "reproducibility": {"random_seed": seed,
                                "cv_strategy": "StratifiedKFold(n_splits=5)"},
            "model_1": {"name": "LR_balanced", "max_iter": max_iter,
                        "cv_f1_mean": round(m1_cv_mean, 4),
                        "cv_f1_std":  round(m1_cv_std,  4),
                        "train_acc":  round(m1_train_acc, 4),
                        "test_acc":   round(m1_test_acc,  4),
                        "test_f1":    round(m1_test_f1,   4),
                        "file":       str(m1_path)},
        }
        if run_smote and m2_path:
            meta["model_2"] = {"name": "LR_SMOTE", "max_iter": max_iter,
                            "cv_f1_mean": round(m2_cv_mean,   4),
                            "cv_f1_std":  round(m2_cv_std,    4),
                            "train_acc":  round(m2_train_acc, 4),
                            "test_acc":   round(m2_test_acc,  4),
                            "test_f1":    round(m2_test_f1,   4),
                            "file":       str(m2_path)}

        with open(model_dir / "LR_SMOTE_model_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
    finally:
        monitor.stop()
        logger.info("Pipeline A resource monitoring saved to %s", monitor_path)
        logger.info("Pipeline A total time: %.1fs", time.time() - pipeline_start_time)
    logger.info("Pipeline A complete")


#feature enginerring

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

    online_cols = ["category_shopping_net", "category_misc_net",
                   "category_grocery_net"]
    for col in online_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip().map({"True": 1, "False": 0})

    df["category_is_online"] = df[online_cols].any(axis=1).astype(int)
    df["online_x_log_amt"]   = df["category_is_online"] * df["log_amt"]
    df["velocity_ratio"]     = (df["customer_num_trans_1_day"]
                                 / (df["customer_num_trans_30_day"] + 1))
    df["trans_hour_sin"] = np.sin(2 * np.pi * df["trans_time_hrs"] / 24)
    df["trans_hour_cos"] = np.cos(2 * np.pi * df["trans_time_hrs"] / 24)
    return df


def _build_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build ColumnTransformer (StandardScaler for numeric, passthrough for cat).

    Args:
        X_train: Training features after feature engineering.

    Returns:
        (unfitted_preprocessor, numeric_features, categorical_features).
    """
    numeric_features = [f for f in num_features
                        if f in X_train.columns]
    categorical_features = [
        col for col in X_train.columns
        if (col.startswith("category_") or col.startswith("gender_"))
        and col not in numeric_features
    ]
    numeric_features     = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))
    categorical_features = [c for c in categorical_features
                            if c not in numeric_features]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
        ("cat", "passthrough", categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


def train_ensemble_pipeline(
    data_path: Path,
    model_dir: Path,
    seed: int = 42,
    n_estimators_rf: int = 200,
    n_estimators_lgb: int = 500,
    n_estimators_xgb: int = 500,
    debug: bool = False,
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

    monitor_path = Path("reports/monitoring/ensemble_pipeline_resource_usage.csv")
    monitor = ResourceMonitor(monitor_path)
    monitor.start()

    pipeline_start_time = time.time()

    try:

        X_train, y_train, X_test, y_test = load_data(data_path)
        if debug:
            run_debug_checks(X_train, y_train, X_test, y_test)

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train["is_fraud"]
        if isinstance(y_test, pd.DataFrame):
            y_test  = y_test["is_fraud"]

        logger.info("Feature engineering ...")
        X_train = _engineer_features_ensemble(X_train)
        X_test  = _engineer_features_ensemble(X_test)

        preprocessor, num_feats, cat_feats = _build_preprocessor(X_train)
        all_feats = num_feats + cat_feats
        X_train = X_train[all_feats].copy()
        X_test  = X_test[all_feats].copy()

        logger.info("Preprocessing %d features ...", len(all_feats))
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep  = preprocessor.transform(X_test)

        logger.info("Applying SMOTE (sampling_strategy=0.3) ...")
        logger.info("Before: %s", dict(pd.Series(y_train).value_counts().sort_index()))
        t0 = time.time()
        smote = SMOTE(random_state=seed, sampling_strategy=0.3)
        X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)
        logger.info("SMOTE in %.1fs  After: %s",
                    time.time() - t0,
                    dict(pd.Series(y_train_res).value_counts().sort_index()))

        scale_pos = int((y_train_res == 0).sum() / (y_train_res == 1).sum())

        ensemble_models = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed),
            "RandomForest": RandomForestClassifier(
                n_estimators=n_estimators_rf, class_weight="balanced",
                random_state=seed, n_jobs=-1),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=n_estimators_lgb, class_weight="balanced",
                learning_rate=0.05, num_leaves=63,
                random_state=seed, verbose=-1),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=n_estimators_xgb, scale_pos_weight=scale_pos,
                learning_rate=0.05, max_depth=6,
                random_state=seed, eval_metric="logloss", verbosity=0),
        }

        tscv = TimeSeriesSplit(n_splits=5)
        all_results: dict = {}

        for name, model in ensemble_models.items():
            logger.info("Training %s ...", name)
            t0 = time.time()
            model.fit(X_train_res, y_train_res)
            logger.info("  Done in %.1fs", time.time() - t0)

            y_pred      = model.predict(X_test_prep)
            y_pred_prob = model.predict_proba(X_test_prep)[:, 1]

            train_acc = accuracy_score(y_train_res, model.predict(X_train_res))
            test_acc  = accuracy_score(y_test, y_pred)
            test_f1   = f1_score(y_test, y_pred)
            prec      = precision_score(y_test, y_pred)
            rec       = recall_score(y_test, y_pred)
            roc_auc   = roc_auc_score(y_test, y_pred_prob)
            avg_prec  = average_precision_score(y_test, y_pred_prob)

            logger.info("  %s — Train: %.4f  F1: %.4f  ROC: %.4f  AP: %.4f",
                        name, train_acc, test_f1, roc_auc, avg_prec)

            cv_scores = cross_val_score(model, X_train_res, y_train_res,
                                        cv=tscv, scoring="f1", n_jobs=-1)
            logger.info("  CV F1 mean=%.4f  std=%.4f",
                        cv_scores.mean(), cv_scores.std())

            all_results[name] = {
                "train_acc": train_acc, "test_acc": test_acc,
                "f1": test_f1, "precision": prec, "recall": rec,
                "roc_auc": roc_auc, "avg_prec": avg_prec,
                "cv_mean_f1": float(cv_scores.mean()),
                "cv_std_f1":  float(cv_scores.std()),
            }

            fname = name.lower() + f"_{timestamp}.joblib"
            mpath = model_dir / fname
            joblib.dump(model, mpath)

            with mlflow.start_run(run_name=f"{name}_rf{n_estimators_rf}_{timestamp}"):
                mlflow.log_params({"model": name, "seed": seed,
                                "smote_strategy": 0.3, "cv_folds": 5,
                                "pipeline": "B"})
                mlflow.log_metrics({"train_acc": train_acc, "test_acc": test_acc,
                                    "f1": test_f1, "precision": prec,
                                    "recall": rec, "roc_auc": roc_auc,
                                    "avg_prec": avg_prec,
                                    "cv_mean_f1": float(cv_scores.mean()),
                                    "cv_std_f1":  float(cv_scores.std())})
                mlflow.sklearn.log_model(model, name.lower())
                mlflow.log_artifact(str(mpath))
                mlflow.log_artifact(str(monitor_path))
                
                report = skl_report(y_test, y_pred)
                report_path = f"reports/figures/{name.lower()}_classification_report.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)
            logger.info("  %s saved and logged -> %s", name, mpath)

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
                "train_samples": int(X_train_res.shape[0]),
                "test_samples":  int(X_test_prep.shape[0]),
                "features":      int(X_train_res.shape[1]),
                "smote_strategy": 0.3,
                "seed": seed,
                **res,
            }
            with open(meta_path, "w") as fh:
                json.dump(existing, fh, indent=2)

        logger.info("Pipeline B results:")
        logger.info("  %-22s  F1      ROC-AUC  AvgPrec", "Model")
        for name, res in all_results.items():
            logger.info("  %-22s  %.4f  %.4f   %.4f",
                        name, res["f1"], res["roc_auc"], res["avg_prec"])
    finally:
        monitor.stop()
        logger.info("Pipeline B resource monitoring saved to %s", monitor_path)
        logger.info("Pipeline B total time: %.1fs", time.time() - pipeline_start_time)
    logger.info("Pipeline B complete")


#CLI

def main() -> None:
    """CLI entrypoint called by ``make train``."""
    parser = argparse.ArgumentParser(
        description="Train all fraud-detection models (LR + ensemble)"
    )
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument(
        "--pipeline",
        choices=["all", "lr", "ensemble"],
        default="all",
        help="Which pipeline to run (default: all)",
    )
    parser.add_argument("--max-iter",         type=int, default=1000)
    parser.add_argument("--smote",            action="store_true", default=True)
    parser.add_argument("--no-smote",         dest="smote", action="store_false")
    parser.add_argument("--n-estimators-rf",  type=int, default=200)
    parser.add_argument("--n-estimators-lgb", type=int, default=500)
    parser.add_argument("--n-estimators-xgb", type=int, default=500)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    if args.pipeline in ("all", "lr"):
        train_lr_pipeline(
            data_path=args.data_path,
            model_dir=args.model_dir,
            max_iter=args.max_iter,
            seed=args.seed,
            run_smote=args.smote,
            debug=args.debug,
        )

    if args.pipeline in ("all", "ensemble"):
        train_ensemble_pipeline(
            data_path=args.data_path,
            model_dir=args.model_dir,
            seed=args.seed,
            n_estimators_rf=args.n_estimators_rf,
            n_estimators_lgb=args.n_estimators_lgb,
            n_estimators_xgb=args.n_estimators_xgb,
            debug=args.debug,
        )

    logger.info("All training complete")


if __name__ == "__main__":
    main()
