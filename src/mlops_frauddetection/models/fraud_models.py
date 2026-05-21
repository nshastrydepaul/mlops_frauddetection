"""Model for fraud detection.

Returns unfitted sklearn-compatible estimators for both pipelines:

  Pipeline A (Musaddiq) — LR variants
    get_lr_balanced()  — LogisticRegression with class_weight='balanced'
    get_lr_smote()     — Same LR used after SMOTE oversampling

  Pipeline B (Israail) — Ensemble models
    get_rf_model()     — RandomForestClassifier
    get_lgb_model()    — LightGBM LGBMClassifier
    get_xgb_model()    — XGBoost XGBClassifier
    build_preprocessor() — re-exported from build_features for convenience
"""

from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)


#Pipeline A

def get_lr_balanced(
    max_iter: int = 1000,
    seed: int = 42,
) -> LogisticRegression:
    """Return unfitted LR with class_weight='balanced' (Model 1).

    Using balanced weights so the model gives more attention to the
    minority fraud classes (FT/FF at ~0.5% each) during training.

    Args:
        max_iter: Maximum solver iterations (default 1000 for convergence
                  on large data with engineered features).
        seed:     Random state for reproducibility.

    Returns:
        Unfitted LogisticRegression instance.
    """
    logger.info("Creating LR balanced model (max_iter=%d, seed=%d)",
                max_iter, seed)
    return LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def get_lr_smote(
    max_iter: int = 1000,
    seed: int = 42,
) -> LogisticRegression:
    """Return unfitted LR for use after SMOTE oversampling (Model 2).

    Identical hyperparameters to get_lr_balanced — the difference is
    that this model trains on SMOTE-balanced data rather than the
    original imbalanced dataset.

    Args:
        max_iter: Maximum solver iterations.
        seed:     Random state for reproducibility.

    Returns:
        Unfitted LogisticRegression instance.
    """
    logger.info("Creating LR SMOTE model (max_iter=%d, seed=%d)",
                max_iter, seed)
    return LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


#Pipeline B

def get_rf_model(
    n_estimators: int = 200,
    seed: int = 42,
) -> RandomForestClassifier:
    """Return unfitted Random Forest classifier.

    Args:
        n_estimators: Number of trees (default 200 balances speed/accuracy).
        seed:         Random state for reproducibility.

    Returns:
        Unfitted RandomForestClassifier instance.
    """
    logger.info("Creating RF model (n_estimators=%d, seed=%d)",
                n_estimators, seed)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def get_lgb_model(
    n_estimators: int = 500,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    """Return unfitted LightGBM classifier.

    Args:
        n_estimators: Number of boosting rounds (default 500).
        seed:         Random state for reproducibility.

    Returns:
        Unfitted LGBMClassifier instance.
    """
    logger.info("Creating LightGBM model (n_estimators=%d, seed=%d)",
                n_estimators, seed)
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        learning_rate=0.05,
        num_leaves=63,
        random_state=seed,
        verbose=-1,
    )


def get_xgb_model(
    n_estimators: int = 500,
    scale_pos_weight: int = 1,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Return unfitted XGBoost classifier.

    Args:
        n_estimators:     Number of boosting rounds (default 500).
        scale_pos_weight: Ratio of negative to positive samples — pass
                          int((y==0).sum() / (y==1).sum()) after SMOTE.
        seed:             Random state for reproducibility.

    Returns:
        Unfitted XGBClassifier instance.
    """
    logger.info("Creating XGBoost model (n_estimators=%d, scale_pos=%d, seed=%d)",
                n_estimators, scale_pos_weight, seed)
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
