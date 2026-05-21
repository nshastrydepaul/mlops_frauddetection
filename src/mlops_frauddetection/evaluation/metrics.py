"""Evaluation metric helpers for fraud detection pipelines.

Covers two evaluation scenarios:

  evaluate_multiclass() — Pipeline A: 4-class TT/TF/FT/FF labels
  evaluate_binary()     — Pipeline B: binary fraud (0/1) labels
  overfitting_check()   — Compare train vs test accuracy difference

Also provides backward-compatible wrappers for the existing
classification_report() and regression_report() functions used
elsewhere in the codebase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)

# 4-class label names for reporting
LABEL_NAMES: dict[int, str] = {
    0: "TT-Legit-Normal",
    1: "TF-Legit-Suspicious",
    2: "FT-Fraud-LowRisk",
    3: "FF-Fraud-HighRisk",
}


def evaluate_multiclass(
    y_true: pd.Series,
    y_pred: Any,
    y_train_true: pd.Series | None = None,
    y_train_pred: Any | None = None,
    model_name: str = "Model",
) -> dict[str, float]:
    """Evaluate a 4-class fraud risk classifier.

    Computes weighted F1, accuracy, and optionally train/test gap
    for overfitting detection.

    Args:
        y_true:        True test labels (0-3).
        y_pred:        Predicted test labels (0-3).
        y_train_true:  True train labels (optional, for overfitting check).
        y_train_pred:  Predicted train labels (optional).
        model_name:    Name for logging output.

    Returns:
        Dict with keys: test_acc, test_f1, train_acc (if provided),
        gap (if provided).
    """
    test_acc = float(sk_metrics.accuracy_score(y_true, y_pred))
    test_f1  = float(sk_metrics.f1_score(y_true, y_pred, average="weighted"))

    logger.info("%s — Test acc: %.4f  Test F1: %.4f", model_name, test_acc, test_f1)
    logger.info("\n%s", sk_metrics.classification_report(
        y_true, y_pred, target_names=list(LABEL_NAMES.values())
    ))

    result: dict[str, float] = {"test_acc": test_acc, "test_f1": test_f1}

    if y_train_true is not None and y_train_pred is not None:
        train_acc = float(sk_metrics.accuracy_score(y_train_true, y_train_pred))
        gap       = abs(train_acc - test_acc) * 100
        result["train_acc"] = train_acc
        result["gap"]       = gap
        status = "No overfitting" if gap < 5 else "Possible overfitting"
        logger.info(
            "%s — Train: %.2f%%  Test: %.2f%%  Gap: %.1f%%  %s",
            model_name, train_acc * 100, test_acc * 100, gap, status,
        )

    return result


def evaluate_binary(
    y_true: pd.Series,
    y_pred: Any,
    y_pred_prob: Any,
    y_train_true: pd.Series | None = None,
    y_train_pred: Any | None = None,
    model_name: str = "Model",
) -> dict[str, float]:
    """Evaluate a binary fraud classifier with probability outputs.

    Args:
        y_true:       True test labels (0/1).
        y_pred:       Predicted test labels (0/1).
        y_pred_prob:  Predicted probabilities for class 1.
        y_train_true: True train labels (optional, for overfitting check).
        y_train_pred: Predicted train labels (optional).
        model_name:   Name for logging output.

    Returns:
        Dict with keys: test_acc, f1, precision, recall,
        roc_auc, avg_prec, and optionally train_acc, gap.
    """
    test_acc = float(sk_metrics.accuracy_score(y_true, y_pred))
    test_f1  = float(sk_metrics.f1_score(y_true, y_pred))
    prec     = float(sk_metrics.precision_score(y_true, y_pred))
    rec      = float(sk_metrics.recall_score(y_true, y_pred))
    roc_auc  = float(sk_metrics.roc_auc_score(y_true, y_pred_prob))
    avg_prec = float(sk_metrics.average_precision_score(y_true, y_pred_prob))

    logger.info(
        "%s — Acc: %.4f  F1: %.4f  ROC-AUC: %.4f  AvgPrec: %.4f",
        model_name, test_acc, test_f1, roc_auc, avg_prec,
    )

    result: dict[str, float] = {
        "test_acc":  test_acc,
        "f1":        test_f1,
        "precision": prec,
        "recall":    rec,
        "roc_auc":   roc_auc,
        "avg_prec":  avg_prec,
    }

    if y_train_true is not None and y_train_pred is not None:
        train_acc = float(sk_metrics.accuracy_score(y_train_true, y_train_pred))
        gap       = abs(train_acc - test_acc) * 100
        result["train_acc"] = train_acc
        result["gap"]       = gap

    return result


def overfitting_check(
    train_acc: float,
    test_acc: float,
    model_name: str = "Model",
    threshold: float = 5.0,
) -> str:
    """Check if a model is overfitting based on train/test accuracy gap.

    Args:
        train_acc:  Training accuracy (0.0-1.0).
        test_acc:   Test accuracy (0.0-1.0).
        model_name: Name for logging.
        threshold:  Gap percentage above which overfitting is flagged.

    Returns:
        Status string: 'No overfitting' or 'Possible overfitting'.
    """
    gap    = abs(train_acc - test_acc) * 100
    status = "No overfitting" if gap < threshold else "Possible overfitting"
    logger.info(
        "%s — Train: %.1f%%  Test: %.1f%%  Gap: %.1f%%  %s",
        model_name, train_acc * 100, test_acc * 100, gap, status,
    )
    return status


# wrapping

def classification_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 as a plain dict.

    Backward-compatible wrapper used by existing code in the codebase.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """
    return {
        "accuracy":  float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(sk_metrics.precision_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(sk_metrics.recall_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "f1":        float(sk_metrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and R^2 as a plain dict.

    Backward-compatible wrapper used by existing code in the codebase.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dict with keys: mae, mse, rmse, r2.
    """
    mse = float(sk_metrics.mean_squared_error(y_true, y_pred))
    return {
        "mae":  float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "r2":   float(sk_metrics.r2_score(y_true, y_pred)),
    }
