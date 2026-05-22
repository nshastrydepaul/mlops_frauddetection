"""
Visualization utilities for Fraud-Anomaly Detection & Behavioral Analytics.

This module contains functions for creating plots, charts, and other visualizations
extracted from the exploratory and model notebooks.

Developer Credits:
    RGhazzal    — EDA visualizations (target variable, amount, categorical, temporal,
                  bivariate, correlation, merchant risk, pair plots)
                  ML model visualizations (ROC, PR curves, confusion matrices,
                  feature importance, class balance)
    Lohith      — Data exploration visualizations (class distribution, amount
                  distribution, fraud by category, avg spend, transaction windows,
                  fraud by time, heatmap DOW vs hour, transaction count, fraud by
                  state, location scatter, age distribution, correlation heatmap)
    Musaddiq    — Pipeline A label distribution (4-class TT/TF/FT/FF)
    Nishanthshastry — Data sampling (data_100k.csv)

Plot functions:
    plot_training_history()         — Training loss/metric curves
    plot_confusion_matrix()         — Single confusion matrix (matplotlib)
    plot_class_distribution()       — Fraud vs legitimate bar + pie [RGhazzal/Lohith]
    plot_amount_distribution()      — Amount histogram + log + boxplot [RGhazzal/Lohith]
    plot_fraud_by_category()        — Fraud rate per category [RGhazzal/Lohith]
    plot_fraud_by_time()            — Fraud rate by hour and DOW [RGhazzal/Lohith]
    plot_feature_importance()       — Top-N feature importances [RGhazzal]
    plot_roc_curves()               — ROC curves for multiple models [RGhazzal]
    plot_precision_recall_curves()  — PR curves for multiple models [RGhazzal]
    plot_confusion_matrices()       — Side-by-side confusion matrices [RGhazzal]
    plot_correlation_heatmap()      — Feature correlation heatmap [RGhazzal/Lohith]
    plot_avg_spend_per_category()   — Median spend per category [Lohith]
    plot_customer_avg_spend_window()— Customer avg spend windows [Lohith]
    plot_customer_location_scatter()— Customer lat/long scatter [Lohith]
    plot_customer_transaction_count()— Customer txn count distribution [Lohith]
    plot_age_distribution_by_class()— Age distribution by fraud label [Lohith]
    plot_fraud_risk_by_age()        — Fraud risk across age groups [Lohith]
    plot_heatmap_dow_vs_hour()      — Fraud rate heatmap DOW vs hour [Lohith]
    plot_label_distribution()       — 4-class label distribution [Musaddiq]
    plot_fraud_by_state()           — Total fraud volume by state [Lohith]
    plot_transactions_by_hour()     — Transaction count by hour [Lohith]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlops_frauddetection.data.exceptions import VisualizationError
from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)

# Project colour palette for consistent branding across all visualizations
FRAUD_COLOR = "#a2191f"  # red   — fraud
LEGIT_COLOR = "#0f62fe"  # blue  — legitimate
NEUTRAL_COLOR = "#6f6f6f"  # grey  — neutral
AMBER_COLOR = "#f1c21b"  # amber — warning


def _ensure_figures_dir(output_path: str) -> Path:
    """Validate and create the output directory if it does not exist.

    Args:
        output_path: Path string for the output file.

    Returns:
        Resolved Path object.

    Raises:
        VisualizationError: If the directory cannot be created.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory confirmed: %s", path.parent)
        return path
    except OSError as e:
        logger.error("Failed to create output directory %s: %s", path.parent, e)
        raise VisualizationError(
            f"Cannot create output directory: {path.parent}"
        ) from e


def _save_or_show(fig: Any, output_path: str | None, label: str) -> None:
    """Save figure to disk or display it, then close.

    Args:
        fig:         Matplotlib figure.
        output_path: Path to save the figure, or None to display.
        label:       Human-readable plot name for logging.

    Raises:
        VisualizationError: If the figure cannot be saved.
    """
    try:
        if output_path:
            path = _ensure_figures_dir(output_path)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            logger.info("[COMPLETE] %s saved -> %s", label, path)
        else:
            plt.show()
            logger.info("[COMPLETE] %s displayed (no output path provided)", label)
    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Failed to save %s to %s: %s", label, output_path, e)
        raise VisualizationError(f"Failed to save {label}: {output_path}") from e
    finally:
        plt.close(fig)


def plot_training_history(
    history: dict[str, list[float]], output_path: str | None = None
) -> None:
    """Plot training history curves.

    Args:
        history:     Dictionary containing training metrics keyed by name.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting training history ...")

    try:
        if not history:
            logger.warning("Training history is empty — skipping plot")
            return

        epochs = max(len(v) for v in history.values())
        logger.info("  Metrics: %s | Epochs: %d", list(history.keys()), epochs)

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric, values in history.items():
            ax.plot(values, label=metric)
            logger.info("  %s — final value: %.4f", metric, values[-1])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        _save_or_show(fig, output_path, "Training history")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_training_history: %s", e)
        raise VisualizationError("Failed to plot training history") from e


def plot_confusion_matrix(
    cm: Any, labels: list[str] | None = None, output_path: str | None = None
) -> None:
    """Plot a single confusion matrix.

    Credit: RGhazzal (1_0-rghazzal-machine-learning-models.ipynb, Cell 40)

    Args:
        cm:          Confusion matrix array.
        labels:      Optional class labels.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting confusion matrix [RGhazzal] ...")

    try:
        cm_array = np.array(cm)
        logger.info("  Shape: %s | Classes: %d", cm_array.shape, cm_array.shape[0])
        if labels:
            logger.info("  Class labels: %s", labels)

        for i, correct in enumerate(cm_array.diagonal()):
            label = labels[i] if labels else str(i)
            total = int(cm_array[i].sum())
            logger.info(
                "  Class %s — correct: %d / %d (%.1f%%)",
                label,
                correct,
                total,
                correct / total * 100 if total > 0 else 0,
            )

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)

        _save_or_show(fig, output_path, "Confusion matrix")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_confusion_matrix: %s", e)
        raise VisualizationError("Failed to plot confusion matrix") from e


def plot_class_distribution(y: pd.Series, output_path: str | None = None) -> None:
    """Plot fraud vs legitimate class distribution as bar and pie charts.

    Credit: RGhazzal (1_0-rghazzal-exploratory-data-analysis.ipynb, Cell 9)
            Lohith   (1_0-lohith-data-exploration.ipynb, Cell 8)

    Args:
        y:           Binary label series (0 = legitimate, 1 = fraud).
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting class distribution [RGhazzal, Lohith] ...")

    try:
        counts = y.value_counts().sort_index()
        labels = ["Legitimate", "Fraud"]
        colors = [LEGIT_COLOR, FRAUD_COLOR]
        logger.info(
            "  Legitimate: %d | Fraud: %d (%.2f%%)",
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(1, 0) / len(y) * 100,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(labels, counts.values, color=colors)
        axes[0].set_title("Transaction count by class")
        axes[0].set_ylabel("Count")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 100, f"{v:,}", ha="center")

        axes[1].pie(
            counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.2f%%",
            startangle=90,
        )
        axes[1].set_title("Class proportion")

        fig.suptitle("Class Distribution", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Class distribution")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_class_distribution: %s", e)
        raise VisualizationError("Failed to plot class distribution") from e


def plot_amount_distribution(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot transaction amount distribution — raw, log-transformed, and boxplot.

    Credit: RGhazzal (1_0-rghazzal-exploratory-data-analysis.ipynb, Cell 11)
            Lohith   (1_0-lohith-data-exploration.ipynb, Cell 10)

    Args:
        df:          DataFrame containing 'amt' and 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting amount distribution [RGhazzal, Lohith] ...")

    try:
        if "amt" not in df.columns:
            raise VisualizationError("Column 'amt' not found in DataFrame")

        logger.info(
            "  Rows: %d | amt mean: %.2f | max: %.2f",
            len(df),
            df["amt"].mean(),
            df["amt"].max(),
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(df["amt"], bins=50, color=NEUTRAL_COLOR, edgecolor="white")
        axes[0].set_title("Amount — raw distribution")
        axes[0].set_xlabel("Amount ($)")

        axes[1].hist(np.log1p(df["amt"]), bins=50, color=LEGIT_COLOR, edgecolor="white")
        axes[1].set_title("Amount — log(1+x) transformed")
        axes[1].set_xlabel("log(1 + Amount)")

        fraud_amt = df.loc[df["is_fraud"] == 1, "amt"]
        legit_amt = df.loc[df["is_fraud"] == 0, "amt"]
        axes[2].boxplot(
            [legit_amt, fraud_amt],
            labels=["Legitimate", "Fraud"],
            patch_artist=True,
            boxprops=dict(facecolor=LEGIT_COLOR, alpha=0.6),
        )
        axes[2].set_title("Amount — box plot by class")
        axes[2].set_ylabel("Amount ($)")

        logger.info(
            "  Fraud amt mean: %.2f | Legit amt mean: %.2f",
            fraud_amt.mean(),
            legit_amt.mean(),
        )
        fig.suptitle("Transaction Amount Distribution", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Amount distribution")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_amount_distribution: %s", e)
        raise VisualizationError("Failed to plot amount distribution") from e


def plot_fraud_by_category(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot fraud rate and count per merchant category.

    Credit: RGhazzal (1_0-rghazzal-exploratory-data-analysis.ipynb, Cell 24)
            Lohith   (1_0-lohith-data-exploration.ipynb, Cell 12)

    Args:
        df:          DataFrame with 'category' and 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting fraud by category [RGhazzal, Lohith] ...")

    try:
        if "category" not in df.columns:
            raise VisualizationError("Column 'category' not found in DataFrame")

        logger.info("  Categories: %d", df["category"].nunique())

        fraud_by_cat = (
            df.groupby("category")["is_fraud"]
            .agg(fraud_count="sum", total="count")
            .assign(fraud_rate=lambda x: x["fraud_count"] / x["total"] * 100)
            .sort_values("fraud_rate", ascending=False)
            .reset_index()
        )
        logger.info(
            "  Highest fraud rate: %s (%.2f%%)",
            fraud_by_cat.iloc[0]["category"],
            fraud_by_cat.iloc[0]["fraud_rate"],
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(
            fraud_by_cat["category"],
            fraud_by_cat["fraud_rate"],
            color=FRAUD_COLOR,
            alpha=0.8,
        )
        axes[0].set_title("Fraud rate by category (%)")
        axes[0].set_xlabel("Fraud rate (%)")

        axes[1].barh(
            fraud_by_cat["category"],
            fraud_by_cat["total"],
            color=LEGIT_COLOR,
            alpha=0.8,
        )
        axes[1].set_title("Total transactions by category")
        axes[1].set_xlabel("Transaction count")

        fig.suptitle("Fraud by Merchant Category", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Fraud by category")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_fraud_by_category: %s", e)
        raise VisualizationError("Failed to plot fraud by category") from e


def plot_fraud_by_time(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot fraud rate by hour of day and day of week.

    Credit: RGhazzal (1_0-rghazzal-exploratory-data-analysis.ipynb, Cell 26)
            Lohith   (1_0-lohith-data-exploration.ipynb, Cell 17)

    Args:
        df:          DataFrame with 'trans_time_hrs', 'trans_time_day',
                     and 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting fraud by time [RGhazzal, Lohith] ...")

    try:
        required = ["trans_time_hrs", "trans_time_day", "is_fraud"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise VisualizationError(
                f"Missing required columns for fraud by time: {missing}"
            )

        logger.info("  Rows: %d", len(df))
        hour_rate = (
            df.groupby("trans_time_hrs")["is_fraud"].mean().mul(100).sort_index()
        )
        dow_rate = df.groupby("trans_time_day")["is_fraud"].mean().mul(100).sort_index()
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        logger.info(
            "  Peak fraud hour: %d:00 (%.2f%%)",
            int(hour_rate.idxmax()),
            float(hour_rate.max()),
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(hour_rate.index, hour_rate.values, color=FRAUD_COLOR, alpha=0.8)
        axes[0].set_title("Fraud rate by hour of day")
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Fraud rate (%)")

        axes[1].bar(
            range(len(dow_rate)), dow_rate.values, color=NEUTRAL_COLOR, alpha=0.8
        )
        axes[1].set_xticks(range(len(dow_labels)))
        axes[1].set_xticklabels(dow_labels)
        axes[1].set_title("Fraud rate by day of week")
        axes[1].set_ylabel("Fraud rate (%)")

        fig.suptitle("Fraud by Time", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Fraud by time")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_fraud_by_time: %s", e)
        raise VisualizationError("Failed to plot fraud by time") from e


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    top_n: int = 15,
    model_name: str = "Model",
    output_path: str | None = None,
) -> None:
    """Plot top-N feature importances for a tree-based model.

    Credit: RGhazzal (1_0-rghazzal-machine-learning-models.ipynb, Cell 26)

    Args:
        feature_names: List of feature names.
        importances:   Corresponding importance scores.
        top_n:         Number of top features to display.
        model_name:    Model name for the plot title.
        output_path:   Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info(
        "[START] Plotting feature importance [RGhazzal] — model: %s ...", model_name
    )

    try:
        if len(feature_names) != len(importances):
            raise VisualizationError(
                f"feature_names ({len(feature_names)}) and importances "
                f"({len(importances)}) must have the same length"
            )

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        logger.info(
            "  Top %d of %d features | Top feature: %s (%.4f)",
            top_n,
            len(feature_names),
            importance_df.iloc[0]["feature"],
            importance_df.iloc[0]["importance"],
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color=LEGIT_COLOR,
            alpha=0.85,
        )
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        _save_or_show(fig, output_path, f"Feature importance ({model_name})")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_feature_importance: %s", e)
        raise VisualizationError(
            f"Failed to plot feature importance for {model_name}"
        ) from e


def plot_roc_curves(
    results: dict[str, dict],
    y_test: pd.Series,
    output_path: str | None = None,
) -> None:
    """Plot ROC curves for multiple models.

    Credit: RGhazzal (1_0-rghazzal-machine-learning-models.ipynb, Cell 34)

    Args:
        results:     Dict keyed by model name, each value must contain 'y_prob'.
        y_test:      True binary test labels.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info(
        "[START] Plotting ROC curves [RGhazzal] — models: %s ...",
        list(results.keys()),
    )

    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        colors = [LEGIT_COLOR, NEUTRAL_COLOR, FRAUD_COLOR, AMBER_COLOR]
        fig, ax = plt.subplots(figsize=(8, 6))

        for (name, v), color in zip(results.items(), colors, strict=False):
            if "y_prob" not in v:
                logger.warning("  %s missing 'y_prob' — skipping", name)
                continue
            fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
            auc = roc_auc_score(y_test, v["y_prob"])
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")
            logger.info("  %s — ROC-AUC: %.4f", name, auc)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, output_path, "ROC curves")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_roc_curves: %s", e)
        raise VisualizationError("Failed to plot ROC curves") from e


def plot_precision_recall_curves(
    results: dict[str, dict],
    y_test: pd.Series,
    output_path: str | None = None,
) -> None:
    """Plot precision-recall curves for multiple models.

    Credit: RGhazzal (1_0-rghazzal-machine-learning-models.ipynb, Cell 37)

    Args:
        results:     Dict keyed by model name, each value must contain 'y_prob'.
        y_test:      True binary test labels.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info(
        "[START] Plotting precision-recall curves [RGhazzal] — models: %s ...",
        list(results.keys()),
    )

    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve

        colors = [LEGIT_COLOR, NEUTRAL_COLOR, FRAUD_COLOR, AMBER_COLOR]
        fig, ax = plt.subplots(figsize=(8, 6))

        for (name, v), color in zip(results.items(), colors, strict=False):
            if "y_prob" not in v:
                logger.warning("  %s missing 'y_prob' — skipping", name)
                continue
            prec, rec, _ = precision_recall_curve(y_test, v["y_prob"])
            ap = average_precision_score(y_test, v["y_prob"])
            ax.plot(rec, prec, color=color, label=f"{name} (AP={ap:.3f})")
            logger.info("  %s — Average Precision: %.4f", name, ap)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Precision-recall curves")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_precision_recall_curves: %s", e)
        raise VisualizationError("Failed to plot precision-recall curves") from e


def plot_confusion_matrices(
    results: dict[str, dict],
    y_test: pd.Series,
    output_path: str | None = None,
) -> None:
    """Plot side-by-side confusion matrices for multiple models.

    Credit: RGhazzal (1_0-rghazzal-machine-learning-models.ipynb, Cell 40)

    Args:
        results:     Dict keyed by model name, each value must contain 'y_pred'.
        y_test:      True binary test labels.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info(
        "[START] Plotting confusion matrices [RGhazzal] — models: %s ...",
        list(results.keys()),
    )

    try:
        from sklearn.metrics import confusion_matrix

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, v) in zip(axes, results.items(), strict=False):
            if "y_pred" not in v:
                logger.warning("  %s missing 'y_pred' — skipping", name)
                continue
            cm = confusion_matrix(y_test, v["y_pred"])
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{name}\n(N={len(y_test):,})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Legitimate", "Fraud"])
            ax.set_yticklabels(["Legitimate", "Fraud"])
            plt.colorbar(im, ax=ax)
            logger.info(
                "  %s — TP: %d  TN: %d  FP: %d  FN: %d",
                name,
                cm[1, 1],
                cm[0, 0],
                cm[0, 1],
                cm[1, 0],
            )

        fig.suptitle("Confusion Matrices", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Confusion matrices")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_confusion_matrices: %s", e)
        raise VisualizationError("Failed to plot confusion matrices") from e


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot a feature correlation heatmap.

    Credit: RGhazzal (1_0-rghazzal-exploratory-data-analysis.ipynb, Cell 32)
            Lohith   (1_0-lohith-data-exploration.ipynb, Cell 29)

    Args:
        df:          DataFrame to compute correlations from.
        columns:     Subset of columns to include. Uses all numeric if None.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting correlation heatmap [RGhazzal, Lohith] ...")

    try:
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise VisualizationError(
                    f"Missing columns for correlation heatmap: {missing}"
                )
            df = df[columns]

        corr = df.select_dtypes(include="number").corr()
        logger.info("  Features: %d", len(corr.columns))

        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(np.ma.masked_where(mask, corr), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Feature Correlation Heatmap")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Correlation heatmap")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_correlation_heatmap: %s", e)
        raise VisualizationError("Failed to plot correlation heatmap") from e


def plot_avg_spend_per_category(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Plot median transaction amount per merchant category by fraud label.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 13)

    Args:
        df:          DataFrame with 'category', 'amt', 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting avg spend per category [Lohith] ...")

    try:
        required = ["category", "amt", "is_fraud"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise VisualizationError(f"Missing required columns: {missing}")

        logger.info("  Categories: %d", df["category"].nunique())

        avg_spend = (
            df.groupby(["category", "is_fraud"])["amt"]
            .median()
            .unstack()
            .rename(columns={0: "Legitimate", 1: "Fraud"})
            .sort_values("Fraud", ascending=False)
        )
        logger.info(
            "  Highest fraud spend: %s ($%.2f)",
            avg_spend["Fraud"].idxmax(),
            avg_spend["Fraud"].max(),
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(avg_spend))
        width = 0.35
        ax.bar(
            [i - width / 2 for i in x],
            avg_spend["Legitimate"],
            width,
            label="Legitimate",
            color=LEGIT_COLOR,
            alpha=0.85,
        )
        ax.bar(
            [i + width / 2 for i in x],
            avg_spend["Fraud"],
            width,
            label="Fraud",
            color=FRAUD_COLOR,
            alpha=0.85,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(avg_spend.index, rotation=45, ha="right")
        ax.set_title("Median Transaction Amount per Category")
        ax.set_ylabel("Median Amount ($)")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, output_path, "Avg spend per category")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_avg_spend_per_category: %s", e)
        raise VisualizationError("Failed to plot avg spend per category") from e


def plot_customer_avg_spend_window(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Plot customer average spend across rolling time windows (1/7/30 day).

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 14)

    Args:
        df:          DataFrame with customer avg spend window columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting customer avg spend windows [Lohith] ...")

    try:
        window_cols = [
            "customer_avg_amout_1_day",
            "customer_avg_amount_7_day",
            "customer_avg_amount_30_day",
        ]
        available = [c for c in window_cols if c in df.columns]
        if not available:
            raise VisualizationError(
                "No customer avg spend window columns found in DataFrame"
            )

        logger.info("  Columns found: %s", available)

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        labels = ["1-day avg", "7-day avg", "30-day avg"]
        for ax, col, label in zip(axes, available, labels, strict=False):
            for fraud_val, color, name in [
                (0, LEGIT_COLOR, "Legitimate"),
                (1, FRAUD_COLOR, "Fraud"),
            ]:
                ax.hist(
                    df.loc[df["is_fraud"] == fraud_val, col],
                    bins=40,
                    alpha=0.6,
                    color=color,
                    label=name,
                )
            ax.set_title(label)
            ax.set_xlabel("Avg Amount ($)")
            ax.legend()

        fig.suptitle("Customer Average Spend by Time Window", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Customer avg spend window")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_customer_avg_spend_window: %s", e)
        raise VisualizationError("Failed to plot customer avg spend window") from e


def plot_customer_location_scatter(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Plot customer location scatter (lat/long) colored by fraud label.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 23)

    Args:
        df:          DataFrame with 'lat', 'long', 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting customer location scatter [Lohith] ...")

    try:
        required = ["lat", "long", "is_fraud"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise VisualizationError(
                f"Missing required columns for location scatter: {missing}"
            )

        sample = df.sample(min(5000, len(df)), random_state=42)
        logger.info(
            "  Sampled %d rows | Fraud points: %d",
            len(sample),
            int(sample["is_fraud"].sum()),
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        legit = sample[sample["is_fraud"] == 0]
        fraud = sample[sample["is_fraud"] == 1]
        ax.scatter(
            legit["long"],
            legit["lat"],
            c=LEGIT_COLOR,
            alpha=0.3,
            s=5,
            label="Legitimate",
        )
        ax.scatter(
            fraud["long"], fraud["lat"], c=FRAUD_COLOR, alpha=0.7, s=10, label="Fraud"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Customer Location — Fraud vs Legitimate")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, output_path, "Customer location scatter")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_customer_location_scatter: %s", e)
        raise VisualizationError("Failed to plot customer location scatter") from e


def plot_customer_transaction_count(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Plot customer transaction count distribution by fraud label.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 19)

    Args:
        df:          DataFrame with 'customer_num_trans_30_day' and 'is_fraud'.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting customer transaction count [Lohith] ...")

    try:
        col = "customer_num_trans_30_day"
        if col not in df.columns:
            raise VisualizationError(f"Column '{col}' not found in DataFrame")

        logger.info(
            "  Rows: %d | Fraud mean: %.2f | Legit mean: %.2f",
            len(df),
            df.loc[df["is_fraud"] == 1, col].mean(),
            df.loc[df["is_fraud"] == 0, col].mean(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        for fraud_val, color, name in [
            (0, LEGIT_COLOR, "Legitimate"),
            (1, FRAUD_COLOR, "Fraud"),
        ]:
            ax.hist(
                df.loc[df["is_fraud"] == fraud_val, col],
                bins=40,
                alpha=0.6,
                color=color,
                label=name,
            )
        ax.set_title("Customer Transaction Count (30-day window)")
        ax.set_xlabel("Transaction count")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, output_path, "Customer transaction count")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_customer_transaction_count: %s", e)
        raise VisualizationError("Failed to plot customer transaction count") from e


def plot_age_distribution_by_class(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Plot customer age distribution by fraud label.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 27)

    Args:
        df:          DataFrame with 'dob' and 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting age distribution by class [Lohith] ...")

    try:
        if "dob" not in df.columns:
            raise VisualizationError("Column 'dob' not found in DataFrame")

        df = df.copy()
        df["age"] = pd.to_datetime("today").year - pd.to_datetime(df["dob"]).dt.year
        logger.info(
            "  Rows: %d | Fraud mean age: %.1f | Legit mean age: %.1f",
            len(df),
            df.loc[df["is_fraud"] == 1, "age"].mean(),
            df.loc[df["is_fraud"] == 0, "age"].mean(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        for fraud_val, color, name in [
            (0, LEGIT_COLOR, "Legitimate"),
            (1, FRAUD_COLOR, "Fraud"),
        ]:
            ax.hist(
                df.loc[df["is_fraud"] == fraud_val, "age"],
                bins=30,
                alpha=0.6,
                color=color,
                label=name,
            )
        ax.set_title("Customer Age Distribution by Class")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, output_path, "Age distribution by class")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_age_distribution_by_class: %s", e)
        raise VisualizationError("Failed to plot age distribution by class") from e


def plot_fraud_risk_by_age(
    df: pd.DataFrame,
    bins: int = 10,
    output_path: str | None = None,
) -> None:
    """Plot fraud risk across customer age groups.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 27)

    Args:
        df:          DataFrame with 'dob' and 'is_fraud' columns.
        bins:        Number of age bins.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting fraud risk by age [Lohith] — bins: %d ...", int(bins))
    try:
        if "dob" not in df.columns:
            raise VisualizationError("Column 'dob' not found in DataFrame")

        df = df.copy()
        df["age"] = pd.to_datetime("today").year - pd.to_datetime(df["dob"]).dt.year
        df["age_group"] = pd.cut(df["age"], bins=bins)
        fraud_by_age = (
            df.groupby("age_group", observed=False)["is_fraud"]
            .mean()
            .mul(100)
            .reset_index()
        )
        logger.info(
            "  Highest risk age group: %s (%.2f%%)",
            str(fraud_by_age.loc[fraud_by_age["is_fraud"].idxmax(), "age_group"]),
            float(fraud_by_age["is_fraud"].max()),
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(
            fraud_by_age["age_group"].astype(str),
            fraud_by_age["is_fraud"],
            color=FRAUD_COLOR,
            alpha=0.8,
        )
        ax.set_title("Fraud Risk Across Age Groups")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Fraud Rate (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        _save_or_show(fig, output_path, "Fraud risk by age")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_fraud_risk_by_age: %s", e)
        raise VisualizationError("Failed to plot fraud risk by age") from e


def plot_heatmap_dow_vs_hour(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot fraud rate heatmap — day of week vs hour of day.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 18)

    Args:
        df:          DataFrame with 'trans_time_hrs', 'trans_time_day', 'is_fraud'.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting DOW vs hour heatmap [Lohith] ...")

    try:
        required = ["trans_time_hrs", "trans_time_day", "is_fraud"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise VisualizationError(
                f"Missing required columns for DOW vs hour heatmap: {missing}"
            )

        logger.info("  Rows: %d", len(df))
        pivot = (
            df.groupby(["trans_time_day", "trans_time_hrs"])["is_fraud"]
            .mean()
            .mul(100)
            .unstack()
        )
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        peak = pivot.stack().idxmax()
        logger.info(
            "  Peak fraud — day: %d, hour: %d (%.2f%%)",
            int(peak[0]),
            int(peak[1]),
            float(pivot.stack().max()),
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([dow_labels[i] for i in pivot.index if i < len(dow_labels)])
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        ax.set_title("Fraud Rate Heatmap — Day of Week vs Hour")
        plt.colorbar(im, ax=ax, label="Fraud Rate (%)")
        plt.tight_layout()
        _save_or_show(fig, output_path, "DOW vs hour heatmap")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_heatmap_dow_vs_hour: %s", e)
        raise VisualizationError("Failed to plot DOW vs hour heatmap") from e


def plot_label_distribution(
    labels: pd.Series,
    label_names: dict[int, str] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot 4-class label distribution for Pipeline A.

    Credit: Musaddiq (1_0-Musaddiq-logistic-regression.ipynb, Cell 10)

    Args:
        labels:      Series of integer labels (0-3).
        label_names: Optional dict mapping int to label name.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting 4-class label distribution [Musaddiq] ...")

    try:
        label_names = label_names or {
            0: "TT-Legit-Normal",
            1: "TF-Legit-Suspicious",
            2: "FT-Fraud-LowRisk",
            3: "FF-Fraud-HighRisk",
        }
        counts = labels.value_counts().sort_index()
        names = [label_names.get(i, str(i)) for i in counts.index]
        colors = [LEGIT_COLOR, NEUTRAL_COLOR, AMBER_COLOR, FRAUD_COLOR]

        for i, (name, cnt) in enumerate(zip(names, counts.values, strict=False)):
            logger.info(
                "  Class %d (%s): %d (%.2f%%)",
                i,
                name,
                cnt,
                cnt / len(labels) * 100,
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(names, counts.values, color=colors[: len(names)], alpha=0.85)
        axes[0].set_title("Label count")
        axes[0].set_ylabel("Count")
        plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

        axes[1].pie(
            counts.values,
            labels=names,
            colors=colors[: len(names)],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("Label proportion")

        fig.suptitle("4-Class Label Distribution (Pipeline A)", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Label distribution")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_label_distribution: %s", e)
        raise VisualizationError("Failed to plot label distribution") from e


def plot_fraud_by_state(
    df: pd.DataFrame,
    top_n: int = 15,
    output_path: str | None = None,
) -> None:
    """Plot total fraud volume by state (top N states).

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 22)

    Args:
        df:          DataFrame with 'state' and 'is_fraud' columns.
        top_n:       Number of top states to display.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting fraud by state [Lohith] — top %d ...", int(top_n))

    try:
        if "state" not in df.columns:
            raise VisualizationError("Column 'state' not found in DataFrame")

        top_states = df["state"].value_counts().head(top_n).index
        fraud_by_state = (
            df[df["state"].isin(top_states)]
            .groupby("state")["is_fraud"]
            .agg(fraud_count="sum", total="count")
            .assign(fraud_rate=lambda x: x["fraud_count"] / x["total"] * 100)
            .sort_values("fraud_count", ascending=True)
        )
        logger.info(
            "  Highest fraud volume: %s (%d transactions)",
            str(fraud_by_state["fraud_count"].idxmax()),
            int(fraud_by_state["fraud_count"].max()),
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(
            fraud_by_state.index,
            fraud_by_state["fraud_count"],
            color=FRAUD_COLOR,
            alpha=0.85,
        )
        axes[0].set_title(f"Total Fraud Volume — Top {top_n} States")
        axes[0].set_xlabel("Fraud count")

        axes[1].barh(
            fraud_by_state.index,
            fraud_by_state["fraud_rate"],
            color=AMBER_COLOR,
            alpha=0.85,
        )
        axes[1].set_title(f"Fraud Rate — Top {top_n} States")
        axes[1].set_xlabel("Fraud rate (%)")

        fig.suptitle("Fraud by State", fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, output_path, "Fraud by state")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_fraud_by_state: %s", e)
        raise VisualizationError("Failed to plot fraud by state") from e


def plot_transactions_by_hour(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot total transaction count by hour of day.

    Credit: Lohith (1_0-lohith-data-exploration.ipynb, Cell 16)

    Args:
        df:          DataFrame with 'trans_time_hrs' and 'is_fraud' columns.
        output_path: Optional path to save the plot.

    Raises:
        VisualizationError: If the plot cannot be created or saved.
    """
    logger.info("[START] Plotting transactions by hour [Lohith] ...")

    try:
        if "trans_time_hrs" not in df.columns:
            raise VisualizationError("Column 'trans_time_hrs' not found in DataFrame")

        hour_counts = df["trans_time_hrs"].value_counts().sort_index()
        night_hours = list(range(0, 6)) + list(range(22, 24))
        colors = [
            FRAUD_COLOR if h in night_hours else NEUTRAL_COLOR
            for h in hour_counts.index
        ]

        logger.info(
            "  Peak hour: %d:00 (%d transactions)",
            int(hour_counts.idxmax()),
            int(hour_counts.max()),
        )

        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(hour_counts.index, hour_counts.values, color=colors, alpha=0.85)
        ax.set_title("Transactions by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Transaction count")
        ax.set_xticks(range(24))
        ax.legend(
            handles=[
                Patch(color=NEUTRAL_COLOR, label="Daytime"),
                Patch(color=FRAUD_COLOR, label="Night (22:00-06:00)"),
            ]
        )
        plt.tight_layout()
        _save_or_show(fig, output_path, "Transactions by hour")

    except VisualizationError:
        raise
    except Exception as e:
        logger.error("Unexpected error in plot_transactions_by_hour: %s", e)
        raise VisualizationError("Failed to plot transactions by hour") from e
