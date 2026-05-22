"""Generate all EDA and model evaluation report figures.

Runs all visualization functions in sequence, tracking progress
and timing per plot. Failed plots are logged and skipped rather
than aborting the full report generation.

Usage
-----
    python -m mlops_frauddetection.generate_reports
    python -m mlops_frauddetection.generate_reports --data-path data/processed
    python -m mlops_frauddetection.generate_reports --figures-path reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlops_frauddetection.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_frauddetection.data.exceptions import VisualizationError
from mlops_frauddetection.logging_config import (
    get_logger,
    get_progress,
    setup_logging,
    timer,
)
from mlops_frauddetection.visualization.visualize import (
    plot_age_distribution_by_class,
    plot_amount_distribution,
    plot_avg_spend_per_category,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_customer_avg_spend_window,
    plot_customer_location_scatter,
    plot_customer_transaction_count,
    plot_fraud_by_category,
    plot_fraud_by_state,
    plot_fraud_by_time,
    plot_fraud_risk_by_age,
    plot_heatmap_dow_vs_hour,
    plot_label_distribution,
    plot_transactions_by_hour,
)

logger = get_logger(__name__)


def generate_reports(data_path: Path, figures_path: Path) -> None:
    """Load data and generate all report figures.

    Args:
        data_path:    Directory containing processed CSVs.
        figures_path: Directory to save generated figures.
    """
    # LOAD PROCESSED DATA
    logger.info("[START] Loading processed data from %s ...", data_path)
    with timer(logger, "Processed data load"):
        x_test = pd.read_csv(data_path / "X_test.csv")
        y_test = pd.read_csv(data_path / "y_test.csv").squeeze()
        x_train = pd.read_csv(data_path / "X_train.csv")
        y_train = pd.read_csv(data_path / "y_train.csv").squeeze()

    # Combine train + test for EDA plots that need full dataset
    df_full = pd.concat([x_train, x_test], ignore_index=True)
    df_full["is_fraud"] = pd.concat([y_train, y_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    logger.info(
        "Processed data loaded — train: %s, test: %s, full: %s",
        x_train.shape,
        x_test.shape,
        df_full.shape,
    )

    # LOAD RAW DATA — needed for plots requiring pre-encoded columns
    # (category, state, lat, long, dob)
    logger.info("[START] Loading raw data from %s ...", RAW_DATA_DIR)
    with timer(logger, "Raw data load"):
        df_raw = pd.read_csv(RAW_DATA_DIR / "data_100k.csv")
    logger.info("Raw data loaded — shape: %s", df_raw.shape)

    # DEFINE ALL PLOTS
    # Each entry: (label, output_filename, callable)
    plots = [
        (
            "Class distribution",
            "class_balance.png",
            lambda df=df_full, fp=figures_path: plot_class_distribution(
                df["is_fraud"],
                str(fp / "class_balance.png"),
            ),
        ),
        (
            "Amount distribution",
            "amount_distribution.png",
            lambda df=df_full, fp=figures_path: plot_amount_distribution(
                df,
                str(fp / "amount_distribution.png"),
            ),
        ),
        (
            "Fraud by category",
            "fraud_rate_by_merchant_category.png",
            lambda df=df_raw, fp=figures_path: plot_fraud_by_category(
                df,
                str(fp / "fraud_rate_by_merchant_category.png"),
            ),
        ),
        (
            "Avg spend per category",
            "avg_spend_per_category.png",
            lambda df=df_raw, fp=figures_path: plot_avg_spend_per_category(
                df,
                str(fp / "avg_spend_per_category.png"),
            ),
        ),
        (
            "Customer avg spend window",
            "customer_average_spend_window.png",
            lambda df=df_full, fp=figures_path: plot_customer_avg_spend_window(
                df,
                str(fp / "customer_average_spend_window.png"),
            ),
        ),
        (
            "Fraud by time",
            "fraud_rate_by_day_of_week.png",
            lambda df=df_full, fp=figures_path: plot_fraud_by_time(
                df,
                str(fp / "fraud_rate_by_day_of_week.png"),
            ),
        ),
        (
            "Transactions by hour",
            "transactions_by_hour_of_day.png",
            lambda df=df_full, fp=figures_path: plot_transactions_by_hour(
                df,
                str(fp / "transactions_by_hour_of_day.png"),
            ),
        ),
        (
            "DOW vs hour heatmap",
            "heatmap_days_of_week_vs_hour.png",
            lambda df=df_full, fp=figures_path: plot_heatmap_dow_vs_hour(
                df,
                str(fp / "heatmap_days_of_week_vs_hour.png"),
            ),
        ),
        (
            "Customer transaction count",
            "customer_transaction_count.png",
            lambda df=df_full, fp=figures_path: plot_customer_transaction_count(
                df,
                str(fp / "customer_transaction_count.png"),
            ),
        ),
        (
            "Fraud by state",
            "total_fraud_volume_by_state.png",
            lambda df=df_raw, fp=figures_path: plot_fraud_by_state(
                df,
                output_path=str(fp / "total_fraud_volume_by_state.png"),
            ),
        ),
        (
            "Customer location scatter",
            "customer_location_scatter.png",
            lambda df=df_raw, fp=figures_path: plot_customer_location_scatter(
                df,
                output_path=str(fp / "customer_location_scatter.png"),
            ),
        ),
        (
            "Age distribution by class",
            "cutomer_age_distribution_by_class.png",
            lambda df=df_raw, fp=figures_path: plot_age_distribution_by_class(
                df,
                str(fp / "cutomer_age_distribution_by_class.png"),
            ),
        ),
        (
            "Fraud risk by age",
            "fraud_risk_across_age_groups.png",
            lambda df=df_raw, fp=figures_path: plot_fraud_risk_by_age(
                df,
                output_path=str(fp / "fraud_risk_across_age_groups.png"),
            ),
        ),
        (
            "Correlation heatmap",
            "correlation_heatmap.png",
            lambda df=df_full, fp=figures_path: plot_correlation_heatmap(
                df,
                output_path=str(fp / "correlation_heatmap.png"),
            ),
        ),
        # Pipeline A — Musaddiq
        (
            "Label distribution",
            "label_distribution.png",
            lambda y=y_all, fp=figures_path: plot_label_distribution(
                y,
                output_path=str(fp / "label_distribution.png"),
            ),
        ),
    ]

    # RUN ALL PLOTS WITH PROGRESS BAR
    passed = []
    failed = []

    with get_progress() as progress:
        task = progress.add_task("Generating report figures", total=len(plots))

        for label, filename, fn in plots:
            progress.update(task, description=label)
            try:
                with timer(logger, label):
                    fn()
                passed.append(filename)
            except VisualizationError as e:
                logger.error("[SKIP] %s failed — %s", label, e)
                failed.append((filename, str(e)))
            except Exception as e:
                logger.error("[SKIP] %s unexpected error — %s", label, e)
                failed.append((filename, str(e)))
            finally:
                progress.advance(task)

    # SUMMARY
    logger.info("=" * 60)
    logger.info("Report generation complete")
    logger.info("  Passed : %d / %d", len(passed), len(plots))
    logger.info("  Failed : %d / %d", len(failed), len(plots))
    if failed:
        logger.warning("Failed plots:")
        for filename, reason in failed:
            logger.warning("  %s — %s", filename, reason)
    logger.info("Figures saved to: %s", figures_path)
    logger.info("=" * 60)


def main() -> None:
    """CLI entrypoint for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate all EDA and model evaluation report figures"
    )
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figures-path", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    setup_logging()

    try:
        generate_reports(args.data_path, args.figures_path)
    except Exception as e:
        logger.critical("Report generation aborted — unexpected error: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
