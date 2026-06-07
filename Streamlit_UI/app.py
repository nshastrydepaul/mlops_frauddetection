from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlops_frauddetection.features.build_features import engineer_features_ensemble
import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

MODEL_API_NAMES = {
    "LightGBM": "lightgbm",
    "Random Forest": "randomforest",
    "XGBoost": "xgboost",
    "Logistic Regression Binary": "logisticregression",
}

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Fraud Detection MLOps Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# =========================
# Model configuration
# =========================

REQUIRED_FEATURES = [
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

METADATA_FILES = {
"Logistic Regression Binary": MODELS_DIR / "RGhazzal_logisticregression_metadata.json",
"Random Forest": MODELS_DIR / "RGhazzal_randomforest_metadata.json",
"LightGBM": MODELS_DIR / "RGhazzal_lightgbm_metadata.json",
"XGBoost": MODELS_DIR / "RGhazzal_xgboost_metadata.json",
}

# =========================
# Helpers
# =========================

def validate_input_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_FEATURES if col not in df.columns]


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    return df[REQUIRED_FEATURES].copy()


def read_json(path: Path):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def flatten_latest_metadata(model_name: str, path: Path) -> dict | None:
    data = read_json(path)

    if not data:
        return None

    if isinstance(data, dict):
        # Your ensemble metadata is usually stored under date keys.
        latest_key = sorted(data.keys())[-1]
        latest = data[latest_key]

        if isinstance(latest, dict):
            return {
                "Model": model_name,
                "Training Date": latest.get("training_date", latest_key),
                "Train Samples": latest.get("train_samples"),
                "Test Samples": latest.get("test_samples"),
                "Features": latest.get("features"),
                "Accuracy": latest.get("test_acc"),
                "Precision": latest.get("precision"),
                "Recall": latest.get("recall"),
                "F1": latest.get("f1"),
                "ROC-AUC": latest.get("roc_auc"),
                "Average Precision": latest.get("avg_prec"),
                "CV Mean F1": latest.get("cv_mean_f1"),
                "CV Std F1": latest.get("cv_std_f1"),
            }

    return None


def log_prediction(model_name: str, rows: int, fraud_count: int) -> None:
    st.session_state.setdefault("predictions_log", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "rows": rows,
            "fraud_count": fraud_count,
        }
    )


# =========================
# Sidebar
# =========================
def render_sidebar(models: dict):
    with st.sidebar:
        st.title("🚨 Fraud MLOps")
        st.caption("Credit Card Fraud Detection Dashboard")

        st.markdown("---")
        st.subheader("Model Status")

        st.metric("API Models Available", len(models))
        st.metric("Available Algorithms", len(models))

        if models:
            st.success("API models ready")
            st.write(", ".join(models.keys()))
        else:
            st.error("No models loaded")

        st.markdown("---")

        show_raw = st.checkbox("Show raw prediction data", value=True)

        log = st.session_state.get("predictions_log", [])
        if log:
            st.markdown("---")
            st.subheader("Session Analytics")
            st.metric("Prediction Runs", len(log))
            st.metric("Rows Scored", sum(x["rows"] for x in log))
            st.metric("Fraud Flags", sum(x["fraud_count"] for x in log))


    return show_raw


# =========================
# Dashboard
# =========================
def render_dashboard(models: dict):
    st.header("Executive Dashboard")

    st.subheader("Project Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Data Processing",
                "Feature Engineering",
                "Model Training",
                "Experiment Tracking",
                "Batch Prediction",
                "Deployment",
            ],
            "Tool / Component": [
                "Pandas / DVC",
                "Custom feature pipeline",
                "Scikit-Learn / XGBoost / LightGBM",
                "MLflow",
                "Streamlit CSV Upload",
                "Hugging Face Spaces + GitHub Actions",
            ],
            "Status": [
                "Implemented",
                "Implemented",
                "Implemented",
                "Implemented",
                "Implemented in this UI",
                "Ready for deployment",
            ],
        }
    )

    st.dataframe(workflow, use_container_width=True)

    st.markdown("---")

    if st.session_state.get("predictions_log"):
        log_df = pd.DataFrame(st.session_state["predictions_log"])
        st.subheader("Current Session Prediction Activity")
        st.dataframe(log_df, use_container_width=True)

        fig = px.bar(
            log_df,
            x="model",
            y="fraud_count",
            title="Fraud Flags by Model in Current Session",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction runs yet. Use the Batch Prediction tab to score a CSV.")

def call_pipeline_b_api(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    api_model_name = MODEL_API_NAMES[model_name]
    predictions = []

    for _, row in df.iterrows():
        payload = {}

        for col in REQUIRED_FEATURES:
            value = row[col]

            if pd.isna(value):
                value = 0

            if isinstance(value, (np.integer, int)):
                payload[col] = int(value)
            elif isinstance(value, (np.floating, float)):
                payload[col] = float(value)
            else:
                payload[col] = value

        response = requests.post(
            f"{API_URL}/predict/pipeline-b/{api_model_name}",
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        predictions.append(response.json())

    return pd.DataFrame(predictions)

# =========================
# Data Explorer
# =========================
def render_data_explorer():
    st.header("Data Explorer")

    if not DATA_DIR.exists():
        st.warning("No data/ folder found.")
        return

    csv_files = list(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        st.info("No CSV files found inside data/.")
        return

    selected_file = st.selectbox(
    "Select dataset",
    csv_files,
    format_func=lambda p: str(p.relative_to(PROJECT_ROOT)),
    key="reports_dataset_dataexplorer")

    df = pd.read_csv(selected_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{df.isna().sum().sum():,}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select numeric column",
                                    numeric_cols,
                                    key="data_explorer_numeric_col",)
        fig = px.histogram(
            df,
            x=selected_col,
            nbins=50,
            title=f"Distribution of {selected_col}",
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Batch Prediction
# =========================
def render_batch_prediction(models: dict, show_raw: bool):
    st.header("Batch Prediction")

    if not models:
        st.error("No models loaded. Check the models/ folder and required packages.")
        return

    selected_model = st.selectbox("Select Model", list(models.keys()))

    st.info(
        "Upload a processed feature dataset in the same format as the training/test feature file. "
        "The file must contain the required engineered feature columns."
    )

    uploaded = st.file_uploader(
        "Upload Processed Feature CSV",
        type=["csv"],
    )

    with st.expander("Expected Feature Columns"):
        st.code("\n".join(REQUIRED_FEATURES))

    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    df = engineer_features_ensemble(df)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    missing = validate_input_columns(df)

    if missing:
        st.error("Uploaded file is missing required feature columns.")
        st.write(missing)
        return

    if st.button("Process Batch", type="primary"):

        try:
            results = call_pipeline_b_api(df, selected_model)
        except Exception as e:
            st.error("Prediction API failed.")
            st.write(e)
            return
        
        # Apply UI threshold to probability output.
        results["threshold_label"] = results["label"].replace({
            "Legit": "Legitimate",
             "Fraud": "Fraud",
            })
        
        final_df = pd.concat([df.reset_index(drop=True), results], axis=1)

        fraud_count = int((final_df["threshold_label"] == "Fraud").sum())
        legit_count = int((final_df["threshold_label"] == "Legitimate").sum())

        log_prediction(selected_model, len(final_df), fraud_count)

        st.subheader("Batch Prediction Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", len(final_df))
        col2.metric("Fraud Flags", fraud_count)
        col3.metric("Legitimate", legit_count)
        col4.metric("Fraud Rate", f"{fraud_count / len(final_df):.2%}")

        fig = px.pie(
            final_df,
            names="threshold_label",
            title=f"Fraud vs Legitimate Predictions — {selected_model}",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Risk Level Distribution")

        risk_counts = final_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_risk = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            title="Transactions by Risk Level",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("---")
        st.subheader("Prediction Results")
        st.dataframe(final_df, use_container_width=True)
        
        if show_raw:
            with st.expander("View Full Prediction Results"):
                st.dataframe(final_df, use_container_width=True)

        csv = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name=f"{selected_model.lower().replace(' ', '_')}_fraud_predictions.csv",
            mime="text/csv",
        )

# =========================
# Model Comparison
# =========================
def render_model_comparison():
    st.header("Model Comparison")

    rows = []

    for model_name, path in METADATA_FILES.items():
        row = flatten_latest_metadata(model_name, path)
        if row:
            rows.append(row)

    if not rows:
        st.info("No model metadata JSON files found or readable.")
        return

    df = pd.DataFrame(rows)

    st.subheader("Latest Model Metrics")
    st.dataframe(df, use_container_width=True)

    if "ROC-AUC" in df.columns:
        best_model = df.sort_values("ROC-AUC", ascending=False).iloc[0]

        st.markdown("---")
        st.subheader("Best Performing Model")

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", best_model["Model"])
        col2.metric("ROC-AUC", f"{best_model['ROC-AUC']:.4f}")
        col3.metric("F1 Score", f"{best_model['F1']:.4f}")

    metric_cols = [
        col for col in
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]
        if col in df.columns
    ]

    if metric_cols:
        plot_df = df.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score",
        )

        fig = px.bar(
            plot_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Performance Comparison",
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================
# Reports
# =========================
def render_reports():
    st.header("Fraud Analytics Dashboard")

    REPORT_DATASET = DATA_DIR / "raw" / "data_25k.csv"
    if not REPORT_DATASET.exists():
        st.error(f"Report dataset not found: {REPORT_DATASET}")
        return
    
    df = pd.read_csv(REPORT_DATASET)

    st.success(f"Loaded report dataset: {REPORT_DATASET.name}")

    fraud_count = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    total_count = len(df)
    legit_count = total_count - fraud_count
    fraud_rate = fraud_count / total_count if total_count > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_count:,}")
    col2.metric("Fraud Transactions", f"{fraud_count:,}")
    col3.metric("Legitimate Transactions", f"{legit_count:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.2%}")

    st.markdown("---")

    st.dataframe(df.head(50), use_container_width=True)

    report_type = st.selectbox(
        "Select report",
        [
            "Class Balance",
            "Amount Distribution",
            "Fraud Rate by Category",
            "Average Spend by Category",
            "Transactions by Hour",
            "Fraud Rate by Day",
            "Fraud Heatmap Day vs Hour",
            "Customer Transaction Windows",
            "Customer Average Spend Windows",
            "Top States by Fraud Volume",
            "Location Scatter",
            "Age Distribution",
            "Fraud Risk by Age Group",
            "Correlation Heatmap",
        ],
        key="fraud_reports_type",
    )

    if "is_fraud" not in df.columns:
        st.error("Dataset must contain `is_fraud` column for these reports.")
        return

    if report_type == "Class Balance":
        counts = df["is_fraud"].value_counts().reset_index()
        counts.columns = ["is_fraud", "count"]

        fig = px.bar(
            counts,
            x="is_fraud",
            y="count",
            color="is_fraud",
            title="Fraud vs Legitimate Transactions",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Amount Distribution":
        if "amt" not in df.columns:
            st.error("Column `amt` not found.")
            return

        fig = px.histogram(
            df,
            x="amt",
            color="is_fraud",
            nbins=60,
            marginal="box",
            title="Amount Distribution: Legitimate vs Fraud",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Fraud Rate by Category":
        if "category" not in df.columns:
            st.error("Column `category` not found.")
            return

        temp = (
            df.groupby("category")["is_fraud"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        temp.columns = ["category", "fraud_rate"]

        fig = px.bar(
            temp,
            x="fraud_rate",
            y="category",
            orientation="h",
            title="Fraud Rate by Merchant Category (%)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Average Spend by Category":
        if "category" not in df.columns or "amt" not in df.columns:
            st.error("Columns `category` and `amt` are required.")
            return

        temp = (
            df.groupby("category")["amt"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            temp,
            x="category",
            y="amt",
            title="Average Transaction Amount by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Transactions by Hour":
        if "trans_time_hrs" not in df.columns:
            st.error("Column `trans_time_hrs` not found.")
            return

        hourly = (
            df.groupby(["trans_time_hrs", "is_fraud"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            hourly,
            x="trans_time_hrs",
            y="count",
            color="is_fraud",
            barmode="group",
            title="Transactions by Hour of Day",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Fraud Rate by Day":
        if "trans_time_day" not in df.columns:
            st.error("Column `trans_time_day` not found.")
            return

        day_map = {
            0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
            4: "Fri", 5: "Sat", 6: "Sun",
        }

        temp = (
            df.groupby("trans_time_day")["is_fraud"]
            .mean()
            .mul(100)
            .reset_index()
        )
        temp["day"] = temp["trans_time_day"].map(day_map)

        fig = px.bar(
            temp,
            x="day",
            y="is_fraud",
            title="Fraud Rate by Day of Week (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Fraud Heatmap Day vs Hour":
        if "trans_time_day" not in df.columns or "trans_time_hrs" not in df.columns:
            st.error("Columns `trans_time_day` and `trans_time_hrs` are required.")
            return

        fraud_only = df[df["is_fraud"] == 1]

        pivot = (
            fraud_only.groupby(["trans_time_day", "trans_time_hrs"])
            .size()
            .reset_index(name="count")
            .pivot(index="trans_time_day", columns="trans_time_hrs", values="count")
            .fillna(0)
        )

        fig = px.imshow(
            pivot,
            aspect="auto",
            text_auto=True,
            title="Fraud Count Heatmap: Day of Week vs Hour",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Customer Transaction Windows":
        cols = [
            "customer_num_trans_1_day",
            "customer_num_trans_7_day",
            "customer_num_trans_30_day",
        ]

        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        selected_col = st.selectbox(
            "Select transaction window",
            cols,
            key="customer_txn_window_col",
        )

        fig = px.histogram(
            df,
            x=selected_col,
            color="is_fraud",
            nbins=40,
            marginal="box",
            title=f"Customer Transaction Window: {selected_col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Customer Average Spend Windows":
        cols = [
            "customer_avg_amout_1_day",
            "customer_avg_amount_7_day",
            "customer_avg_amount_30_day",
        ]

        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        selected_col = st.selectbox(
            "Select average spend window",
            cols,
            key="customer_avg_spend_window_col",
        )

        fig = px.histogram(
            df,
            x=selected_col,
            color="is_fraud",
            nbins=40,
            marginal="box",
            title=f"Customer Average Spend Window: {selected_col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Top States by Fraud Volume":
        if "state" not in df.columns:
            st.error("Column `state` not found.")
            return

        top_n = st.slider(
            "Top N states",
            5,
            30,
            15,
            key="top_states_slider",
        )

        temp = (
            df[df["is_fraud"] == 1]["state"]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        temp.columns = ["state", "fraud_count"]

        fig = px.bar(
            temp,
            x="state",
            y="fraud_count",
            title=f"Top {top_n} States by Fraud Volume",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Location Scatter":
        if "lat" not in df.columns or "long" not in df.columns:
            st.error("Columns `lat` and `long` are required.")
            return

        sample_size = st.slider(
            "Sample size",
            1000,
            min(50000, len(df)),
            min(10000, len(df)),
            key="location_sample_size",
        )

        sample = df.sample(n=sample_size, random_state=42)

        fig = px.scatter(
            sample,
            x="long",
            y="lat",
            color="is_fraud",
            title="Customer Locations Colored by Fraud",
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Age Distribution":
        if "dob" not in df.columns:
            st.error("Column `dob` not found.")
            return

        temp = df.copy()
        temp["dob"] = pd.to_datetime(temp["dob"], errors="coerce")
        temp["age"] = (pd.Timestamp.today() - temp["dob"]).dt.days // 365

        fig = px.histogram(
            temp,
            x="age",
            color="is_fraud",
            nbins=40,
            marginal="box",
            title="Customer Age Distribution by Class",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Fraud Risk by Age Group":
        if "dob" not in df.columns:
            st.error("Column `dob` not found.")
            return

        temp = df.copy()
        temp["dob"] = pd.to_datetime(temp["dob"], errors="coerce")
        temp["age"] = (pd.Timestamp.today() - temp["dob"]).dt.days // 365

        temp["age_group"] = pd.cut(
            temp["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
        )

        grouped = (
            temp.groupby("age_group", observed=True)["is_fraud"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            grouped,
            x="age_group",
            y="is_fraud",
            title="Fraud Risk Across Age Groups (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Correlation Heatmap":
        drop_cols = ["cc_num", "acct_num", "unix_time", "zip", "trans_time_secs"]

        numeric_df = df.select_dtypes(include=np.number).drop(
            columns=drop_cols,
            errors="ignore",
        )

        selected_cols = st.multiselect(
            "Select columns for correlation",
            numeric_df.columns.tolist(),
            default=numeric_df.columns.tolist()[: min(12, len(numeric_df.columns))],
            key="correlation_selected_cols",
        )

        if not selected_cols:
            st.warning("Select at least one numeric column.")
            return

        corr = numeric_df[selected_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# About
# =========================
def render_about():
    st.header("About This Project")

    st.markdown(
        """
        ### Credit Card Fraud Detection Dashboard

        This application provides an interactive interface for detecting potentially
        fraudulent credit card transactions using machine learning models deployed
        through a cloud-hosted prediction API.

        ### Project Objective

        The objective of this project is to identify suspicious financial transactions
        by analyzing customer behavior, transaction patterns, merchant characteristics,
        and engineered fraud indicators.

        ### Models Used

        - Logistic Regression
        - Random Forest
        - LightGBM
        - XGBoost

        ### Feature Engineering

        The application automatically generates fraud-related features such as:

        - Log-transformed transaction amount
        - Customer transaction velocity
        - Merchant risk indicators
        - Online transaction flags
        - Time-based transaction features
        - Customer spending behavior metrics
        - Category-based transaction features

        ### Application Features

        - Data exploration and visualization
        - Batch fraud prediction from uploaded CSV files
        - Risk-level classification
        - Model comparison
        - Fraud analytics reports
        - Downloadable prediction results

        ### Deployment Architecture

        - Streamlit frontend hosted on Hugging Face Spaces
        - FastAPI prediction service hosted on Google Cloud Run
        - Cloud API integration using HTTP requests
        - GitHub-based deployment workflow

        ### Prediction Workflow

        1. Upload transaction data in CSV format
        2. Feature engineering is applied
        3. Selected model sends data to the Cloud Run API
        4. Fraud prediction and risk level are returned
        5. Results can be reviewed and downloaded

        ### Important Note

        Uploaded CSV files should follow the transaction schema expected by the
        feature engineering pipeline. Predictions are generated through the deployed
        FastAPI service, not by loading models directly inside the Streamlit app.
        """
    )


# =========================
# Main
# =========================
def main():
    models = MODEL_API_NAMES
    show_raw = render_sidebar(models)

    st.title("🚨 Fraud Detection MLOps Dashboard")
    st.caption("End-to-end Streamlit UI for fraud analytics, model comparison, and batch prediction.")

    tabs = st.tabs(
        [
            "Dashboard",
            "Data Explorer",
            "Batch Prediction",
            "Model Comparison",
            "Reports",
            "About",
        ]
    )

    with tabs[0]:
        render_dashboard(models)

    with tabs[1]:
        render_data_explorer()

    with tabs[2]:
        render_batch_prediction(models, show_raw)

    with tabs[3]:
        render_model_comparison()

    with tabs[4]:
        render_reports()

    with tabs[5]:
        render_about()


if __name__ == "__main__":
    main()
