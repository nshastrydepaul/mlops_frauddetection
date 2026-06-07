"""FastAPI service for fraud detection predictions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import functions_framework

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts fraud risk class for credit card transactions",
    version="1.0.0",
)

# 4-class label map (Pipeline A)
label_map = {
    0: "TT-Legit-Normal",
    1: "TF-Legit-Suspicious",
    2: "FT-Fraud-LowRisk",
    3: "FF-Fraud-HighRisk",
}

# GCS bucket config
GCS_BUCKET = os.environ.get("GCS_BUCKET", "fraud-detection-mlops-data")
GCS_MODELS_PREFIX = os.environ.get("GCS_MODELS_PREFIX", "models")

# Local model paths (fallback for local development)
LOCAL_MODELS_A = {
    "lr_balanced": "models/lr_balanced_20260606.joblib",
    "lr_smote": "models/lr_smote_20260521.joblib",
}
LOCAL_MODELS_B = {
    "lightgbm": "models/lightgbm_20260606.joblib",
    "randomforest": "models/randomforest_20260606.joblib",
    "xgboost": "models/xgboost_20260606.joblib",
    "logisticregression": "models/logisticregression_20260606.joblib",
}

# GCS model paths
GCS_MODELS_A = {
    "lr_balanced": "lr_balanced_20260606.joblib",
    "lr_smote": "lr_smote_20260521.joblib",
}
GCS_MODELS_B = {
    "lightgbm": "lightgbm_20260606.joblib",
    "randomforest": "randomforest_20260606.joblib",
    "xgboost": "xgboost_20260606.joblib",
    "logisticregression": "logisticregression_20260606.joblib",
}

_model_cache: dict = {}


def _load_model(model_name: str, is_pipeline_b: bool = False):
    """Load model from GCS or local fallback. Cache in memory."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Try GCS first
    try:
        from google.cloud import storage
        gcs_filename = GCS_MODELS_B[model_name] if is_pipeline_b else GCS_MODELS_A[model_name]
        gcs_path = f"{GCS_MODELS_PREFIX}/{gcs_filename}"

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            m = joblib.load(tmp.name)

        _model_cache[model_name] = m
        return m

    except Exception:
        # Fallback to local
        local_path = LOCAL_MODELS_B[model_name] if is_pipeline_b else LOCAL_MODELS_A[model_name]
        if Path(local_path).exists():
            m = joblib.load(local_path)
            _model_cache[model_name] = m
            return m
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' not found locally or in GCS")


# Load default model at startup
try:
    default_model = _load_model("lr_balanced")
except Exception:
    default_model = None


#  Pipeline A input (44 features) 

class TransactionA(BaseModel):
    """Pipeline A transaction  44 features, 4-class output."""
    city_pop: float
    job: float
    amt: float
    merchant: float
    customer_num_trans_1_day: float
    customer_num_trans_7_day: float
    customer_num_trans_30_day: float
    trans_time_secs: float
    trans_time_hrs: float
    trans_time_is_night: int
    trans_time_day: int
    trans_date_is_weekend: int
    customer_avg_amout_1_day: float
    customer_avg_amount_7_day: float
    customer_avg_amount_30_day: float
    merchant_num_trans_1_day: float
    merchant_num_trans_7_day: float
    merchant_num_trans_30_day: float
    merchant_risk_1_day: float
    merchant_risk_7_day: float
    merchant_risk_30_day: float
    merchant_risk_90_day: float
    customer_txn_count: float
    avg_amt_per_customer: float
    merchant_txn_count: float
    gender_M: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_kids_pets: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int
    amt_ratio: float
    combined_risk: float
    amt_risk_score: float
    is_high_spend: int
    night_high_amt: int


#  Pipeline B input (38 features) 

class TransactionB(BaseModel):
    """Pipeline B transaction  38 features, binary output (Fraud/Legit)."""
    log_amt: float
    city_pop: float
    customer_num_trans_1_day: float
    customer_num_trans_7_day: float
    customer_num_trans_30_day: float
    customer_avg_amout_1_day: float
    customer_avg_amount_7_day: float
    customer_avg_amount_30_day: float
    merchant_num_trans_1_day: float
    merchant_num_trans_7_day: float
    merchant_num_trans_30_day: float
    merchant_risk_1_day: float
    merchant_risk_7_day: float
    merchant_risk_30_day: float
    merchant_risk_90_day: float
    trans_time_hrs: float
    trans_hour_sin: float
    trans_hour_cos: float
    trans_time_is_night: int
    trans_date_is_weekend: int
    trans_time_day: int
    category_is_online: int
    online_x_log_amt: float
    velocity_ratio: float
    gender_M: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_kids_pets: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int


def _build_features_a(t: TransactionA) -> np.ndarray:
    return np.array([[
        t.city_pop, t.job, t.amt, t.merchant,
        t.customer_num_trans_1_day, t.customer_num_trans_7_day, t.customer_num_trans_30_day,
        t.trans_time_secs, t.trans_time_hrs, t.trans_time_is_night,
        t.trans_time_day, t.trans_date_is_weekend,
        t.customer_avg_amout_1_day, t.customer_avg_amount_7_day, t.customer_avg_amount_30_day,
        t.merchant_num_trans_1_day, t.merchant_num_trans_7_day, t.merchant_num_trans_30_day,
        t.merchant_risk_1_day, t.merchant_risk_7_day, t.merchant_risk_30_day, t.merchant_risk_90_day,
        t.customer_txn_count, t.avg_amt_per_customer, t.merchant_txn_count,
        t.gender_M,
        t.category_food_dining, t.category_gas_transport, t.category_grocery_net,
        t.category_grocery_pos, t.category_health_fitness, t.category_home,
        t.category_kids_pets, t.category_misc_net, t.category_misc_pos,
        t.category_personal_care, t.category_shopping_net, t.category_shopping_pos,
        t.category_travel,
        t.amt_ratio, t.combined_risk, t.amt_risk_score, t.is_high_spend, t.night_high_amt,
    ]])


def _build_features_b(t: TransactionB) -> np.ndarray:
    return np.array([[
        t.log_amt, t.city_pop,
        t.customer_num_trans_1_day, t.customer_num_trans_7_day, t.customer_num_trans_30_day,
        t.customer_avg_amout_1_day, t.customer_avg_amount_7_day, t.customer_avg_amount_30_day,
        t.merchant_num_trans_1_day, t.merchant_num_trans_7_day, t.merchant_num_trans_30_day,
        t.merchant_risk_1_day, t.merchant_risk_7_day, t.merchant_risk_30_day, t.merchant_risk_90_day,
        t.trans_time_hrs, t.trans_hour_sin, t.trans_hour_cos,
        t.trans_time_is_night, t.trans_date_is_weekend, t.trans_time_day,
        t.category_is_online, t.online_x_log_amt, t.velocity_ratio,
        t.gender_M,
        t.category_food_dining, t.category_gas_transport, t.category_grocery_net,
        t.category_grocery_pos, t.category_health_fitness, t.category_home,
        t.category_kids_pets, t.category_misc_net, t.category_misc_pos,
        t.category_personal_care, t.category_shopping_net, t.category_shopping_pos,
        t.category_travel,
    ]])


#  Endpoints 

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": default_model is not None,
        "gcs_bucket": GCS_BUCKET,
        "pipeline_a_models": list(GCS_MODELS_A.keys()),
        "pipeline_b_models": list(GCS_MODELS_B.keys()),
    }


@app.get("/models")
def list_models():
    return {
        "pipeline_a": list(GCS_MODELS_A.keys()),
        "pipeline_b": list(GCS_MODELS_B.keys()),
        "all": list(GCS_MODELS_A.keys()) + list(GCS_MODELS_B.keys()),
        "source": f"gs://{GCS_BUCKET}/{GCS_MODELS_PREFIX}/"
    }


@app.post("/predict")
def predict(transaction: TransactionA):
    """Default predict  uses lr_balanced (Pipeline A, 4-class)."""
    if default_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    features = _build_features_a(transaction)
    prediction = int(default_model.predict(features)[0])
    return {
        "model": "lr_balanced",
        "pipeline": "A",
        "prediction": prediction,
        "label": label_map[prediction],
        "risk_level": "HIGH" if prediction >= 2 else "LOW",
    }


@app.post("/predict/pipeline-a/{model_name}")
def predict_pipeline_a(model_name: str, transaction: TransactionA):
    """Pipeline A  4-class fraud risk (lr_balanced, lr_smote)."""
    if model_name not in GCS_MODELS_A:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not in Pipeline A. Available: {list(GCS_MODELS_A.keys())}"
        )
    m = _load_model(model_name, is_pipeline_b=False)
    features = _build_features_a(transaction)
    prediction = int(m.predict(features)[0])
    return {
        "model": model_name,
        "pipeline": "A",
        "prediction": prediction,
        "label": label_map[prediction],
        "risk_level": "HIGH" if prediction >= 2 else "LOW",
    }


@app.post("/predict/pipeline-b/{model_name}")
def predict_pipeline_b(model_name: str, transaction: TransactionB):
    """Pipeline B  binary fraud detection (lightgbm, xgboost, randomforest)."""
    if model_name not in GCS_MODELS_B:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not in Pipeline B. Available: {list(GCS_MODELS_B.keys())}"
        )
    m = _load_model(model_name, is_pipeline_b=True)
    features = _build_features_b(transaction)
    prediction = int(m.predict(features)[0])
    return {
        "model": model_name,
        "pipeline": "B",
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legit",
        "risk_level": "HIGH" if prediction == 1 else "LOW",
    }


@app.post("/predict/simple")
def predict_simple(
    amt: float,
    merchant_risk_30_day: float,
    trans_time_is_night: int = 0,
    avg_amt_per_customer: float = 50.0,
    gender_M: int = 1,
):
    """Simplified endpoint  key features only, others default to 0."""
    if default_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    amt_ratio = amt / (avg_amt_per_customer + 1)
    combined_risk = merchant_risk_30_day * 0.6
    amt_risk_score = amt_ratio * merchant_risk_30_day
    is_high_spend = 1 if amt > avg_amt_per_customer * 1.5 else 0
    night_high_amt = 1 if (trans_time_is_night == 1 and amt > 100) else 0

    features = np.zeros((1, 44))
    features[0, 2] = amt
    features[0, 9] = trans_time_is_night
    features[0, 19] = merchant_risk_30_day
    features[0, 20] = merchant_risk_30_day
    features[0, 23] = avg_amt_per_customer
    features[0, 25] = gender_M
    features[0, 39] = amt_ratio
    features[0, 40] = combined_risk
    features[0, 41] = amt_risk_score
    features[0, 42] = is_high_spend
    features[0, 43] = night_high_amt

    prediction = int(default_model.predict(features)[0])
    return {
        "model": "lr_balanced",
        "pipeline": "A",
        "prediction": prediction,
        "label": label_map[prediction],
        "risk_level": "HIGH" if prediction >= 2 else "LOW",
    }

@functions_framework.http
def fraud_predict(request):
    """GCP Cloud Functions entry point  wraps FastAPI app."""
    import json
    from starlette.testclient import TestClient
    
    client = TestClient(app)
    
    response = client.request(
        method=request.method,
        url=request.path + ("?" + request.query_string.decode() if request.query_string else ""),
        content=request.get_data(),
        headers=dict(request.headers),
    )
    
    return response.content, response.status_code, dict(response.headers)

#  GCP Cloud Run / Cloud Functions entry point 

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)