PROJECT_ID="project-ff2892f0-e40d-452f-aa0"
REGION="us-central1"
BUCKET_NAME="fraud-detection-mlops-data"
IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/fraud-detection/fraud-mlops:latest"

echo "Section 3.2: GCP Custom Training Job"

echo "Step 1: Create GCS bucket"
gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}

echo "Step 2: Upload data to GCS"
gcloud storage cp data/processed/x_train.csv gs://${BUCKET_NAME}/data/processed/
gcloud storage cp data/processed/x_test.csv gs://${BUCKET_NAME}/data/processed/
gcloud storage cp data/processed/y_train.csv gs://${BUCKET_NAME}/data/processed/
gcloud storage cp data/processed/y_test.csv gs://${BUCKET_NAME}/data/processed/

echo "Step 3: Upload models to GCS"
gcloud storage cp models/lr_balanced_20260606.joblib gs://${BUCKET_NAME}/models/
gcloud storage cp models/lr_smote_20260521.joblib gs://${BUCKET_NAME}/models/
gcloud storage cp models/lightgbm_20260606.joblib gs://${BUCKET_NAME}/models/
gcloud storage cp models/randomforest_20260606.joblib gs://${BUCKET_NAME}/models/
gcloud storage cp models/xgboost_20260606.joblib gs://${BUCKET_NAME}/models/
gcloud storage cp models/logisticregression_20260606.joblib gs://${BUCKET_NAME}/models/

echo "Step 4: Submit Vertex AI training job"
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="fraud-detection-training-$(date +%Y%m%d)" \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${IMAGE_URI} \
    --project=${PROJECT_ID}

echo "completed!"
echo "Monitor at: https://console.cloud.google.com/ai/platform/jobs"
echo "GCS bucket: gs://${BUCKET_NAME}"