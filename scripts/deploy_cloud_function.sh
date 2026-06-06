
PROJECT_ID="project-ff2892f0-e40d-452f-aa0"
REGION="us-central1"
FUNCTION_NAME="fraud-detection-predict"
BUCKET_NAME="fraud-detection-mlops-data"

echo "Section 3.3: Deploy to Cloud Functions"

echo "Step 1: Enable required APIs"
gcloud services enable cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    --project=${PROJECT_ID}

echo "Step 2: Deploy FastAPI to Cloud Functions"
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=api/ \
    --entry-point=fraud_predict \
    --trigger-http \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=300s \
    --set-env-vars="GCS_BUCKET=${BUCKET_NAME},GCS_MODELS_PREFIX=models" \
    --project=${PROJECT_ID} \

echo "compelted"
echo "Function URL:"
gcloud functions describe ${FUNCTION_NAME} \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(serviceConfig.uri)"