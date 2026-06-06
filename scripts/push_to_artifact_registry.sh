

PROJECT_ID="project-ff2892f0-e40d-452f-aa0"
REGION="us-central1"
REPO_NAME="fraud-detection"
IMAGE_NAME="fraud-mlops"

echo "Section 3.1: GCP Artifact Registry"

echo "Step 1: Create Artifact Registry repository"
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Fraud detection ML pipeline Docker images" \
    --project=${PROJECT_ID}

echo "Step 2: Authenticate Docker to push to GCP"
gcloud auth configure-docker ${REGION}-docker.pkg.dev

echo "Step 3: Pull existing image from Docker Hub"
docker pull nshastry00/fraud-detection-mlops:latest

echo "Step 4: Tag image for Artifact Registry"
docker tag nshastry00/fraud-detection-mlops:latest \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest

echo "Step 5: Push to Artifact Registry"
docker push \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest

echo "completed"
echo "Image URI: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
echo "View at: https://console.cloud.google.com/artifacts"