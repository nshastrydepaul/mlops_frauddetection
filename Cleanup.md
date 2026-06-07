# GCP Resource Cleanup Guide

## Overview
This document shows the necessary steps required to delete all GCP resources that have been created for Phase 3 to ensure that no extra charges are incurred after grading.


---

## Cleanup Commands

### Step 1  Delete Cloud Functions
```bash
gcloud functions delete fraud-detection-predict \
    --region=us-central1 \
    --project=project-ff2892f0-e40d-452f-aa0
```

### Step 2  Delete Cloud Run service
```bash
gcloud run services delete fraud-detection-predict \
    --region=us-central1 \
    --project=project-ff2892f0-e40d-452f-aa0
```

### Step 3  Delete GCS bucket
```bash
# Delete all contents first
gcloud storage rm -r gs://fraud-detection-mlops-data

# Verify bucket is empty
gcloud storage ls gs://fraud-detection-mlops-data
```

### Step 4  Delete Artifact Registry
```bash
gcloud artifacts repositories delete fraud-detection \
    --location=us-central1 \
    --project=project-ff2892f0-e40d-452f-aa0
```

### Step 5  Delete Service Account
```bash
gcloud iam service-accounts delete \
    github-actions-sa@project-ff2892f0-e40d-452f-aa0.iam.gserviceaccount.com \
    --project=project-ff2892f0-e40d-452f-aa0
```

### Step 6  Disable APIs
```bash
gcloud services disable \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    --project=project-ff2892f0-e40d-452f-aa0
```

### Step 7  Verify cleanup
```bash
# Check no functions running
gcloud functions list --project=project-ff2892f0-e40d-452f-aa0

# Check no buckets remaining
gcloud storage ls

# Check no Artifact Registry repos
gcloud artifacts repositories list --location=us-central1

# Check Vertex AI jobs
gcloud ai custom-jobs list --region=us-central1
```

---

## Cost Monitoring

Set up a budget alert to avoid surprise charges:

```bash
# View current billing
gcloud billing accounts list

# Check current spend
gcloud billing budgets list \
    --billing-account=01EE2A-B2A5DF-8FB351
```

Or go to: https://console.cloud.google.com/billing

---

##  Important Notes

- Keep resources active until assignment is graded
- Free tier limits: Cloud Functions (2M requests/month), Cloud Run (2M requests/month)
- GCS storage: first 5GB free in us-central1
- Vertex AI training jobs: billed per minute  always check job status
- Delete resources in order above to avoid dependency errors

---
