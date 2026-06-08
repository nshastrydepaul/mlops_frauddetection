# PHASE 3: Continuous Machine Learning (CML) & Deployment

## Overview
Phase 3 implements continuous integration/continuous deployment (CI/CD) pipelines and productionizes Fraud-Anomoly Detection & Behavioral Analytics on cloud infrastructure. This phase covers automated testing, containerized workflows, CML integration, and multi-platform deployment options including GCP, Cloud Run, and serverless functions.

---

## 1. Continuous Integration & Testing

- [x] **Unit Tests**: Write pytest test scripts for data processing and model components
- [x] **Integration Tests**: Create integration tests for full training pipeline
- [x] **Test Coverage**: Aim for >80% code coverage with pytest-cov
- [x] **GitHub Actions - Tests**: Create workflow for running tests on every push
  - [x] Trigger on: push to main/develop branches and PRs
  - [x] Test across multiple Python versions if applicable
  - [x] Report coverage metrics
- [x] **GitHub Actions - Code Quality**: Create workflow for:
  - [x] Running ruff linter
  - [x] Type checking with mypy
  - [x] Formatting checks
- [x] **GitHub Actions - Docker Build**: Create workflow for building Docker image
  - [x] Build on PR and main branch push
  - [x] Test built image
- [x] **Pre-commit Hooks**: Set up pre-commit hooks for:
  - [x] Formatting (black/ruff)
  - [x] Linting
  - [x] Type checking
  - [x] Trailing whitespace
- [x] **Test Documentation**: Document how to run tests locally and in CI

---

## 2. Continuous Docker Building & CML

- [x] **Automated Docker Builds**: Configure Docker build pipeline triggered by:
  - [x] Commits to main branch
  - [x] Version tags
  - [x] Manual workflow dispatch
- [x] **Docker Push**: Implement push to container registry (Docker Hub, GitHub Container Registry, or GCP)
- [x] **CML Initialization**: Initialize CML in repository
- [x] **CML Workflow**: Create GitHub Actions workflow for CML that:
  - [x] Trains model on workflow runner
  - [x] Generates performance metrics
  - [x] Creates visualizations/plots
  - [x] Comments results on PR
- [x] **CML Metrics Output**: Document format and sample output of CML metrics
- [x] **CML Plots**: Generate sample plots and document in CML workflow
- [x] **Model Comparison**: Create CML output showing comparison of current vs. baseline model
- [x] **Workflow Documentation**: Document CML workflow setup and customization

---

### Implemented Workflows

| Workflow File | Description |
|---|---|
| `.github/workflows/docker-build.yml` | Builds and pushes Docker images automatically |
| `.github/workflows/cml.yml` | Runs model training and posts PR evaluation reports |
| `.github/workflows/ci.yml` | Linting, formatting, testing, and type checking |

---

### Docker CI/CD Features

Implemented automated Docker workflows with:

- Docker Buildx
- Docker Hub authentication
- SHA-based image tagging
- latest image tagging
- pull request image validation
- workflow_dispatch support
- version tag build support

Docker images are automatically published to:

```bash
nshastry00/fraud-detection-mlops
```

### Continuous Machine Learning (CML)

Implemented a GitHub Actions based CML workflow that:

- Runs fraud detection model training automatically
- Generates classification reports
- Produces evaluation metrics
- Uploads generated artifacts
- Generates monitoring outputs
- Posts automated evaluation summaries on pull requests

### DVC Remote Integration & Secure Dataset Retrieval

Phase 3 additionally integrated secure DVC-based dataset retrieval inside GitHub Actions workflows using Google Drive as the remote storage backend.

Since large processed datasets are not stored directly inside GitHub, CI/CD runners dynamically retrieve versioned datasets using DVC before executing training and evaluation workflows.

The GitHub Actions workflows now automatically:

* authenticate securely with Google Drive using GitHub Secrets
* configure DVC remotes dynamically during runtime
* execute `dvc pull` to retrieve processed datasets
* ensure reproducible training and evaluation pipelines inside CI/CD

### Secure Secret Management

Sensitive credentials are securely managed through GitHub Actions Secrets, including:

* `DOCKER_USERNAME`
* `DOCKER_PASSWORD`
* `GDRIVE_CREDENTIALS`

This architecture enables secure cloud-authenticated dataset retrieval without exposing credentials directly in the repository.

### DVC Workflow Integration

The CI and CML workflows now follow the sequence:

```text
Install Dependencies
- Configure Google Drive Credentials
- Configure DVC Authentication
- Pull Versioned Dataset via DVC
- Execute Training / Testing Pipelines
```

This establishes a reproducible enterprise-style MLOps workflow combining:

* GitHub Actions
* DVC
* Docker
* Google Drive remote storage
* Continuous Machine Learning (CML)
* automated model evaluation

### Model Evaluation Outputs

#### The automated CML workflow evaluates and compares:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

#### Generated outputs include:

- Precision
- Recall
- F1-score
- Confusion matrices
- MLflow comparison visualizations
- Resource usage monitoring

### Generated Artifacts

Artifacts uploaded from GitHub Actions include:

- report.md
- reports/figures/
- reports/monitoring/

### Evidence

#### Docker Workflow

- GitHub Actions successful workflow runs
- Docker Hub image repository screenshots
- Docker SHA tag generation
- Automated Docker push logs

#### CML Workflow
- Pull request evaluation comments
- Generated classification reports
- Monitoring outputs
- Uploaded workflow artifacts

---

## 3. Deployment on GCP

> All detailed explanation, steps, screenshots, and evidence for Section 3 (FastAPI, GCP Artifact Registry, Vertex AI Training Job, Cloud Functions, and Cloud Run deployment) are documented in the full deployment report referenced below.

 **Full GCP deployment evidence:** [docs/GCP_DEPLOYMENT_REPORT.md](/reports/GCP_DEPLOYMENT_REPORT.md)

- [x] **GCP Project Setup**: Create GCP project and enable necessary APIs
- [x] **Service Account**: Create service account with appropriate permissions for:
  - [x] Artifact Registry
  - [x] Vertex AI
  - [x] Cloud Run
  - [x] Cloud Functions
  - [ ] Compute Engine (not used)
- [x] **Artifact Registry**: Set up Artifact Registry for storing Docker images
  - [x] Create repository in Artifact Registry
  - [x] Configure authentication from CI/CD
  - [x] Push Docker images to registry
- [x] **Vertex AI Training (Option A)**: Set up custom training on Vertex AI
  - [x] Create training container image
  - [x] Configure training job specification
  - [x] Document how to submit training jobs
- [ ] **Compute Engine Training (Option B)**: Set up training on Compute Engine instance (not used)
  - [ ] Create VM instance with GPU if needed
  - [ ] Document SSH access and training process
  - [ ] Set up instance for automated training
- [x] **Model Registry**: Store trained models in GCS bucket with versioning
  - [x] Create GCS bucket for models
  - [x] Implement model upload from training
  - [x] Document model retrieval process
- [x] **FastAPI Service**: Create FastAPI application for model serving
  - [x] Define inference endpoint(s)
  - [x] Implement request validation
  - [x] Add health check endpoint
  - [x] Document API specification
- [x] **Cloud Functions Deployment (Option A)**: Deploy inference as Cloud Function
  - [x] Package model and FastAPI app for Cloud Functions
  - [x] Create Cloud Function with appropriate memory/timeout
  - [x] Configure HTTP trigger
  - [x] Document invocation and response format
- [x] **Cloud Run Deployment (Option B)**: Deploy as containerized service on Cloud Run: (partially done for learning purpose | actual run is based on Option A & C)
  - [ ] Create Dockerfile optimized for Cloud Run
  - [] Test locally with Cloud Run emulator
  - [x] Deploy to Cloud Run with auto-scaling
  - [x] Document deployment process
- [x] **Streamlit/Gradio Deployment (Option C)**: Deploy demo app on HuggingFace Spaces
  - [x] Create Streamlit or Gradio interface for model
  - [x] Push to GitHub repository
  - [x] Deploy to HuggingFace Spaces
  - [x] Document feature walkthrough
- [x] **Load Testing**: Test deployment with load testing tool (locust) 
  - [x] Establish baseline performance metrics
  - [x] Document scaling characteristics
- [x] **Monitoring Setup**: Configure Cloud Monitoring and Cloud Logging
  - [x] Set up log aggregation
  - [x] Create monitoring dashboards
  - [x] Set up alerts for anomalies

  Testing & Monitoring explanation & evidence exist in : [docs/GCP_DEPLOYMENT_REPORT.md](/reports/GCP_DEPLOYMENT_REPORT.md)

  Full Testing log : [Test_result](/reports/load_test_results.txt)

---

## 4. Documentation & Repository Updates

- [x] **Comprehensive README**: Update README with:
  - [x] Architecture diagram showing all components (*Updated exisitng architecture with all components relevant to PHASE3*)
  - [x] CI/CD pipeline overview
  - [x] Deployment instructions for each option (Cloud Run, Cloud Functions, HuggingFace)
  - [x] GCP setup and configuration guide
  - [x] How to invoke deployed models
  - [x] Monitoring and troubleshooting guide
  - [x] Cost estimation and optimization tips
- [x] **Deployment Guide**: Create detailed DEPLOYMENT.md [docs/GCP_DEPLOYMENT_REPORT.md](/reports/GCP_DEPLOYMENT_REPORT.md)  with:
  - [x] Step-by-step GCP setup instructions
  - [x] Cloud Run deployment procedure
  - [x] Cloud Functions configuration
  - [x] Environment variables and secrets management
  - [x] Rollback procedures
- [x] **API Documentation**: Document all endpoints with:
  - [x] Request/response schemas
  - [x] Example curl/Python requests
  - [x] Error codes and messages  [troubleshooting.md](troubleshooting.md) 
- [x] **Architecture Documentation**: Include diagrams showing:
  - [x] Data pipeline
  - [x] Training pipeline
  - [x] Inference/serving architecture
  - [x] CI/CD workflow
- [x] **Screenshots/Demos**: Add:
  - [x] Cloud Run dashboard screenshot
  - [x] Monitoring dashboard screenshot
  - [x] Streamlit/Gradio app screenshot
  - [x] API response example
  - [x] CML workflow output sample
- [x] **Troubleshooting Guide**: Document solutions for:   [troubleshooting.md](troubleshooting.md) 
  - [x] Common deployment errors
  - [x] Authentication issues
  - [x] Performance problems
  - [x] Cost overruns
- [x] **Resource Cleanup Reminder**: Create CLEANUP.md with instructions for: [Cleanup.md](Cleanup.md)
  - [x] Deleting GCP resources (VMs, databases, etc.)
  - [x] Cleaning up Cloud Storage buckets
  - [x] Disabling APIs to avoid charges
  - [x] Cost monitoring recommendations
- [x] **Contributing Guide Update**: Update CONTRIBUTING.md with:
  - [x] CI/CD requirements
  - [x] Testing requirements for PRs
  - [x] Deployment process documentation
- [x] **Changelog**: Maintain CHANGELOG.md documenting releases and deployments. [Changelog.md](Changelogs.md) 

---

| File | Description |
|---|---|
| [GCP_deployment_report.md](reports/GCP_DEPLOYMENT_REPORT.md) | Full GCP deployment evidence  Section 3 |
| [Cleanup.md](CLEANUP.md) | GCP resource cleanup instructions |
| [Contribute.md](CONTRIBUTE.md) | CI/CD requirements, testing, deployment process |
| [Changelog.md](CHANGELOG.md) | Release history and deployment log |
| [reports/pr_musaddiq_phase3.txt](reports/pr_musaddiq_phase3.txt) | Phase 3 PR description |

--- 

## Team Contributions  Phase 3

| Team Member | Section | Responsibilities |
|---|---|---|
| Israail Ghazzal | Section 1 | pytest unit/integration tests, GitHub Actions CI workflow, pre-commit hooks, test coverage |
| Nishanth Shastry | Section 2 | Automated Docker builds, CML workflow, Docker Hub CI/CD, DVC remote integration |
| Musaddiq Vavartar | Section 3 | FastAPI service, GCP Artifact Registry, Vertex AI training job, Cloud Functions deployment, Cloud Run |
| Lohith Poola | Section 4 | Interactive UI, README updates, API documentation, architecture documentation, deployment guide |
| All Members | Section 5 | load testing, monitoring setup |
| All Members | Section 6 | PHASE3.md documentation, GCP resource cleanup, CLEANUP.md, CONTRIBUTING.md |

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables.

