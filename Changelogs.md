# Changelog

All notable changes to this project are documented here.

---

## [Phase 3] 

-TODO

### Added — Israail (Section 1)
-TODO

### Added — Nishanth (Section 2)
### Section 2 Changes

### 2.1 CI/CD Automation
* Configured GitHub Actions CI workflow
* Added Ruff linting and formatting checks
* Added mypy static type checking
* Added pytest test execution with coverage reporting
* Added Codecov integration for coverage uploads
* Simplified CI execution to Python 3.11

### 2.2 CML Automation
* Configured Continuous Machine Learning (CML) workflow
* Automated DVC data retrieval from Google Drive remote
* Automated model training execution in GitHub Actions
* Automated experiment reporting generation
* Added artifact upload for reports and monitoring outputs

### 2.3 Dockerization
* Implemented multi-stage Docker build for optimized image size
* Added Docker Hub image build and push workflow
* Added automated Docker image validation in CI
* Configured Docker Buildx and image metadata generation
* Added support for tagged image releases

### 2.4 Dependency Management
* Refactored requirements.txt into production dependencies
* Created requirements_dev.txt for development tooling
* Removed duplicate and conflicting package definitions
* Fixed Python version compatibility issues in CI

### 2.5 MLOps Infrastructure
* Integrated DVC with GitHub Actions workflows
* Automated model artifact management
* Added workflow support for reproducible training runs
* Improved repository automation and deployment readiness

One pull request as evidence:

* Full PR description: [reports/pr_nishanth_phase3.txt](reports/pr_nishanth_phase3.txt)

### Added — Musaddiq (Section 3)
- FastAPI service with 6 model endpoints (Pipeline A + B)
- GCP Artifact Registry repository `fraud-detection`
- Vertex AI custom training job (Job ID: 7785243246136918016)
- GCS bucket `fraud-detection-mlops-data` with data + models
- Cloud Functions deployment (live URL: https://fraud-detection-predict-lj4hg275tq-uc.a.run.app)
- Cloud Run service via Cloud Functions Gen2
- GCS model loading in `api/main.py`
- `scripts/push_to_artifact_registry.sh`
- `scripts/gcp_train.sh`
- `scripts/deploy_cloud_function.sh`
- DVC pointers for all 6 retrained models

One pull request as evidence: - Full PR description: [reports/pr_musaddiq_phase3.txt](reports/pr_musaddiq_phase3.txt)

### Added — Lohith (Section 4)
-TODO

---

## [Phase 2] — 2026-05-21

### Added
- MLflow experiment tracking (4 experiments)
- Scalene + cProfile profiling
- Hydra configuration management
- Modular ML pipeline refactoring
- Pipeline A: LR balanced + SMOTE 
- Pipeline B: LightGBM, XGBoost, RandomForest 
- Docker containerization 
- Monitoring with psutil/MLflow 

---

## [Phase 1] — 2026-05-07

### Added
- Initial project setup
- Data preprocessing pipeline
- Baseline models
- DVC data versioning
- Basic logging with Rich
