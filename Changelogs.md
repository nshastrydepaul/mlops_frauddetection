# Changelog

All notable changes to this project are documented here.

---

## [Phase 3] 

-TODO

### Added — Israail (Section 1)
-TODO

### Added — Nishanth (Section 2)

* GitHub Actions CI pipeline with Ruff, mypy, pytest, and Codecov integration
* Multi-stage Docker containerization for reproducible training and deployment
* Docker Hub image publishing via automated GitHub Actions workflow
* Continuous Machine Learning (CML) workflow for automated training, reporting, and artifact generation
* DVC integration with Google Drive remote for dataset and model versioning in CI/CD
* Automated monitoring pipeline including drift detection, performance monitoring, and retraining workflows
* Prometheus-compatible monitoring metrics and system resource tracking
* Streamlit dashboard integration for monitoring and experiment visualization
* Dependency management refactoring with separate production and development requirements
* GitHub Actions workflows for CI validation, CML automation, Docker build/publish, and deployment verification
* End-to-end MLOps automation covering data versioning, model training, monitoring, reporting, containerization, and deployment workflows
* Documentation and workflow standardization for reproducible ML engineering practices

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