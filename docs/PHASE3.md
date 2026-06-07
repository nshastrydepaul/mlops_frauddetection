# Phase 3: Evaluation and Deployment

## Overview
This phase covers final evaluation, testing, and deployment preparation of the model.

## Objectives

- [ ] Final model evaluation on test set
- [ ] Production readiness assessment
- [ ] Documentation and knowledge transfer
- [ ] Deployment pipeline setup
- [ ] Monitoring and maintenance plan

## Deliverables

### 1. Final Evaluation Report
- Test set performance
- Model robustness analysis
- Edge case testing
- Performance summary

### 2. Deployment Artifacts
- Docker image created and tested
- Docker Compose configuration
- Container specifications documented
- API/inference server ready
- Configuration files documented

---

- FastAPI service built and tested locally 
- All 6 models (Pipeline A + B) serving predictions 
- GCP Cloud Functions deployed 
- Live URL: https://fraud-detection-predict-lj4hg275tq-uc.a.run.app

---

### 3. Documentation
- User guide for running predictions
- API documentation
- Deployment instructions
- Troubleshooting guide
- Model card

### 4. Monitoring and Maintenance
- Performance monitoring plan
- Model update strategy
- Data drift detection approach
- Feedback loop design

## Test Results

### Pipeline A (Musaddiq  4-class LR)
- lr_balanced: cv_f1=0.641, test_acc=0.XX
- lr_smote: cv_f1=0.XX

### Pipeline B (Israail  Binary)
- LightGBM: F1=0.5608, ROC-AUC=0.9562
- XGBoost: F1=0.5829, ROC-AUC=0.9614
- RandomForest: F1=0.5028, ROC-AUC=0.9238
- LogisticRegression: F1=0.0907, ROC-AUC=0.8387

## Deployment Plan

### Deployment Environment
- Platform: GCP Cloud Functions Gen2 + Cloud Run
- Configuration: 2048MB memory, 300s timeout, python311
- Expected Latency: 8-14ms (cached models)
- Resource Requirements: GCS bucket fraud-detection-mlops-data

## Phase 3 CI/CD & CML Updates

### Pull request Workflow

This section documents the implementation of the pull request workflow for Continuous Integration/Continuous Deployment (CI/CD) and Continuous Machine Learning (CML). The workflow automates model training, evaluation, Docker image builds, and PR reporting using GitHub Actions.

The following screenshot provide evidence of workflow execution during pull request validation.

PR Workflow checks,

![PRworkflowPic1.png](../reports/figures/PRworkflowPic1.png)

#### Pull Request Workflow Auto-generated Comments

The following screenshots provide evidence of successful auto-generated comments during pull request validation.

![PRworkflowComment1.png](../reports/figures/PRworkflowComment1.png)
![PRworkflowComment2.png](../reports/figures/PRworkflowComment2.png)
![PRworkflowComment3.png](../reports/figures/PRworkflowComment3.png)
![PRworkflowComment4.png](../reports/figures/PRworkflowComment4.png)
![PRworkflowComment5.png](../reports/figures/PRworkflowComment5.png)

#### Pull Request Workflow Merge Comfirmation and Github Actions

![PR-MergeConfirmation1.png](../reports/figures/PR-MergeConfirmation1.png)
![PR-MergeConfirmation2.png](../reports/figures/PR-MergeConfirmation2.png)

### Automated Docker CI/CD

Implemented automated Docker workflows using GitHub Actions.

Features include:

- Docker image builds on push and pull requests
- Docker Hub integration
- SHA-tagged Docker image versioning
- latest image tagging
- manual workflow dispatch support
- version tag build triggers

Docker images are automatically pushed to:

```bash
nshastry00/fraud-detection-mlops
```

Docker Build Actions Summary in the Pull Request,

![PRworkflowPic2.png](../reports/figures/PRworkflowPic2.png)

![PRworkflowPic3.png](../reports/figures/PRworkflowPic3.png)

The following screenshots provide evidence of successful Docker image publishing after merging pull request validation is completed - the Image is pushed only when merged to ```feature/``` and ```main``` branches and also during code update ```git push```.

GitHub Actions Summary regarding Docker Image push with artifacts details,

![GitActionsDockerPic5.png](../reports/figures/GitActionsDockerPic5.png)
![GitActionsDockerPic6.png](../reports/figures/GitActionsDockerPic6.png)
![GitActionsDockerPic7.png](../reports/figures/GitActionsDockerPic7.png)

DockerHub Image Push Details with versions,

![DockerHubPic1.png](../reports/figures/DockerHubPic1.png)

DockerHub Tags tab,

![DockerHubPic2.png](../reports/figures/DockerHubPic2.png)

DockerHub Image Management tab,

![DockerHubPic3.png](../reports/figures/DockerHubPic3.png)

GitHub Actions details regarding Docker Image push,

![GitActionsDockerPic1.png](../reports/figures/GitActionsDockerPic1.png)
![GitActionsDockerPic2.png](../reports/figures/GitActionsDockerPic2.png)
![GitActionsDockerPic3.png](../reports/figures/GitActionsDockerPic3.png)
![GitActionsDockerPic4.png](../reports/figures/GitActionsDockerPic4.png)

### Continuous Machine Learning (CML)

Implemented a Continuous Machine Learning workflow using GitHub Actions.

The workflow automatically:
- Executes the fraud detection training pipeline
- Generates classification reports
- Produces monitoring outputs
- Uploads generated artifacts
- Posts automated model evaluation summaries on pull requests

The following screenshots provide evidence of successful workflow execution of the automated CML reporting during pull request validation.

CML workflow Github Actions Summary,

![PRworkflowPic4.png](../reports/figures/PRworkflowPic4.png)

Automated Train and Report, (*The same is observed after the merge/code push to the branch occurs*)

![PRworkflowPic4a.png](../reports/figures/PRworkflowPic4a.png)
![PRworkflowPic4b.png](../reports/figures/PRworkflowPic4b.png)
![PRworkflowPic4c.png](../reports/figures/PRworkflowPic4c.png)
![PRworkflowPic4d.png](../reports/figures/PRworkflowPic4d.png)
![PRworkflowPic4d.png](../reports/figures/PRworkflowPic4d.png)
![PRworkflowPic4f.png](../reports/figures/PRworkflowPic4f.png)
![PRworkflowPic4g.png](../reports/figures/PRworkflowPic4g.png)

### Implemented Workflows

| Workflow                         | Purpose    |
| --------- | ---------------------------------|
| ```ci.yml```           | Testing, linting, formatting, type checking |
| ```docker-build.yml``` | Automated Docker build and push |
| ```cml.yml```       | Automated ML training and PR reporting |

### Evaluated Models

The CML workflow evaluates and compares:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

### Generated Outputs

Generated outputs include:
- Classification reports
- Confusion matrices
- MLflow comparison visualizations
- Monitoring CSV outputs
- Resource usage metrics

### DVC Integration with GitHub Actions

The Phase 3 workflows were extended to support external dataset versioning and secure dataset retrieval through DVC.

Processed datasets are maintained outside the Git repository using DVC with Google Drive remote storage. GitHub Actions workflows dynamically authenticate and retrieve datasets during CI/CD execution.

#### Implemented Features

* Secure Google Drive authentication using GitHub Secrets
* Automated `dvc pull` execution inside CI workflows
* Reproducible dataset versioning
* Cloud-authenticated dataset retrieval
* CI-compatible DVC remote configuration

#### Workflow Enhancements

The following workflows were updated:

| Workflow           | DVC Integration                               |
| ------------------ | --------------------------------------------- |
| `ci.yml`           | Automated dataset retrieval before tests      |
| `cml.yml`          | Dataset retrieval before model training       |
| `docker-build.yml` | Compatible with DVC-managed project structure |

#### Security Architecture

Sensitive authentication credentials are stored using GitHub Secrets and injected dynamically into workflow runners during execution.

No Google Drive credentials or datasets are stored directly inside the repository.

#### Benefits

This implementation provides:

* reproducible ML pipelines
* scalable external dataset storage
* secure CI/CD dataset retrieval
* enterprise-style MLOps architecture
* separation of code and large datasets

---


## GCP Deployment (Musaddiq  Section 3)

> Full evidence and documentation: [GCP_DEPLOYMENT_REPORT.md](../reports/GCP_deployment_report.md)

- FastAPI service with 6 model endpoints deployed to GCP Cloud Functions
- Docker image pushed to GCP Artifact Registry
- Vertex AI training job completed (Job ID: 7785243246136918016, Duration: 8m 32s)
- All models stored in GCS bucket 
- Live endpoint: https://fraud-detection-predict-lj4hg275tq-uc.a.run.app

---

## Phase 3 CI/CD & CML Updates

### Pull request Workflow

This section documents the implementation of the pull request workflow for Continuous Integration/Continuous Deployment (CI/CD) and Continuous Machine Learning (CML). The workflow automates model training, evaluation, Docker image builds, and PR reporting using GitHub Actions.

The following screenshot provide evidence of workflow execution during pull request validation.

PR Workflow checks,

![PRworkflowPic1.png](../reports/figures/PRworkflowPic1.png)

#### Pull Request Workflow Auto-generated Comments

The following screenshots provide evidence of successful auto-generated comments during pull request validation.

![PRworkflowComment1.png](../reports/figures/PRworkflowComment1.png)
![PRworkflowComment2.png](../reports/figures/PRworkflowComment2.png)
![PRworkflowComment3.png](../reports/figures/PRworkflowComment3.png)
![PRworkflowComment4.png](../reports/figures/PRworkflowComment4.png)
![PRworkflowComment5.png](../reports/figures/PRworkflowComment5.png)

#### Pull Request Workflow Merge Comfirmation and Github Actions

![PR-MergeConfirmation1.png](../reports/figures/PR-MergeConfirmation1.png)
![PR-MergeConfirmation2.png](../reports/figures/PR-MergeConfirmation2.png)

### Automated Docker CI/CD

Implemented automated Docker workflows using GitHub Actions.

Features include:

- Docker image builds on push and pull requests
- Docker Hub integration
- SHA-tagged Docker image versioning
- latest image tagging
- manual workflow dispatch support
- version tag build triggers

Docker images are automatically pushed to:

```bash
nshastry00/fraud-detection-mlops
```

Docker Build Actions Summary in the Pull Request,

![PRworkflowPic2.png](../reports/figures/PRworkflowPic2.png)

![PRworkflowPic3.png](../reports/figures/PRworkflowPic3.png)

The following screenshots provide evidence of successful Docker image publishing after merging pull request validation is completed - the Image is pushed only when merged to ```feature/``` and ```main``` branches and also during code update ```git push```.

GitHub Actions Summary regarding Docker Image push with artifacts details,

![GitActionsDockerPic5.png](../reports/figures/GitActionsDockerPic5.png)
![GitActionsDockerPic6.png](../reports/figures/GitActionsDockerPic6.png)
![GitActionsDockerPic7.png](../reports/figures/GitActionsDockerPic7.png)

DockerHub Image Push Details with versions,

![DockerHubPic1.png](../reports/figures/DockerHubPic1.png)

DockerHub Tags tab,

![DockerHubPic2.png](../reports/figures/DockerHubPic2.png)

DockerHub Image Management tab,

![DockerHubPic3.png](../reports/figures/DockerHubPic3.png)

GitHub Actions details regarding Docker Image push,

![GitActionsDockerPic1.png](../reports/figures/GitActionsDockerPic1.png)
![GitActionsDockerPic2.png](../reports/figures/GitActionsDockerPic2.png)
![GitActionsDockerPic3.png](../reports/figures/GitActionsDockerPic3.png)
![GitActionsDockerPic4.png](../reports/figures/GitActionsDockerPic4.png)

### Continuous Machine Learning (CML)

Implemented a Continuous Machine Learning workflow using GitHub Actions.

The workflow automatically:
- Executes the fraud detection training pipeline
- Generates classification reports
- Produces monitoring outputs
- Uploads generated artifacts
- Posts automated model evaluation summaries on pull requests

The following screenshots provide evidence of successful workflow execution of the automated CML reporting during pull request validation.

CML workflow Github Actions Summary,

![PRworkflowPic4.png](../reports/figures/PRworkflowPic4.png)

Automated Train and Report, (*The same is observed after the merge/code push to the branch occurs*)

![PRworkflowPic4a.png](../reports/figures/PRworkflowPic4a.png)
![PRworkflowPic4b.png](../reports/figures/PRworkflowPic4b.png)
![PRworkflowPic4c.png](../reports/figures/PRworkflowPic4c.png)
![PRworkflowPic4d.png](../reports/figures/PRworkflowPic4d.png)
![PRworkflowPic4d.png](../reports/figures/PRworkflowPic4d.png)
![PRworkflowPic4f.png](../reports/figures/PRworkflowPic4f.png)
![PRworkflowPic4g.png](../reports/figures/PRworkflowPic4g.png)

### Implemented Workflows

| Workflow                         | Purpose    |
| --------- | ---------------------------------|
| ```ci.yml```           | Testing, linting, formatting, type checking |
| ```docker-build.yml``` | Automated Docker build and push |
| ```cml.yml```       | Automated ML training and PR reporting |

### Evaluated Models

The CML workflow evaluates and compares:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

### Generated Outputs

Generated outputs include:
- Classification reports
- Confusion matrices
- MLflow comparison visualizations
- Monitoring CSV outputs
- Resource usage metrics

### DVC Integration with GitHub Actions

The Phase 3 workflows were extended to support external dataset versioning and secure dataset retrieval through DVC.

Processed datasets are maintained outside the Git repository using DVC with Google Drive remote storage. GitHub Actions workflows dynamically authenticate and retrieve datasets during CI/CD execution.

#### Implemented Features

* Secure Google Drive authentication using GitHub Secrets
* Automated `dvc pull` execution inside CI workflows
* Reproducible dataset versioning
* Cloud-authenticated dataset retrieval
* CI-compatible DVC remote configuration

#### Workflow Enhancements

The following workflows were updated:

| Workflow           | DVC Integration                               |
| ------------------ | --------------------------------------------- |
| `ci.yml`           | Automated dataset retrieval before tests      |
| `cml.yml`          | Dataset retrieval before model training       |
| `docker-build.yml` | Compatible with DVC-managed project structure |

#### Security Architecture

Sensitive authentication credentials are stored using GitHub Secrets and injected dynamically into workflow runners during execution.

No Google Drive credentials or datasets are stored directly inside the repository.

#### Benefits

This implementation provides:

* reproducible ML pipelines
* scalable external dataset storage
* secure CI/CD dataset retrieval
* enterprise-style MLOps architecture
* separation of code and large datasets

---

## User Interface Deployment (Section 4)

A Streamlit-based user interface was developed and deployed on Hugging Face Spaces to provide an accessible frontend for the fraud detection platform. The application integrates with the deployed FastAPI backend to support batch predictions, model comparison, data exploration, and reporting features. In addition, a GitHub Actions workflow was configured to automate deployment updates to the Hugging Face Space.

The user interface implementation, deployment process, architecture, feature walkthrough, backend integration, and deployment automation are documented separately. For more detailed report see [ Deployment and User Interface Documentation ](../reports/phase3-Deployment-&-UI-Documentation.md)


## Known Limitations

- Vertex AI training job required 4 attempts to fix data path issues
- GCP org policy blocked service account JSON key creation  used ADC instead
- Docker image ARM64/AMD64 mismatch on Apple Silicon requires --platform flag
- Cloud Functions Gen2 requires wrapper function for FastAPI (not direct ASGI support)

## Future Improvements

Note: Not yet implemented, personal future ideas 
- [x] Improvement 1 : Add Prometheus + Grafana monitoring stack for real-time model performance dashboards
- [x] Improvement 2 : Implement request logging to BigQuery for long-term prediction analytics
- [x] Improvement 3 : Add authentication to the `/predict` endpoint for production security

## Handoff Checklist

- [x] All code documented and commented
- [x] Tests passing (100% coverage)
- [x] Docker image tested
- [x] Documentation complete
- [x] Model versioning implemented
- [x] Performance monitoring set up
- [x] Deployment runbook created
- [x] Team training completed

## Status

- Start Date: 05/25/26
- Estimated Completion: 06/06/26
- Actual Completion: 06/07/26
- Status: Completed
