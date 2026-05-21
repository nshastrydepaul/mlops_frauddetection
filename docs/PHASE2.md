# Phase 2: Model Development

## Overview
Phase 2 focused on operationalizing the Fraud Anomaly Classification & Behavioral Analytics platform through containerization, runtime validation, experiment reproducibility, logging improvements, and scalable machine learning workflows.

This phase expanded the Phase 1 machine learning pipelines into a reproducible MLOps environment capable of supporting:

* containerized preprocessing
* reproducible model training
* MLflow-compatible execution
* scalable ensemble experimentation
* structured logging and diagnostics
* artifact persistence

## Objectives

- [x] Implement baseline model
- [x] Train and evaluate initial models
- [ ] Hyperparameter tuning
- [x] Cross-validation and performance analysis
- [x] Model comparison and selection

## Deliverables

### 1. Model Implementation
- Model architecture defined
- Training pipeline implemented
- Evaluation metrics chosen
- Baseline performance established

Completed implementations include:

* Logistic Regression baseline
* SMOTE + Logistic Regression
* Random Forest
* LightGBM
* XGBoost

Additional completed functionality:

* reusable preprocessing pipeline
* engineered fraud behavior features
* train/test automation
* ensemble experimentation workflows
* reusable sklearn preprocessing utilities

### 2. Experiment Tracking
- All experiments logged and documented
- MLflow experiment tracking configured

Current experiment tracking support includes:

* MLflow-compatible execution workflows
* model artifact persistence
* runtime metric logging
* reproducible Docker execution
* metadata artifact generation

Tracked artifacts include:

* trained `.joblib` model files
* metadata JSON files
* confusion matrices
* evaluation metrics

### 3. Performance Analysis
- Model comparison results
- Hyperparameter sensitivity analysis
- Feature importance analysis
- Error analysis and patterns

Completed performance evaluation includes:

* F1-score evaluation
* ROC-AUC evaluation
* Average Precision metrics
* confusion matrix generation
* TimeSeriesSplit cross-validation
* model comparison across:

  * Logistic Regression
  * Random Forest
  * LightGBM
  * XGBoost

### 4. Model Artifacts
- Best model saved and versioned
- Model evaluation report
- Training curves and visualizations
- Configuration documentation

Generated artifacts include:

* versioned trained models
* metadata JSON files
* DVC-tracked model artifacts
* evaluation figures
* confusion matrices
* preprocessing artifacts

Artifacts are persisted through:

* Docker volume mounting
* DVC integration
* local artifact tracking

---

## Containerization & Runtime Validation

Phase 2 operationalized the project using Docker and Docker Compose.

### Docker Features

* Multi-stage Docker builds
* Python 3.11 reproducible runtime
* containerized preprocessing
* containerized model training
* MLflow-compatible execution
* artifact persistence support

### Docker Validation

The following workflows were successfully validated inside Docker:

* preprocessing pipeline
* feature engineering
* Logistic Regression training
* Random Forest training
* LightGBM training
* XGBoost training
* SMOTE workflows

Docker execution completed successfully with:

```text
exited with code 0
```

### Runtime Dependency Debugging

The following container/runtime issues were identified and resolved:

* missing make dependency
* missing curl dependency
* missing git dependency
* missing libgomp1 dependency for LightGBM

---

## Logging & Monitoring

Implemented logging features include:

* structured INFO/WARNING/ERROR logs
* runtime diagnostics
* preprocessing validation logs
* dataset shape logging
* fraud distribution logging
* traceback visibility using Rich logging

Validation checks include:

* NaN detection
* preprocessing validation
* train/test validation
* feature engineering validation

---

## Model Selection

### Chosen Models

| Model               | Purpose                               |
| ------------------- | ------------------------------------- |
| Logistic Regression | Baseline fraud classification         |
| Random Forest       | Ensemble benchmark                    |
| LightGBM            | Gradient boosting benchmark           |
| XGBoost             | High-performance fraud classification |

### Performance Metrics

Models were evaluated using:

* Accuracy
* F1-score
* ROC-AUC
* Average Precision
* Cross-validation metrics

---

### Chosen Model
- Model Type: 
- Best Hyperparameters: 
- Performance Metrics: 

## Key Results

* XGBoost achieved the strongest fraud detection performance
* Ensemble methods improved non-linear fraud detection capability
* Feature engineering significantly improved fraud representation
* Containerized execution successfully reproduced local training workflows
* Dockerized workflows improved reproducibility and operational consistency

## Challenges and Solutions

| Challenge                              | Resolution                           |
| -------------------------------------- | ------------------------------------ |
| Docker dependency failures             | Added required native Linux packages |
| LightGBM runtime errors                | Installed libgomp1 runtime support   |
| Container reproducibility issues       | Added Docker Compose orchestration   |
| Artifact persistence inside containers | Added persistent volume mounting     |
| Dependency inconsistencies             | Standardized Python 3.11 runtime     |


## Next Steps

Move to Phase 3 once model is selected and meets performance requirements.

Phase 3 will focus on:

* CI/CD integration and automated deployment pipelines
* API and inference server deployment
* production readiness validation
* monitoring and maintenance workflows
* cloud deployment strategies
* scalable inference and serving pipelines
* deployment documentation and operational runbooks

## Status

- Start Date: 
- Estimated Completion: 
- Actual Completion: In Progress
- Status: Operationalization Completed
