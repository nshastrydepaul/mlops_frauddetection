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

## Monitoring & Debugging

Implemented monitoring and debugging features include:

* psutil-based CPU and RAM monitoring
* optional GPU monitoring using GPUtil
* CSV-based resource usage logging during training
* structured INFO/WARNING/ERROR logs
* runtime diagnostics
* dataset shape logging
* fraud class distribution logging
* assertion-based validation checks
* pdb line-by-line debugging support
* For detailed Monitoring and Debugging, see the [Monitoring & Debug Report](reports/phase2-monitoring-debugging.md)

Validation checks include:

* NaN detection
* preprocessing validation
* train/test validation
* feature engineering validation

---

## Profiling & Optimization (Section 3.2)

### Framework Profiling  Scalene

**Tool:** Scalene v1.5.19  line-level CPU + memory profiler for scikit-learn users

**How we ran it:**
```bash
scalene run --memory --output profile.html scripts/profile_training.py
```

**Scalene HTML output:** `profile.html`

**Key findings:**

| File | % of Time | Time (s) |
|---|---|---|
| train_model.py | 87.9% | 12.281s |
| system_monitoring.py | 12.1% | 1.687s |

**Top functions by memory copies:**

| Function | Line | Memory Copies |
|---|---|---|
| train_lr_pipeline | 315 | 115 |
| load_data | 124 | 7 |
| _clean_data_lr | 152 | 5 |

**Optimization applied:**
```python
# cross_val_score parallelized across CPU cores
cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
```

**cProfile output:** `reports/profiling_output.txt` (22,194 lines)

---

## Experiment Management & Tracking (Section 4)

### MLflow Setup

**Tracking URI:** `sqlite:///mlflow.db`
**Experiment name:** `fraud-anomaly-detection`

**Start UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://127.0.0.1:5000
```

### 4 Experiments Tracked

| Experiment | Parameters | Runs |
|---|---|---|
| fraud-anomaly-detection | max_iter=1000, RF=200, LGB=500 | 6 |
| fraud-anomaly-detection-v2-max-iter500 | max_iter=500, LR only | 2 |
| fraud-anomaly-detection-v3-rf100-lgb200 | RF=100, LGB=200 | 6 |
| fraud-anomaly-detection-v4-rf300-lgb100 | RF=300, LGB=100 | 6 |

### Best Model

**LightGBM from Experiment 3 (v3-rf100-lgb200)**
- cv_mean_f1: 0.718
- ROC-AUC: 0.999
- LGB=200 estimators with learning_rate=0.05

For detailed experiment comparison see: [Experiment Comparison Report](reports/phase2-experiment-comparision.md)

---

## Model Selection

### Chosen Models

| Model               | Purpose                               |
| ------------------- | ------------------------------------- |
| Logistic Regression | Baseline fraud classification         |
| Random Forest       | Ensemble benchmark                    |
| LightGBM            | Gradient boosting benchmark           |
| XGBoost             | High-performance fraud classification |

### Chosen Model as Best
- Model Type: LightGBM (Pipeline B) + LR_balanced (Pipeline A)
- Best Hyperparameters: LGB n_estimators=200, learning_rate=0.05, max_iter=1000
- Performance Metrics: cv_mean_f1=0.718, ROC-AUC=0.999 (LightGBM)

### Performance Metrics

Models were evaluated using:

* Accuracy
* F1-score
* ROC-AUC
* Average Precision
* Cross-validation metrics

---

## Key Results

* XGBoost achieved the strongest fraud detection performance
* Ensemble methods improved non-linear fraud detection capability
* Feature engineering significantly improved fraud representation
* Containerized execution successfully reproduced local training workflows
* Dockerized workflows improved reproducibility and operational consistency
* LightGBM (v3-rf100-lgb200) achieved best cv_mean_f1 of 0.718 and ROC-AUC of 0.999 across all experiments
* 4 MLflow experiments tracked with different hyperparameters  more LR iterations (1000 vs 500) consistently improved performance
* Scalene profiling identified train_model.py consuming 87.9% of runtime  cross_val_score parallelized with n_jobs=-1 for ~4x speedup
* cProfile output (22,194 lines) saved to reports/profiling_output.txt for full bottleneck analysis

## Challenges and Solutions

| Challenge                              | Resolution                           |
| -------------------------------------- | ------------------------------------ |
| Docker dependency failures             | Added required native Linux packages |
| LightGBM runtime errors                | Installed libgomp1 runtime support   |
| Container reproducibility issues       | Added Docker Compose orchestration   |
| Artifact persistence inside containers | Added persistent volume mounting     |
| Dependency inconsistencies             | Standardized Python 3.11 runtime     |
| Scalene segfault on macOS ARM64 with n_jobs=-1 | Used cProfile as fallback + lightweight 8k row script for Scalene |
| MLflow file store missing meta.yaml | Switched to SQLite backend (sqlite:///mlflow.db) |
| DVC Google Drive token expired | Re-authenticated using gdrive_client_id credentials |
| Metadata JSON merge conflicts | Added RGhazzal_*_metadata.json to .gitignore |
| cross_val_score slow (95s, 61% of runtime) | Parallelized with n_jobs=-1 (~4x speedup) |


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

- Start Date:  05/08/2026
- Estimated Completion: 05/19/2026
- Actual Completion: 05/21/2026
- Status: Completed
