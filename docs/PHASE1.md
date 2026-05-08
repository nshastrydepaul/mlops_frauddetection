# Phase 1: Project Setup and Data Exploration

## Overview

Phase 1 focused on setting up the foundational MLOps workflow for the Fraud Anomaly Classification & Behavioral Analytics project.

The team established the repository structure, configured the development environment, explored the dataset, implemented preprocessing and feature engineering pipelines, and trained baseline machine learning models.

---

## Objectives

- [x] Set up development environment
- [x] Explore and understand the dataset
- [x] Document dataset characteristics
- [x] Create preprocessing pipeline
- [x] Implement feature engineering
- [x] Establish baseline evaluation metrics
- [x] Configure DVC versioning

---

## Deliverables

### 1. Development Environment
- Python 3.11 environment configured
- Dependencies managed using `requirements.txt`
- Pre-commit hooks configured
- Testing environment validated

### 2. Data Exploration
- Dataset statistics analyzed
- Fraud class imbalance identified
- Transaction behavior patterns explored across time, location and spending
- Key predictive features identified for downstream modeling    
- Visualizations generated for all major distribution and behavioral patterns
- For the full detailed analysis, see the [Data Exploration Report](reports/phase1-data-exploration-report.md)

### 3. Data Processing Pipeline
- Data loading functions implemented
- Leakage columns removed
- Categorical encoding pipeline created
- Train/test splitting automated
- Processed datasets exported reproducibly

### 4. Feature Engineering
Implemented behavioral analytics features:
- Customer transaction counts
- Average transaction amount per customer
- Merchant transaction frequency

### 4.1 Additional Feature Engineering
- `amt_ratio` - Ratio of transaction amount compared to average amount spent by customer.
- `combined_risk` - Weighing of 30-day and 7-day risk of merchants.
- `amt_risk_score` - Combination of `amt_ratio` and `merchant_risk`.
- `is_high_spend` - Indicator variable for transaction amount greater than 1.5 times customer’s average spending.
- `night_high_amt` - Indicator variable for high-value transactions made during night time.

### 5. 4-Class Fraud Risk Label Scheme
- Created multi-label scheme merging `is_fraud` with merchant risk score
- TT (Truth Truth), TF (Truth False), FT (False Truth), FF (False False)
- FT/FF classification was based on the `merchant_risk_30_day` threshold (48/52 balanced ratio)

### 6. Baseline Modeling
- Logistic Regression baseline trained
- SMOTE imbalance experimentation completed
- Evaluation metrics generated using accuracy and F1-score
- Confusion matrices generated

## 7. Random Forest, LightGBM, XGBoost implementation & evaluation


### 8. Model Versioning
- Trained models were versioned using DVC and kept in Google Drive remote storage 
- Model metadata is present in the file models/LR_SMOTE_model_metadata.json

---

## Key Findings

- Strong class imbalance exists across fraud categories
- Behavioral features improved fraud representation
- For the full detailed analysis, see the [Data Exploration Report](reports/phase1-data-exploration-report.md)
- Logistic Regression generalized consistently between training and testing
- SMOTE oversampling reduced generalization performance
- Random Forest is expected to improve performance on non-linear fraud patterns

---

## Next Steps

Phase 2 will focus on:
- Containerization using Docker
- Experiment tracking improvements
- Model persistence
- CI/CD integration
- Monitoring and deployment preparation

---

## Status

- Start Date: 04/30/2026
- Estimated Completion: 05/04/2026
- Actual Completion: 05/07/2026
- Status: Completed
