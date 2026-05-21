# PHASE 1: Project Design & Model Development

## Overview
Phase 1 focused on establishing the foundational MLOps workflow for the Fraud Anomaly Classification & Behavioral Analytics project.

The team successfully configured the repository structure, development environment, preprocessing pipeline, feature engineering workflow, dataset versioning using DVC, and baseline machine learning models.

This phase emphasized reproducibility, modular project organization, collaborative development practices, and baseline experimentation using financial transaction data.

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Define the problem statement, goals, and success metrics for Fraud-Anomoly Detection & Behavioral Analytics
- [x] **Detailed Description**: Write a 300+ word project description covering the business context, technical approach, and expected outcomes
- [x] **Dataset Selection**: Choose appropriate dataset(s) and document the selection justification
- [x] **Dataset Description**: Document dataset characteristics (size, features, format, sources)
- [x] **Model Considerations**: Identify initial model architectures and algorithms suitable for your problem
- [x] **Open-Source Tools**: Document and justify the selection of open-source tools and libraries for the project

---

## 2. Code Organization & Setup

- [x] **GitHub Repository**: Create repository with cookiecutter MLOps structure
- [x] **Environment Setup**: Configure Python virtual environment (venv or conda)
- [x] **Dependency Management**: Create and maintain requirements.txt or pyproject.toml
- [x] **Project Structure**: Organize code with clear separation of concerns (src/, tests/, data/, etc.)
- [x] **Version Pinning**: Pin all critical dependencies to specific versions
- [x] **Installation Documentation**: Document how to set up the development environment

---

## 3. Version Control & Collaboration

- [x] **Regular Commits**: Establish commit discipline with descriptive, atomic commits
- [x] **Branching Strategy**: Implement feature branching (e.g., git-flow or GitHub Flow)
- [x] **Pull Request Process**: Establish PR template and review requirements
- [x] **Team Roles**: Clearly define responsibilities (author: MergeDeployGraduate, team members, reviewers)
- [x] **Code Review Guidelines**: Document code review expectations and checklist
- [x] **Commit History**: Maintain clean, readable git history for project traceability

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: Create reproducible scripts for data cleaning and preprocessing
- [x] **Normalization**: Implement feature normalization/standardization with proper documentation
- [ ] **Data Augmentation**: Develop and document data augmentation strategies if applicable
- [x] **Data Documentation**: Create data dictionary with feature descriptions and data types
- [x] **Data Splits**: Define and implement train/validation/test split strategy
- [x] **Data Validation**: Create scripts to validate data quality and consistency
- [x] **DVC Setup (Optional)**: Initialize DVC for data versioning if managing large datasets

---

## 5. Model Training

- [ ] **Training Environment**: Set up local/cloud training environment with GPU support if needed
- [x] **Baseline Model**: Implement and train a baseline model
- [ ] **Hyperparameter Configuration**: Document baseline hyperparameters and rationale
- [x] **Evaluation Metrics**: Define and calculate relevant metrics (accuracy, F1, RMSE, etc.)
- [x] **Model Persistence**: Save trained models with version information
- [ ] **Training Reproducibility**: Ensure training is reproducible (seed management, logging)
- [x] **Performance Baseline**: Document baseline model performance as reference point

---

## 6. Documentation & Reporting

- [x] **README**: Create comprehensive README with:
  - [x] Project overview and objectives
  - [x] Setup and installation instructions
  - [x] Quick start guide for running training
  - [x] Dependencies and requirements
  - [x] Contributing guidelines
  - [x] License information
- [ ] **Code Docstrings**: Add docstrings to all functions and classes (NumPy/Google style)
- [ ] **Code Style**: Implement ruff configuration for linting
- [ ] **Type Hints**: Add type hints throughout codebase
- [ ] **Type Checking**: Configure mypy for static type checking
- [x] **Makefile**: Create Makefile with commands for:
  - [x] `make setup` - install dependencies
  - [x] `make train` - run training pipeline
  - [x] `make test` - run tests
  - [x] `make lint` - run linting checks
  - [x] `make format` - auto-format code
- [ ] **CONTRIBUTING.md**: Document contribution guidelines and development workflow
- [ ] **API Documentation**: Document all public APIs and interfaces

---

## 7. Feature Engineering

Additional behavioral features were engineered to improve fraud pattern detection and customer behavior analysis.

### Engineered Features

- `customer_txn_count`
  - Total number of transactions performed by a customer

- `avg_amt_per_customer`
  - Average transaction amount per customer

- `merchant_txn_count`
  - Total transactions processed by a merchant

## 7.1 Additional Feature Engineering:

- `amt_ratio`  Amount involved in transaction compared to customer's average spending
- `combined_risk`  Combination of the two types of merchant risk mentioned previously
- `amt_risk_score`  Interaction effect of `amt_ratio` on `combined_risk`
- `is_high_spend`  Boolean variable indicating that the transaction amount is more than 1.5 times the customer's average
- `night_high_amt`  Boolean variable representing

These features help capture transactional behavior patterns commonly associated with fraud analytics systems.

---

## 8. Data Processing Pipeline

The preprocessing pipeline performs the following steps:

1. Load raw transaction dataset
2. Remove unnecessary geographic and leakage columns
3. Apply feature engineering transformations
4. Encode categorical variables
5. Split dataset using stratified train/test splitting
6. Save processed datasets for reproducible training

### Data Cleaning
The following columns were removed to reduce leakage and unnecessary dimensionality:
- SSN
- Credit card number
- Account number
- Transaction identifiers
- Exact geographic coordinates

### Encoding Strategy
- High-cardinality features were label encoded
- Low-cardinality categorical features were one-hot encoded

---

## 9. Baseline Model Results

### Logistic Regression (Class Weight Balanced)

| Metric | Value |
|---|---|
| Train Accuracy | 66.12% |
| Test Accuracy | 65.28% |
| Train F1 Score | 0.6648 |
| Test F1 Score | 0.6555 |

The baseline Logistic Regression model demonstrated stable generalization performance with similar train and test metrics.

### SMOTE Experimentation

SMOTE oversampling was evaluated to address class imbalance. While synthetic oversampling improved minority-class representation during training, it reduced test generalization performance. The original baseline configuration was retained for Phase 1 benchmarking.

### SMOTE + Logistic Regression

| Metric | Value |
|---|---|
| Train Accuracy | 51.33% |
| Test Accuracy | 60.77% |
| Train F1 Score | 0.5016 |
| Test F1 Score | 0.6149 |

---

## Phase 1 Achievements

### Completed Components

- Established modular MLOps repository structure
- Configured Python environment and dependency management
- Integrated DVC for dataset versioning
- Implemented preprocessing pipeline
- Developed feature engineering transformations
- Performed exploratory data analysis
- Trained Logistic Regression baseline model
- Conducted SMOTE imbalance experimentation
- Generated evaluation metrics and confusion matrices
- Configured linting, testing, and pre-commit hooks

### Key Findings

- The dataset contains strong class imbalance across fraud categories
- Behavioral features improved transaction representation
- Logistic Regression generalized consistently between train and test datasets
- SMOTE oversampling reduced test generalization performance
- Ensemble models such as Random Forest are expected to outperform linear baselines

---

## Team Contributions

| Team Member | Responsibilities |
|---|---|
| Nishanth Shastry | DVC setup, preprocessing pipeline, feature engineering, project documentation, code review |
| Raail | Logistic Regression, Random Forest, LightGBM, XGBoost implementation & evaluation, project documentation  |
| Musaddiq | Logistic Regression baseline model, feature engineering, project documentation, code review |
| Lohith | Exploratory Data Analysis, visualization, project documentation, document review |

> **Checklist:** Use this as a guide for documenting your Phase 1 deliverables.
