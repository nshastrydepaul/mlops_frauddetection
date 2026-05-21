# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
Phase 2 focuses on scaling and operationalizing Fraud-Anomoly Detection & Behavioral Analytics by implementing containerization, advanced monitoring, profiling, experiment tracking, and comprehensive logging. This phase ensures your model can be reliably deployed, monitored in production, and continuously improved through systematic experimentation.

---

## 1. Containerization

- [x] **Dockerfile Creation**: Build Dockerfile for model training and inference
- [x] **Base Image Selection**: Choose appropriate base image (python:3.x, nvidia/cuda, etc.)
- [x] **Environment Variables**: Define and document required environment variables
- [x] **Build Instructions**: Document how to build Docker image with examples
- [x] **Run Instructions**: Document how to run container with proper volume/network config
- [x] **Container Testing**: Test container locally to ensure consistency with host environment
- [x] **Docker Compose (Optional)**: Create docker-compose.yml for multi-service setups
- [x] **Environment Consistency**: Verify that containerized training produces identical results to local training

Containerization Summary

A multi-stage Docker workflow was implemented using:

```bash
FROM python:3.11-slim
```
Docker Compose orchestration was added for reproducible execution of:

- preprocessing
- feature engineering
- model training
- MLflow tracking
- artifact generation

The containerized environment successfully executed:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- SMOTE training workflows

Docker runtime debugging resolved several native dependency issues including:

- missing make
- missing curl
- missing git
- missing libgomp1 for LightGBM runtime support

Containerized execution was validated using:

```bash
docker compose build --no-cache
docker compose up
```

The final workflow completed successfully with:

```text
exited with code 0
```

confirming reproducible end-to-end ML execution inside Docker.

---

## 2. Monitoring & Debugging

- [ ] **Debugging Tools**: Set up pdb/ipdb for interactive debugging
- [ ] **Debugging Documentation**: Document how to debug in containerized environment
- [ ] **Debug Scenario 1**: Create example scenario and solution document for [specific problem]
- [ ] **Debug Scenario 2**: Create example scenario and solution document for [specific problem]
- [x] **Logging for Debugging**: Implement detailed logging at critical points in code
- [x] **Model Assertion Checks**: Add assertions to catch data/model anomalies early
- [x] **Training Validation**: Implement sanity checks (NaN detection, shape validation, etc.)

Implemented Monitoring & Validation

The project currently implements:

- dataset shape validation
- fraud distribution logging
- preprocessing validation
- runtime metric logging
- NaN detection and handling
- structured INFO/WARNING/ERROR logs

Validation checks were added throughout:

- preprocessing pipeline
- feature engineering
- train/test splitting
- model training workflows

Rich logging provides:

- colored logs
- runtime diagnostics
- structured console output
- traceback visibility

---

## 3. Profiling & Optimization

- [ ] **CPU Profiling**: Use cProfile to profile training and inference
- [ ] **Memory Profiling**: Profile memory usage with memory_profiler or similar
- [ ] **GPU Profiling (if applicable)**: Use PyTorch Profiler or similar for GPU workloads
- [ ] **Profiling Results**: Document baseline profiling results and bottlenecks identified
- [x] **Optimization 1**: Implement and measure optimization (e.g., vectorization, caching)
- [x] **Optimization 2**: Implement and measure additional optimization
- [x] **Performance Benchmarks**: Document before/after performance metrics
- [x] **Optimization Documentation**: Explain each optimization and its impact

Implemented Optimizations

Current optimizations include:

- vectorized feature engineering
- reusable sklearn preprocessing pipelines
- optimized preprocessing workflows
- SMOTE balancing integration
- reusable feature engineering utilities
- parallelized ensemble training where supported

Performance evaluation includes:

- F1-score comparison
- ROC-AUC tracking
- Average Precision metrics
- cross-validation evaluation
- comparison across Logistic Regression, Random Forest, LightGBM, and XGBoost

Training metrics and runtime outputs are logged and persisted as artifacts during execution.

---

## 4. Experiment Management & Tracking

- [ ] **MLflow Setup**: Initialize MLflow tracking server and client configuration
  - OR **Weights & Biases Setup**: Initialize W&B project and team workspace
- [ ] **Metric Logging**: Log training/validation metrics for each experiment
- [ ] **Parameter Logging**: Log all hyperparameters and configuration values
- [ ] **Model Artifact Logging**: Save model checkpoints and artifacts to tracking system
- [ ] **Experiment Comparison**: Create comparison of at least 3 different experiments
- [ ] **Visualization**: Generate performance comparison charts/plots
- [ ] **Best Model Selection**: Document criteria and process for selecting best model from experiments
- [ ] **Experiment Documentation**: Create table summarizing all experiments with results

---

## 5. Application & Experiment Logging

- [ ] **Logger Setup**: Configure Python logger with appropriate handlers and formatters
  - OR **Rich Library Setup**: Use rich for enhanced console output and logging
- [ ] **Log Levels**: Implement and use DEBUG, INFO, WARNING, ERROR appropriately
- [ ] **Log Messages**: Add informative log messages at key points in code
- [ ] **Training Log Example**: Document and include sample training log output
- [ ] **Inference Log Example**: Document and include sample inference log output
- [ ] **Error Logging**: Implement comprehensive error logging with context
- [ ] **Performance Logging**: Log timing information for performance analysis
- [ ] **Log Rotation**: Configure log rotation to prevent disk space issues

---

## 6. Configuration Management

- [ ] **Hydra Setup**: Install and configure Hydra for config management
- [ ] **Config Files**: Create YAML config files for train/eval/inference configurations
- [ ] **Config Structure**: Organize configs with appropriate hierarchy (base, model, data, etc.)
- [ ] **Config Example 1**: Create and document sample training config
- [ ] **Config Example 2**: Create and document alternative config (different hyperparameters)
- [ ] **Config Validation**: Implement config validation and schema checking
- [ ] **Override Documentation**: Document how to override config values from command line
- [ ] **Config Version Control**: Version all configs alongside code

---

## 7. Documentation & Repository Updates

- [x] **README Update**: Update README to include:
  - [x] Containerization section with Docker usage
  - [ ] Debugging and profiling guide
  - [ ] Experiment tracking setup instructions
  - [ ] Configuration management guide
  - [ ] Logging usage examples
- [ ] **Architecture Documentation**: Document system architecture with diagrams
- [x] **Setup Guide**: Update setup guide to include all Phase 2 tools
- [x] **Examples**: Add examples of running with different configurations
- [x] **Tool Integration**: Document how all tools work together
- [x] **Troubleshooting**: Add troubleshooting section for common issues
- [x] **Performance Guide**: Document how to profile and optimize
- [x] **Version Compatibility**: Document version requirements for all tools

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables.
