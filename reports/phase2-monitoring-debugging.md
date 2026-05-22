# Monitoring and Debugging

## 2.1 Monitoring

To monitor the health and performance of the fraud detection training pipelines, this project implements a lightweight monitoring solution using `psutil`, `GPUtil`.

The monitoring system was integrated into the project training pipelines located in:

```text
src/mlops_frauddetection/train_model.py
```

The resource monitoring utility was implemented in:

```text
src/mlops_frauddetection/monitoring/system_monitoring.py
```

---

# Monitoring Tools Used

## psutil-Based Monitoring

A lightweight monitoring script was implemented using the `psutil` library.

This script continuously logs system resource usage during training into CSV files.

### Why psutil Was Chosen

`psutil` was selected because it is a lightweight, cross platform library that integrates well into the project. It eliminates the need for complex external infrastructure like Prometheus making system montioring straightforward and self-contained.

---

## GPUtil GPU Monitoring

Optional GPU monitoring was added using `GPUtil`.

If a compatible GPU is available, the monitor records:
- GPU utilization
- GPU memory usage

If no supported GPU exists, the monitor safely records GPU fields as unavailable.

---

# Metrics Monitored and Meaning

The monitoring script records the following system metrics:

| Metric | Description |
|---|---|
| `cpu_percent` | CPU utilization percentage |
| `memory_percent` | Total system RAM utilization percentage |
| `memory_used_mb` | Amount of RAM currently used |
| `memory_available_mb` | Available free RAM |
| `process_memory_mb` | RAM used specifically by the Python training process |
| `disk_usage_percent` | Disk usage percentage |
| `gpu_load_percent` | GPU utilization percentage |
| `gpu_memory_used_mb` | GPU memory currently used |
| `gpu_memory_total_mb` | Total GPU memory available |

CPU utilization is recorded as a percentage because CPU usage is normally measured using utilization percentages.

RAM metrics are recorded both as percentages and MB values to show actual memory consumption.

---

# Monitoring Implementation

Monitoring was added to both major training pipelines:

```text
train_lr_pipeline()
train_ensemble_pipeline()
```

Each pipeline starts monitoring before training begins and stops monitoring in a `finally` block to ensure cleanup even if training fails.

Example structure:

```python
monitor_path = Path("reports/monitoring/lr_pipeline_resource_usage.csv")

monitor = ResourceMonitor(monitor_path)
monitor.start()

pipeline_start_time = time.time()

try:
    # training code
    ...
finally:
    monitor.stop()
```

---

# Monitoring Output Files

Monitoring results are saved to CSV files:

```text
reports/monitoring/lr_pipeline_resource_usage.csv
reports/monitoring/ensemble_pipeline_resource_usage.csv
```

These files contain timestamped resource monitoring data collected during model training.

---

# 2.2 Debugging Practices

Debugging techniques were applied to the project training code.

The debugging implementation includes:
- assertion checks
- runtime validation
- logging
- Python `pdb` debugging

The debugging code was implemented in:

```text
src/mlops_frauddetection/train_model.py
```

---

# Debug Mode

A command-line debugging flag was added:

```bash
--debug
```

This enables additional validation and debugger support during training.

Usage:

```bash
python -m mlops_frauddetection.train_model --pipeline lr --debug
```

or:

```bash
python -m mlops_frauddetection.train_model --pipeline ensemble --debug
```

---

# Debug Validation Checks

A helper function called `run_debug_checks()` was implemented.

This function validates:
- datasets are not empty
- feature and label counts match
- shapes are correct
- class distributions are correct


# pdb Debugger Usage

Python's built-in `pdb` debugger was used for line-by-line debugging.

The debugger is triggered using:

```python
pdb.set_trace()
```

When training is run with `--debug`, execution pauses and enters the debugger.

Useful debugger commands and meaning:

| Command | Meaning |
|---|---|
| `p` | print variable |
| `n` | execute next line |
| `c` | continue execution |
| `q` | quit debugger |

---

# Common Debugging Scenarios

## Scenario 1: Empty Dataset

### Problem

Training fails because one or more datasets are empty.

### Debugging Approach

Assertions stop execution immediately if datasets are empty.

### Resolution

Verify that preprocessing successfully generated:

```text
X_train.csv
y_train.csv
X_test.csv
y_test.csv
```

---

## Scenario 2: Feature and Label Mismatch

### Problem

Training fails because feature rows and label rows do not match.

### Debugging Approach

Assertions validate row counts before model training begins.


### Resolution

Regenerate train-test splits and verify preprocessing consistency.

---

## Scenario 3: Unexpected Class Distribution

### Problem

Fraud datasets are highly imbalanced. Incorrect class distributions can negatively affect model performance.

### Debugging Approach

The debug mode logs class distributions.

Example:

```python
logger.info(
    "DEBUG y_train class distribution:\n%s",
    y_train.value_counts()
)
```

### Resolution

Inspect preprocessing logic, SMOTE configuration, and target labels.

---

### Conclusion

This monitoring and debugging setup keeps the training pipelines running smoothly. while `psutil` watches the system's health, the `--debug` mode catches data errors before they can ruin a training run. Together this simple tools make the fraud detection system reliable, transparent and easy to fix if somehting goes wrong.
