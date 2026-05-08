# Fraud Anomaly Classification & Behavioral Analytics

Scalable ML system for fraud & Anomoly detection and transaction behavior analysis

## Overview

Welcome to Fraud Anomaly Classification & Behavioral Analytics! This project is designed to provide a scalable, production-ready machine learning pipeline.

## Quick Start

### Installation

```bash
cd mlops_frauddetection 

# Create virtual environment
python -m venv .venv 
source .venv/bin/activate    # macOS/Linux 
# .venv\Scripts\activate     # Windows 

# Using pip

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Using uv (faster alternative)

# Install runtime dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -e ".[dev]"
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict
```

## Documentation

- [Getting Started](getting_started.md)
- [API Reference](api.md)

## Project Structure

```
mlops_frauddetection/                  # Repository root
├── src/
│   └── mlops_frauddetection/          # Importable package (src/ layout)
│       ├── config.py                  # Paths + typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel ABC + concrete Model
│       ├── evaluation/                # Metric helpers
│       ├── visualization/             # Plot helpers
│       ├── utils/                     # seed, io
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── data/                              # raw/ and processed/
├── models/                            # Trained artifacts
├── tests/                             # Unit tests
├── docs/                              # MkDocs docs
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

This project is licensed under the MIT License. See LICENSE for details.
