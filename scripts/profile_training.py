"""
Profiling the LR fraud-anomaly training pipeline using Scalene.

Runing it with:
    scalene scripts/profile_training.py

Or with cProfile:
    python -m cProfile -s cumulative \
    scripts/profile_training.py \
    > reports/profiling_output.txt
"""

from pathlib import Path

from mlops_frauddetection.train_model import train_lr_pipeline

if __name__ == "__main__":
    train_lr_pipeline(
        data_path=Path("data/processed"),
        model_dir=Path("models"),
        max_iter=1000,
        seed=42,
        run_smote=False,  # faster for profiling
    )
