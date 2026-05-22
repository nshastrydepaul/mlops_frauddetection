"""Reproducibility: seed every RNG we care about in one call."""

from __future__ import annotations

import os
import random

import numpy as np

from mlops_frauddetection.logging_config import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and any framework-specific RNGs available.

    Args:
        seed: Integer seed for all RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seed set: %d (Python, NumPy, PYTHONHASHSEED)", seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.info("PyTorch seed set: %d (CPU + CUDA)", seed)
        else:
            logger.info("PyTorch seed set: %d (CPU only — CUDA not available)", seed)
    except ImportError:
        logger.debug("PyTorch not installed — skipping torch seed")

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        logger.info("TensorFlow seed set: %d", seed)
    except ImportError:
        logger.debug("TensorFlow not installed — skipping tf seed")
