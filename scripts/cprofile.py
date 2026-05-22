"""
cProfile profiling script for the fraud detection training pipeline.
"""

import cProfile
import pstats
from pathlib import Path

from mlops_frauddetection.train_model import main

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

PROFILE_BINARY = OUTPUT_DIR / "cprofile_training.prof"
PROFILE_TEXT = OUTPUT_DIR / "cprofile_training_output.txt"


def run_profile() -> None:
    """
    Run training pipeline with cProfile.
    """

    profiler = cProfile.Profile()

    profiler.enable()

    main()

    profiler.disable()

    profiler.dump_stats(PROFILE_BINARY)

    with PROFILE_TEXT.open("w") as f:
        stats = pstats.Stats(profiler, stream=f)

        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)

    print(f"Saved binary profile to: {PROFILE_BINARY}")
    print(f"Saved readable report to: {PROFILE_TEXT}")


if __name__ == "__main__":
    run_profile()
