"""
Run a sequence of five MLflow experiments using different model configurations.

Each experiment is defined by a separate config file, allowing MLflow to track
different models and hyperparameters independently. This makes it easy to compare
results later using the experiment tracking UI or comparison scripts.

This script is intended to be run once after project setup to populate MLflow
with a baseline set of experiments for evaluation.
"""

import subprocess
import sys


CONFIGS = [
    ("configs/config.yaml",        "RandomForest (baseline)"),
    ("configs/config_rf_200.yaml", "RandomForest (200 estimators)"),
    ("configs/config_gb.yaml",     "GradientBoosting"),
    ("configs/config_lr.yaml",     "LogisticRegression"),
    ("configs/config_dt.yaml",     "DecisionTree"),
]


def run_experiment(config_path: str, label: str) -> bool:
    """Run a single training experiment and return True if it succeeded."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Config:  {config_path}")
    print("="*60)

    result = subprocess.run(
        [sys.executable, "src/train.py", config_path],
        capture_output=False,
    )

    if result.returncode == 0:
        print(f"✓ {label} completed successfully")
        return True
    else:
        print(f"✗ {label} failed with exit code {result.returncode}")
        return False


if __name__ == "__main__":
    results = []
    for config_path, label in CONFIGS:
        success = run_experiment(config_path, label)
        results.append((label, success))

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for label, success in results:
        status = "✓ passed" if success else "✗ failed"
        print(f"  {status}  {label}")

    failed = [label for label, success in results if not success]
    if failed:
        print(f"\n{len(failed)} experiment(s) failed. Check output above.")
        sys.exit(1)
    else:
        print(f"\nAll 5 experiments completed. Run compare_experiments.py to compare results.")