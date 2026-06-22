"""
Compare MLflow experiment runs and select the best one based on ROC-AUC.

Ranks multiple runs from a batch of experiments so you can quickly see which
hyperparameters performed best without using the MLflow UI.
"""

import argparse
import mlflow
import pandas as pd
import yaml
 
 
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file and return it as a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
 
 
def compare_experiments(
    config_path: str = "configs/config.yaml",
    primary_metric: str = "metrics.roc_auc",
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Retrieve all MLflow runs for an experiment and display a ranked table.

    Runs are ordered using MLflow's server-side sorting for efficiency. Column names
    are simplified for readability.

    Returns the full sorted DataFrame for further analysis or filtering.
    """
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment found with name '{experiment_name}'.")
        print("Run train.py at least once to create experiments.")
        return pd.DataFrame()

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{primary_metric} DESC"],
    )

    if runs_df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        return runs_df
    
    primary_col = primary_metric.split(".")[-1]

    default_metric_cols = [
        "metrics.roc_auc",
        "metrics.f1",
        "metrics.precision",
        "metrics.recall",
        "metrics.accuracy",
    ]

    extra_metric_cols = [primary_metric] if primary_metric not in default_metric_cols else []

    display_cols = [
        "run_id",
        "params.model_type",
        "params.n_estimators",
        "params.max_depth",
        "params.class_weight",
    ] + default_metric_cols + extra_metric_cols + [
    "start_time",
    "status",
    ]
    
    available_cols = [c for c in display_cols if c in runs_df.columns]

    rename_map = {
        "params.model_type":   "model",
        "params.n_estimators": "n_est",
        "params.max_depth":    "depth",
        "params.class_weight": "class_wt",
        "metrics.roc_auc":     "roc_auc",
        "metrics.f1":          "f1",
        "metrics.precision":   "precision",
        "metrics.recall":      "recall",
        "metrics.accuracy":    "accuracy",
    }

    if primary_metric not in rename_map:
        rename_map[primary_metric] = primary_col

    display_df = runs_df[available_cols].copy().rename(columns=rename_map)

    # Include the custom metric at the front of metric_cols so it appears first
    # in the statistics section and the best run summary
    metric_cols = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    if primary_col not in metric_cols:
        metric_cols = [primary_col] + metric_cols
    
    for col in metric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Total runs: {len(display_df)}")
    print("=" * 80)
    print(f"\nTop {min(n_top, len(display_df))} runs (ranked by {primary_col}):")
    print(display_df.head(n_top).to_string(index=False))
 
    # Best run summary highlights whichever metric was used for ranking
    best_run = display_df.iloc[0]
    print("\n" + "=" * 80)
    print(f"BEST RUN SUMMARY (ranked by {primary_col})")
    print("=" * 80)
    print(f"  Run ID:  {best_run.get('run_id', 'N/A')}")
    print(f"  Model:   {best_run.get('model', 'N/A')}")
    print(f"  {primary_col:<12} {best_run.get(primary_col, 'N/A')}  ← ranked by this")
    for col in [c for c in metric_cols if c != primary_col]:
        print(f"  {col:<12} {best_run.get(col, 'N/A')}")
 
    print("\n" + "=" * 80)
    print("METRIC STATISTICS (all runs)")
    print("=" * 80)
    for col in metric_cols:
        if col in display_df.columns:
            vals = display_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:12s}  min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")
 
    return display_df
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MLflow experiment runs.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="configs/config.yaml",
        help="Path to the YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--metric",
        default="metrics.roc_auc",
        help="MLflow metric to rank runs by (default: metrics.roc_auc)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top runs to display (default: 5)",
    )
    args = parser.parse_args()
    compare_experiments(args.config_path, primary_metric=args.metric, n_top=args.top)