"""Train a model while tracking results with MLflow.

Preprocessing and evaluation functions are imported from separate modules, so
each part of the pipeline is independent and easier to test.
"""

import os
import pickle
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple

from preprocess import load_config, load_data, preprocess_pipeline, split_data
from evaluate import print_evaluation_report, exit_if_thresholds_not_met, get_feature_importance


def get_model(config: dict):
    """Create a scikit-learn model based on the type specified in the config.

    All model hyperparameters are defined in config.yaml, so experiments can be
    changed by editing the config file and rerunning the pipeline.

    If the model type is not supported, a ValueError is raised immediately to
    avoid silent failures or incorrect defaults.
    """
    m = config["model"]
    t = m["type"]
    rs = m.get("random_state", 42)

    if t == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", None),
            min_samples_split=m.get("min_samples_split", 2),
            min_samples_leaf=m.get("min_samples_leaf", 1),
            class_weight=m.get("class_weight", "balanced"),
            random_state=rs,
            n_jobs=-1,
        )
    elif t == "GradientBoostingClassifier":
        return GradientBoostingClassifier(
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", 3),
            learning_rate=m.get("learning_rate", 0.1),
            random_state=rs,
        )
    elif t == "LogisticRegression":
        return LogisticRegression(
            C=m.get("C", 1.0),
            class_weight=m.get("class_weight", "balanced"),
            random_state=rs,
            max_iter=m.get("max_iter", 1000),
        )
    elif t == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            max_depth=m.get("max_depth", None),
            min_samples_split=m.get("min_samples_split", 2),
            class_weight=m.get("class_weight", "balanced"),
            random_state=rs,
        )
    else:
        raise ValueError(f"Unsupported model type: {t}")


def save_artifacts(model, encoders: dict, scaler, output_dir: str = "models/") -> None:
    """Save the model and all preprocessing components in one file using pickle.

    This includes encoders and the scaler so that inference can reproduce the
    exact same transformations used during training.

    This is important because recreating these objects later could produce
    different mappings or scaling values, leading to inconsistent predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in [("model", model), ("encoders", encoders), ("scaler", scaler)]:
        with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
    print(f"Artifacts saved to {output_dir}")


def load_artifacts(output_dir: str = "models/"):
    """Load and return (model, encoders, scaler) from a previously saved run."""
    artifacts = {}
    for name in ["model", "encoders", "scaler"]:
        with open(os.path.join(output_dir, f"{name}.pkl"), "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts["model"], artifacts["encoders"], artifacts["scaler"]


def train(
    config_path: str = "configs/config.yaml",
    check_thresholds: bool = True,
) -> Tuple[object, dict]:
    """Run one full training experiment and log everything to MLflow.

    The sequence is: load config → set up MLflow run → log hyperparameters →
    load and preprocess data → train → evaluate → log metrics → save artifacts.
    Setting check_thresholds=False is useful during development when you want
    to inspect results without the run aborting on a missed threshold.
    """
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        m = config["model"]

        # Log every hyperparameter from config so each run is fully reproducible
        mlflow.log_params({
            "model_type":        m["type"],
            "n_estimators":      m.get("n_estimators", "N/A"),
            "max_depth":         m.get("max_depth", "N/A"),
            "min_samples_split": m.get("min_samples_split", "N/A"),
            "min_samples_leaf":  m.get("min_samples_leaf", "N/A"),
            "class_weight":      m.get("class_weight", "N/A"),
            "learning_rate":     m.get("learning_rate", "N/A"),
            "C":                 m.get("C", "N/A"),
            "test_size":         config["data"]["test_size"],
            "random_state":      config["data"]["random_state"],
            "data_version":      config["data"].get("dvc_version", "unknown"),
        })

        print(f"Loading data from: {config['data']['raw_path']}")
        df = load_data(config["data"]["raw_path"])
        print(f"Data shape: {df.shape}")

        X, y, encoders, scaler = preprocess_pipeline(df, config)
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        model = get_model(config)
        print(f"Training {m['type']}...")
        model.fit(X_train, y_train)

        metrics = print_evaluation_report(model, X_test, y_test)

        mlflow.log_metrics({
            "roc_auc":   metrics["roc_auc"],
            "f1":        metrics["f1"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "accuracy":  metrics["accuracy"],
        })

        importance_df = get_feature_importance(model, X_train.columns.tolist())
        if not importance_df.empty:
            print("\nTop 5 Feature Importances:")
            print(importance_df.head(5).to_string(index=False))

        # Log the model object itself as an MLflow artifact for later retrieval
        mlflow.sklearn.log_model(model, artifact_path="model")

        os.makedirs("models", exist_ok=True)
        save_artifacts(model, encoders, scaler, "models/")

        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")

    if check_thresholds:
        exit_if_thresholds_not_met(metrics, config)

    return model, metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    train(config_path)