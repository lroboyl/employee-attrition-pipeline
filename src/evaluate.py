"""
Functions for evaluating an HR attrition prediction model.

Includes metrics, reporting, and threshold checks so each evaluation step can
be used independently. The functions can be tested without running training.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate key classification metrics and return them in a dictionary.
    Uses predicted probabilities instead of class labels so ROC-AUC can be computed
    more accurately. A custom threshold can be used to turn probabilities into
    labels, allowing experimentation without retraining the model.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_test, y_pred),
    }


def print_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Display evaluation results in a readable format and return the metrics.

    Prints the confusion matrix because summary metrics like F1 score can hide
    important differences in prediction behavior. The raw counts (TP, FP, FN, TN)
    help show how well the model handles each class.
    """
    metrics = evaluate_model(model, X_test, y_test, threshold)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"ROC-AUC Score:  {metrics['roc_auc']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))

    return metrics


def check_thresholds(metrics: Dict[str, float], config: dict) -> bool:
    """
    Check model metrics against minimum thresholds from the config.

    Prints PASS/FAIL for each metric so results are easy to read in logs.
    Returns True only if all thresholds are met, otherwise False.
    """
    thresholds = config.get("thresholds", {})
    passed = True

    checks = [
        ("roc_auc",   "min_roc_auc"),
        ("f1",        "min_f1"),
        ("precision", "min_precision"),
        ("recall",    "min_recall"),
    ]

    print("\nThreshold Checks:")
    for metric_key, threshold_key in checks:
        if threshold_key in thresholds:
            min_val = thresholds[threshold_key]
            actual_val = metrics.get(metric_key, 0)
            status = "PASS" if actual_val >= min_val else "FAIL"
            if status == "FAIL":
                passed = False
            print(f"  {metric_key}: {actual_val:.4f} >= {min_val} [{status}]")

    return passed


def exit_if_thresholds_not_met(metrics: Dict[str, float], config: dict) -> None:
    """
    Stop the program if any metric fails its threshold.

    Exits with an error code (1) so CI/CD fails when the model does not meet
    quality requirements like recall or ROC-AUC.
    """
    if not check_thresholds(metrics, config):
        print("\nERROR: Model does not meet minimum performance thresholds!")
        sys.exit(1)
    else:
        print("\nAll performance thresholds met.")


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Return feature importances as a sorted dataframe for tree-based models.

    If the model does not support feature_importances_,
    an empty dataframe is returned so callers can safely check .empty instead of
    handling errors.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)