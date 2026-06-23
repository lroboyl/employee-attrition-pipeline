"""
Monitor data drift for the HR attrition model.

Compares the training data distribution to simulated production data using
Evidently. If the overall drift score exceeds the threshold in config.yaml,
the script exits with code 1 so it can be used as an automated health check
in CI or scheduled jobs.
"""

import os
import sys
import glob
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file and return it as a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_reference_data(config: dict) -> pd.DataFrame:
    """
    Load the training CSV and apply basic cleaning.

    Drops constant columns and converts the target column (Attrition) to binary
    so the dataset matches the pipeline schema. No scaling or encoding is applied
    because drift tools need the original feature values.
    """
    df = pd.read_csv(config["data"]["raw_path"])

    drop_existing = [c for c in config["features"]["drop"] if c in df.columns]
    df = df.drop(columns=drop_existing)

    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Shuffle so tail() doesn't capture a non-representative slice
    df = df.sample(frac=1, random_state=config["data"]["random_state"]).reset_index(drop=True)
    
    return df

def split_reference_and_production(
    df: pd.DataFrame, production_fraction: float = 0.30, random_state: int = 42
):
    """
    Create a holdout production split with no overlap from the reference data.

    Uses a proper train/test split instead of slicing so the two datasets contain
    different rows. This ensures the production data is truly unseen before any
    drift is added.
    """
    reference_df, production_df = train_test_split(
        df, test_size=production_fraction, random_state=random_state
    )
    return reference_df.reset_index(drop=True), production_df.reset_index(drop=True)

def add_drift(production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realistic drift to selected features to simulate production changes.

    Applies controlled shifts to six key features such as age, income, distance,
    tenure, work-life balance, and overtime to mimic real-world trends like a
    younger workforce, higher pay, more remote distance, and increased workload.

    A fixed random seed ensures the results are reproducible across runs.
    """
    rng = np.random.default_rng(seed=99)
    prod_df = production_df.copy()
 
    if "Age" in prod_df.columns:
        noise = rng.normal(loc=-5, scale=3, size=len(prod_df))
        prod_df["Age"] = (prod_df["Age"] + noise).clip(18, 60).round().astype(int)
 
    if "MonthlyIncome" in prod_df.columns:
        multiplier = rng.uniform(1.10, 1.25, size=len(prod_df))
        prod_df["MonthlyIncome"] = (prod_df["MonthlyIncome"] * multiplier).round().astype(int)
 
    if "DistanceFromHome" in prod_df.columns:
        noise = rng.exponential(scale=4, size=len(prod_df))
        prod_df["DistanceFromHome"] = (prod_df["DistanceFromHome"] + noise).clip(1, 29).round().astype(int)
 
    if "YearsAtCompany" in prod_df.columns:
        noise = rng.normal(loc=-2, scale=1.5, size=len(prod_df))
        prod_df["YearsAtCompany"] = (prod_df["YearsAtCompany"] + noise).clip(0, 40).round().astype(int)
 
    if "WorkLifeBalance" in prod_df.columns:
        shift_mask = rng.random(len(prod_df)) < 0.35
        prod_df.loc[shift_mask, "WorkLifeBalance"] = rng.choice([1, 2], size=shift_mask.sum())
 
    if "OverTime" in prod_df.columns:
        overtime_mask = rng.random(len(prod_df)) < 0.20
        prod_df.loc[overtime_mask, "OverTime"] = "Yes"
 
    return prod_df


def run_drift_detection(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    config: dict,
    output_dir: str = "reports/",
    keep_last_n: int = 10,
) -> dict:
    """
    Run Evidently drift detection and save the report as an HTML file.

    Drops the target column so only feature drift is measured, since model reliability
    depends on whether input features have changed, not the label distribution.

    Ensures both datasets have matching columns to avoid silent errors. Returns a
    summary including drift share, number of features, details of drifted features,
    and the report file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    target_col = config["features"]["target"]
    ref_features = reference_df.drop(columns=[target_col], errors="ignore")
    prod_features = production_df.drop(columns=[target_col], errors="ignore")

    common_cols = [c for c in ref_features.columns if c in prod_features.columns]
    ref_features = ref_features[common_cols]
    prod_features = prod_features[common_cols]

    logger.info("Running drift detection on %d features", len(common_cols))
    logger.info("  Reference rows: %d", len(ref_features))
    logger.info("  Production rows: %d", len(prod_features))
 
    report = Report([DataDriftPreset()])
    result = report.run(ref_features, prod_features)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    result.save_html(report_path)
    logger.info("Drift report saved to: %s", report_path)

    # Keeping only the most recent keep_last_n
    existing_reports = sorted(glob.glob(os.path.join(output_dir, "drift_report_*.html")))
    for old_report in existing_reports[:-keep_last_n]:
        os.remove(old_report)
        logger.info("Removed old report: %s", old_report)
 
    # Parse results using the evidently 0.7.x dict structure:
    # DriftedColumnsCount gives overall drift share and count
    # ColumnValueDrift gives per-column drift detection
    result_dict = result.dict()
 
    drifted_features = []
    total_features = len(common_cols)
    drift_share = 0.0

    try:
        metrics = result_dict.get("metrics", [])
        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            value = metric.get("value", {})

            # Overall drift share from DriftedColumnsCount
            if "DriftedColumnsCount" in metric_name:
                drift_share = value.get("share", 0.0)

            # Per-column drift from ValueDrift(column=...)
            # value is a float (the test statistic/p-value), not a dict
            # drift is detected when value < threshold in the config
            if metric_name.startswith("ValueDrift(column="):
                config_data = metric.get("config", {})
                col_name = config_data.get("column", "unknown")
                threshold = config_data.get("threshold", 0.05)
                drift_score = float(value) if value is not None else None
                drift_detected = drift_score is not None and drift_score < threshold
                if drift_detected:
                    drifted_features.append({
                        "feature": col_name,
                        "drift_score": drift_score,
                        "is_drifted": True,
                    })
    except Exception as e:
        logger.warning("Could not parse drift results dict: %s", e)
        logger.warning("Check the HTML report for full results: %s", report_path)
 
    return {
        "drifted_features": drifted_features,
        "total_features":   total_features,
        "drift_share":      drift_share,
        "report_path":      report_path,
    }


def log_drift_summary(drift_results: dict) -> None:
    """Log a short summary sorted by drift score so the features with the highest drift appear first."""
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total features monitored: %d", drift_results["total_features"])
    logger.info("Features with drift:      %d", len(drift_results["drifted_features"]))
    logger.info("Overall drift share:      %.2f%%", drift_results["drift_share"] * 100)
 
    if drift_results["drifted_features"]:
        logger.warning("Drifted features detected:")
        for feat in sorted(
            drift_results["drifted_features"],
            key=lambda x: x.get("drift_score") or 0,
            reverse=True,
        ):
            score = feat["drift_score"]
            if score is None:
                score_str = "N/A"
            elif score < 0.0001:
                score_str = "<0.0001"
            else:
                score_str = f"{score:.4f}"
            logger.warning("  %-35s  drift_score=%s", feat["feature"], score_str)
    else:
        logger.info("No individual feature drift detected.")


def main(config_path: str = "configs/config.yaml") -> None:
    """
    Run the full drift detection process and fail if drift is too high.

    Uses a threshold defined in the config so it can be adjusted over time as the
    team better understands what level of drift impacts model performance.
    """
    config = load_config(config_path)
    drift_threshold = config["thresholds"].get("drift_threshold", 0.3)

    logger.info("Loading data...")
    df = load_reference_data(config)
 
    logger.info("Splitting into reference and production sets...")
    reference_df, production_df = split_reference_and_production(
        df, random_state=config["data"]["random_state"]
    )
 
    logger.info("Add drift into production data...")
    production_df = add_drift(production_df)
 
    drift_results = run_drift_detection(
        reference_df,
        production_df,
        config,
        output_dir=config["output"].get("reports_path", "reports/"),
    )
 
    log_drift_summary(drift_results)
 
    drift_share = drift_results["drift_share"]
    logger.info("Drift threshold: %.2f%%", drift_threshold * 100)
 
    if drift_share > drift_threshold:
        logger.warning(
            "ALERT: Drift share %.2f%% exceeds threshold %.2f%%. Investigate and consider retraining.",
            drift_share * 100, drift_threshold * 100,
        )
        sys.exit(1)
    else:
        logger.info(
            "Drift share %.2f%% is within acceptable threshold. Continue monitoring.",
            drift_share * 100,
        )
 
 
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config_path)