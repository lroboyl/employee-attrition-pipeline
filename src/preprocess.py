"""
Data preprocessing functions for HR Attrition prediction.

"""

import os
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Also resolves the MLflow tracking URI to an absolute path so experiment logs
    are always saved to the same mlruns folder, no matter where the script is run
    from.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_uri = config.get("mlflow", {}).get("tracking_uri", "mlruns")
    if not os.path.isabs(raw_uri) and "://" not in raw_uri:
        config.setdefault("mlflow", {})["tracking_uri"] = os.path.join(project_root, raw_uri)
 
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Read the raw CSV from disk.

    Raises FileNotFoundError if the path does not exist, and ValueError if the
    file is empty, so callers get a clear message.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Data file is empty: {file_path}")

    return df


def drop_constant_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Remove columns that carry no information (constants or arbitrary IDs).

    Silently skips any names in columns_to_drop that aren't actually present,
    so the function is safe to call even when the schema changes between runs.
    Raises ValueError if columns_to_drop is not a list, catching the common
    mistake of passing a single string.
    """
    if not isinstance(columns_to_drop, list):
        raise ValueError("columns_to_drop must be a list")

    existing_cols = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=existing_cols)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a copy of the dataframe.

    - Numeric columns are filled with the column median.
    - Categorical columns are filled with the most common value (mode).

    Using the median helps prevent extreme values from affecting the fill value.
    The original dataframe is not modified because the function works on a copy.
    """ 
    result = df.copy()

    for col in result.columns:
        if result[col].dtype in ["int64", "float64"]:
            if result[col].isnull().any():
                result[col] = result[col].fillna(result[col].median())
        else:
            if result[col].isnull().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val[0])

    return result


def encode_target(df: pd.DataFrame, target_col: str = "Attrition") -> pd.DataFrame:
    """
    Convert the target column from text to numbers.

    - 'Yes' becomes 1
    - 'No' becomes 0

    Raises a KeyError if the column does not exist.
    Raises a ValueError if the column contains values other than 'Yes' or 'No'.
    """ 
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    result = df.copy()
    unique_vals = set(result[target_col].unique())
    unexpected = unique_vals - {"Yes", "No"}

    if unexpected:
        raise ValueError(f"Unexpected values in target column: {unexpected}")

    result[target_col] = result[target_col].map({"Yes": 1, "No": 0})
    return result


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    fit_encoders: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Convert categorical columns to numeric values using label encoding.

    Returns the encoded dataframe and the fitted encoders.

    Use the same encoders from training data when encoding validation or new data
    to keep category-to-number mappings consistent.

    Raises a ValueError if categorical_cols is not a list.
    """
    if not isinstance(categorical_cols, list):
        raise ValueError("categorical_cols must be a list")

    result = df.copy()
    encoders = fit_encoders if fit_encoders is not None else {}

    for col in categorical_cols:
        if col not in result.columns:
            continue

        if col in encoders:
            le = encoders[col]
            known_classes = set(le.classes_)
            # Map unseen labels to the first known class rather than crashing
            result[col] = result[col].apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            result[col] = le.transform(result[col])
        else:
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
            encoders[col] = le

    return result, encoders


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    fit_scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric columns so they have a mean of 0 and a standard deviation of 1.

    Use the scaler fitted on the training data when transforming validation or test
    data to ensure consistent scaling and avoid data leakage.
    """
    result = df.copy()
    existing_cols = [c for c in numeric_cols if c in result.columns]

    if fit_scaler is not None:
        result[existing_cols] = fit_scaler.transform(result[existing_cols])
        return result, fit_scaler

    scaler = StandardScaler()
    result[existing_cols] = scaler.fit_transform(result[existing_cols])
    return result, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    config: dict,
    fit: bool = True,
    encoders: Optional[dict] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.Series, dict, StandardScaler]:
    """
    Apply all preprocessing steps to a dataframe.

    The process:
    1. Remove columns with only one value.
    2. Fill missing values.
    3. Convert the target column to binary values.
    4. Encode categorical columns as numbers.
    5. Split the data into features (X) and target (y).
    6. Scale numeric columns.

    For validation or test data, set fit=False and provide the existing encoders
    and scaler so the same transformations are used without retraining them.

    Returns X, y, encoders, and scaler.
    """
    if not fit and encoders is None:
        raise ValueError(
            "encoders must be provided when fit=False. "
            "Pass the encoders returned from a fit=True call on the training data."
        )
    if not fit and scaler is None:
        raise ValueError(
            "scaler must be provided when fit=False. "
            "Pass the scaler returned from a fit=True call on the training data."
        )
    
    target_col = config["features"]["target"]

    df = drop_constant_columns(df, config["features"]["drop"])
    df = handle_missing_values(df)
    df = encode_target(df, target_col)

    fit_encoders_arg = None if fit else encoders
    df, fitted_encoders = encode_categorical_features(
        df, config["features"]["categorical"], fit_encoders_arg
    )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    fit_scaler_arg = None if fit else scaler
    X, fitted_scaler = scale_numeric_features(
        X, config["features"]["numeric"], fit_scaler_arg
    )

    return X, y, fitted_encoders, fitted_scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets while preserving class proportions.

    Stratification ensures both sets keep the same class balance as the original
    data, which is important when classes are imbalanced.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    config = load_config()
    df = load_data(config["data"]["raw_path"])
    print(f"Loaded data: {df.shape}")

    X, y, encoders, scaler = preprocess_pipeline(df, config)
    print(f"Preprocessed features: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")