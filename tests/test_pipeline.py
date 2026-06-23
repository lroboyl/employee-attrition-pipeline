import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (
    load_config,
    handle_missing_values,
    encode_target,
    encode_categorical_features,
    drop_constant_columns,
    preprocess_pipeline,
    split_data,
)
from evaluate import evaluate_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small synthetic dataframe that mimics the real HR dataset."""
    return pd.DataFrame({
        "Age":             [25, 35, 45, 30, None],
        "MonthlyIncome":   [3000, 5000, 8000, 4000, 6000],
        "DistanceFromHome":[5, 10, 3, 15, 20],
        "Attrition":       ["Yes", "No", "No", "Yes", "No"],
        "Department":      ["Sales", "R&D", "HR", "Sales", "R&D"],
        "OverTime":        ["Yes", "No", "No", "Yes", "No"],
        "Gender":          ["Female", "Male", "Male", "Female", "Male"],
    })


@pytest.fixture(scope="module")
def full_df():
    """The real IBM HR dataset, used for data validation tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def config():
    """Project config loaded from YAML."""
    return load_config(CONFIG_PATH)


# ── Unit tests: handle_missing_values ─────────────────────────────────────────

def test_fill_missing_replaces_nulls(sample_df):
    """Null values in numeric columns should be gone after filling."""
    result = handle_missing_values(sample_df)
    assert result["Age"].isna().sum() == 0, "There should be no missing values in Age after filling"


def test_fill_missing_uses_correct_median(sample_df):
    """The filled value should equal the column median calculated from non-null rows.

    Non-null Ages are [25, 35, 45, 30], whose median is 32.5.
    The null at index 4 should be replaced with 32.5.
    """
    result = handle_missing_values(sample_df)
    expected_median = sample_df["Age"].median()
    assert result.loc[4, "Age"] == expected_median, (
        f"Expected null filled with median {expected_median}, got {result.loc[4, 'Age']}"
    )


def test_fill_missing_does_not_modify_original(sample_df):
    """The original dataframe must remain unchanged.

    handle_missing_values uses df.copy() internally, so calling it should
    leave the input completely unchanged.
    """
    null_count_before = sample_df["Age"].isna().sum()
    handle_missing_values(sample_df)
    assert sample_df["Age"].isna().sum() == null_count_before, (
        "Original dataframe should not be modified after calling handle_missing_values"
    )


def test_fill_missing_leaves_complete_columns_unchanged(sample_df):
    """Columns with no nulls should be the same after filling."""
    result = handle_missing_values(sample_df)
    pd.testing.assert_series_equal(result["MonthlyIncome"], sample_df["MonthlyIncome"])


def test_fill_missing_no_nulls_remain(sample_df):
    """After filling, the entire dataframe should contain zero null values."""
    result = handle_missing_values(sample_df)
    assert result.isnull().sum().sum() == 0, "No nulls should remain anywhere in the result"


# ── Unit tests: encode_target ─────────────────────────────────────────────────

def test_encode_target_maps_yes_to_one(sample_df):
    """'Yes' in the Attrition column should become the integer 1."""
    result = encode_target(sample_df)
    assert result.loc[0, "Attrition"] == 1, "Row 0 is 'Yes' so it should encode to 1"


def test_encode_target_maps_no_to_zero(sample_df):
    """'No' in the Attrition column should become the integer 0."""
    result = encode_target(sample_df)
    assert result.loc[1, "Attrition"] == 0, "Row 1 is 'No' so it should encode to 0"


def test_encode_target_only_produces_binary_values(sample_df):
    """The encoded column should contain only 0 and 1, nothing else."""
    result = encode_target(sample_df)
    assert set(result["Attrition"].unique()).issubset({0, 1}), (
        "Encoded Attrition column should only contain values 0 and 1"
    )


def test_encode_target_does_not_modify_original(sample_df):
    """The input dataframe should still have the original string values after encoding."""
    original_values = sample_df["Attrition"].tolist()
    encode_target(sample_df)
    assert sample_df["Attrition"].tolist() == original_values, (
        "Original dataframe shouldn't be modified after calling encode_target"
    )


def test_encode_target_raises_for_missing_column(sample_df):
    """Passing a column name that doesn't exist should raise a KeyError."""
    with pytest.raises(KeyError):
        encode_target(sample_df, target_col="NonexistentColumn")


def test_encode_target_raises_for_unexpected_values():
    """If the target column contains values other than 'Yes'/'No', raise ValueError.

    This guards against datasets where attrition is encoded differently,
    e.g. 'TRUE'/'FALSE' or '1'/'0' as strings.
    """
    bad_df = pd.DataFrame({"Attrition": ["Maybe", "Yes", "No"]})
    with pytest.raises(ValueError, match="Unexpected values"):
        encode_target(bad_df)


# ── Unit tests: encode_categorical_features ───────────────────────────────────

def test_encode_categoricals_produces_integer_dtype(sample_df):
    """After label encoding, the column dtype should be an integer type."""
    result, _ = encode_categorical_features(sample_df, ["Department"])
    assert result["Department"].dtype in [np.int64, np.int32, int], (
        "LabelEncoder should produce an integer column, not strings"
    )


def test_encode_categoricals_does_not_modify_original(sample_df):
    """Label encoding shouldn't change the values in the original dataframe."""
    original_values = sample_df["Department"].tolist()
    encode_categorical_features(sample_df, ["Department"])
    assert sample_df["Department"].tolist() == original_values, (
        "Original dataframe should still contain string values after encoding"
    )


def test_encode_categoricals_returns_encoder_for_each_column(sample_df):
    """The returned encoders dict should have one entry per encoded column."""
    _, encoders = encode_categorical_features(sample_df, ["Department", "OverTime"])
    assert "Department" in encoders, "Encoders dict should contain an entry for Department"
    assert "OverTime" in encoders, "Encoders dict should contain an entry for OverTime"


def test_encode_categoricals_raises_for_non_list_input(sample_df):
    """Passing a plain string instead of a list should raise a ValueError."""
    with pytest.raises(ValueError):
        encode_categorical_features(sample_df, "Department")


def test_encode_categoricals_skips_columns_not_in_dataframe(sample_df):
    """If a requested column doesn't exist, the function should skip it silently."""
    result, encoders = encode_categorical_features(sample_df, ["NonexistentCol"])
    assert "NonexistentCol" not in encoders, (
        "A column that doesn't exist shouldn't appear in the encoders dict"
    )


# ── Unit tests: drop_constant_columns ─────────────────────────────────────────

def test_drop_columns_removes_specified_column(sample_df):
    """A column listed in columns_to_drop should not appear in the result."""
    result = drop_constant_columns(sample_df, ["Gender"])
    assert "Gender" not in result.columns, "Gender should have been dropped"


def test_drop_columns_ignores_names_not_in_dataframe(sample_df):
    """Requesting to drop a column that doesn't exist shouldn't raise or change shape."""
    result = drop_constant_columns(sample_df, ["NonexistentColumn"])
    assert result.shape == sample_df.shape, (
        "Shape should be unchanged when the column to drop doesn't exist"
    )


def test_drop_columns_raises_for_non_list_input(sample_df):
    """Passing a string instead of a list should raise a ValueError."""
    with pytest.raises(ValueError):
        drop_constant_columns(sample_df, "Gender")


# ── Data validation tests ─────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "NumCompaniesWorked", "OverTime", "TotalWorkingYears",
    "PerformanceRating", "WorkLifeBalance", "YearsAtCompany",
]


def test_dataset_has_expected_columns(full_df):
    """Every column the pipeline depends on must be present in the raw CSV."""
    for col in EXPECTED_COLUMNS:
        assert col in full_df.columns, f"Expected column '{col}' is missing from the dataset"


def test_target_column_contains_only_yes_or_no(full_df):
    """Attrition should only ever contain 'Yes' or 'No', no other strings."""
    unique_values = set(full_df["Attrition"].unique())
    assert unique_values.issubset({"Yes", "No"}), (
        f"Unexpected values in Attrition column: {unique_values - {'Yes', 'No'}}"
    )


def test_age_is_within_working_range(full_df):
    """All employees should be between 18 and 65 years old."""
    assert full_df["Age"].between(18, 65).all(), (
        "Age values found outside the expected working range of [18, 65]"
    )


def test_monthly_income_is_positive(full_df):
    """MonthlyIncome should always be a positive number."""
    assert (full_df["MonthlyIncome"] > 0).all(), (
        "MonthlyIncome contains zero or negative values"
    )


def test_no_column_is_entirely_null(full_df):
    """No column should be completely empty. That would indicate a data loading error."""
    for col in full_df.columns:
        assert full_df[col].isnull().sum() < len(full_df), (
            f"Column '{col}' is entirely null, which likely means a load error"
        )


def test_dataset_has_enough_rows_to_train(full_df):
    """The dataset should have at least 1,000 rows to produce a meaningful model."""
    assert len(full_df) >= 1000, (
        f"Dataset only has {len(full_df)} rows — too few for reliable training"
    )


def test_years_at_company_is_non_negative(full_df):
    """You can't have a negative number of years at a company."""
    assert (full_df["YearsAtCompany"] >= 0).all(), (
        "YearsAtCompany contains negative values"
    )


def test_performance_rating_is_in_valid_range(full_df):
    """IBM's performance rating scale runs from 1 to 4."""
    assert full_df["PerformanceRating"].between(1, 4).all(), (
        "PerformanceRating contains values outside the expected scale of 1–4"
    )


# ── Model validation tests ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model(config):
    """Train a small, fast model on the real data for use in model validation tests.

    We use n_estimators=20 (not the production value of 100) to keep tests fast.
    The goal here is not to evaluate production performance but to verify that
    the model object itself does what it is supposed to do.
    """
    df = pd.read_csv(DATA_PATH)
    X, y, _, _ = preprocess_pipeline(df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_predictions_have_correct_shape(trained_model):
    """predict() should return exactly one label per row in the test set."""
    model, X_test, y_test = trained_model
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),), (
        f"Expected predictions shape ({len(X_test)},), got {preds.shape}"
    )


def test_predictions_are_binary(trained_model):
    """A binary classifier should only output 0 or 1, never any other value."""
    model, X_test, y_test = trained_model
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1}), (
        f"Predictions contained unexpected values: {set(preds) - {0, 1}}"
    )


def test_predict_proba_shape_matches_test_set(trained_model):
    """predict_proba() should return one probability pair per row."""
    model, X_test, y_test = trained_model
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2), (
        f"Expected proba shape ({len(X_test)}, 2), got {proba.shape}"
    )


def test_class_probabilities_sum_to_one(trained_model):
    """For each row, the probabilities for class 0 and class 1 must sum to 1.0."""
    model, X_test, y_test = trained_model
    proba = model.predict_proba(X_test)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
        err_msg="Class probabilities do not sum to 1.0 for every row")


def test_model_meets_minimum_roc_auc(trained_model):
    """
    Even the fast 20-tree model should clear a basic ROC-AUC bar of 0.70.

    If this fails, something is wrong with the preprocessing or feature set,
    not just with hyperparameter tuning.
    """
    model, X_test, y_test = trained_model
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["roc_auc"] >= 0.70, (
        f"Model ROC-AUC {metrics['roc_auc']:.4f} is below the minimum acceptable threshold of 0.70"
    )

# ── Edge case: fit=False without encoders or scaler ───────────────────────────
 
def test_pipeline_raises_if_fit_false_without_encoders(config):
    """
    Ensure preprocess_pipeline is not called incorrectly on test data.

    If fit=False is used without providing existing encoders, the function should
    fail immediately to avoid accidentally fitting new encoders on test data.
    """
    df = pd.read_csv(DATA_PATH)
    with pytest.raises(ValueError, match="encoders must be provided"):
        preprocess_pipeline(df, config, fit=False, encoders=None)
 
 
def test_pipeline_raises_if_fit_false_without_scaler(config):
    """
    Ensure preprocess_pipeline is not used incorrectly when fit=False.

    If fit=False is set but no scaler is provided, the function fails immediately
    to prevent accidental refitting on test data.

    Fitting a new scaler on test data would use different mean and variance than
    training, which would distort all numeric features and lead to invalid
    predictions.
    """ 
    df = pd.read_csv(DATA_PATH)
    _, _, fitted_encoders, _ = preprocess_pipeline(df, config)
    with pytest.raises(ValueError, match="scaler must be provided"):
        preprocess_pipeline(df, config, fit=False, encoders=fitted_encoders, scaler=None)