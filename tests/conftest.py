"""
Shared test fixtures for Finance ML test suite.

Phase 7 (Restructuring Plan):
This conftest.py provides centralized fixtures that can be used across
all test modules in unit/, integration/, and regression/ directories.

Usage:
    Fixtures defined here are automatically available to all tests.
    Import is not required - pytest discovers and injects them.

Example:
    def test_something(sample_financial_df):
        # sample_financial_df is automatically injected
        assert len(sample_financial_df) > 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Sample DataFrames
# =============================================================================


@pytest.fixture
def sample_financial_df() -> pd.DataFrame:
    """Create a small sample financial DataFrame for testing.

    Contains basic columns used across many tests:
    - ticker, sector, region (categorical)
    - last_price, price_target (price columns)
    - market_cap, revenue, ebitda (financial metrics)

    Returns:
        DataFrame with 10 sample stocks across 3 sectors
    """
    np.random.seed(42)
    n = 10

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:02d}" for i in range(n)],
            "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n),
            "region": np.random.choice(["US", "EU", "APAC"], n),
            "last_price": np.random.uniform(10, 500, n).round(2),
            "price_target": np.random.uniform(10, 600, n).round(2),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            "revenue": np.random.uniform(1e8, 1e11, n),
            "ebitda": np.random.uniform(1e7, 1e10, n),
            "volatility": np.random.uniform(0.1, 0.8, n),
        }
    )


@pytest.fixture
def sample_predictions_df(sample_financial_df) -> pd.DataFrame:
    """Create a DataFrame following the standardized predictions schema.

    Extends sample_financial_df with prediction-related columns:
    - y_true, y_pred, y_pred_calibrated
    - pred_p10, pred_p50, pred_p90 (quantile predictions)
    - abs_error, pct_error

    Returns:
        DataFrame with standardized prediction columns
    """
    df = sample_financial_df.copy()

    # Add prediction columns
    df["y_true"] = df["price_target"]
    df["y_pred"] = df["price_target"] * np.random.uniform(0.9, 1.1, len(df))
    df["y_pred_calibrated"] = df["y_pred"] * 1.02  # Slight calibration adjustment

    # Quantile predictions
    df["pred_p10"] = df["y_pred"] * 0.85
    df["pred_p50"] = df["y_pred"]
    df["pred_p90"] = df["y_pred"] * 1.15

    # Error metrics
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["pct_error"] = df["abs_error"] / df["y_true"] * 100

    return df


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Create a DataFrame for classification testing.

    Contains features and labels for event classification:
    - Numeric features (feature_1 through feature_5)
    - Categorical features (sector, region)
    - Event labels (0=Neutral, 1=Positive, 2=Negative)

    Returns:
        DataFrame with 50 samples for classification tests
    """
    np.random.seed(42)
    n = 50

    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "sector": np.random.choice(["Tech", "Finance", "Energy", "Healthcare"], n),
            "region": np.random.choice(["US", "EU", "APAC", "ROTW"], n),
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
            "feature_4": np.random.randn(n),
            "feature_5": np.random.randn(n),
            "last_price": np.random.uniform(10, 100, n),
            "price_target": np.random.uniform(10, 100, n),
            "event_label": np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]),
        }
    )


@pytest.fixture
def sample_regression_data():
    """Create synthetic X, y data for regression testing.

    Returns:
        Tuple of (X_df, y_series) with 200 samples and 10 features
    """
    np.random.seed(42)
    n = 200
    n_features = 10

    X = np.random.randn(n, n_features)
    y = np.dot(X, np.random.randn(n_features)) + np.random.randn(n) * 0.1 + 100

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")

    return X_df, y_series


# =============================================================================
# DataFrame with Missing Values
# =============================================================================


@pytest.fixture
def sample_df_with_nulls() -> pd.DataFrame:
    """Create a DataFrame with intentional missing values for imputation tests.

    Returns:
        DataFrame with NaN values in various columns
    """
    np.random.seed(42)
    n = 20

    df = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(n)],
            "sector": ["Tech"] * 10 + ["Finance"] * 10,
            "last_price": np.random.uniform(10, 200, n),
            "price_target": np.random.uniform(10, 200, n),
            "revenue": np.random.uniform(1e8, 1e10, n),
            "ebitda": np.random.uniform(1e7, 1e9, n),
            "market_cap": np.random.uniform(1e9, 1e11, n),
        }
    )

    # Introduce NaN values
    df.loc[0, "revenue"] = np.nan
    df.loc[1, "ebitda"] = np.nan
    df.loc[2, "price_target"] = np.nan
    df.loc[5, "market_cap"] = np.nan
    df.loc[10, "revenue"] = np.nan
    df.loc[15, "ebitda"] = np.nan

    return df


# =============================================================================
# Temporary Directory Fixture
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs.

    Returns:
        Path object pointing to a temporary directory
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def simple_trained_model(sample_regression_data):
    """Create a simple trained Ridge regression model for testing.

    Returns:
        Fitted Ridge model
    """
    from sklearn.linear_model import Ridge

    X, y = sample_regression_data
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    return model


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Return default configuration values for testing.

    Returns:
        Dict with common configuration parameters
    """
    return {
        "random_seed": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "n_jobs": 1,
        "verbose": False,
    }
