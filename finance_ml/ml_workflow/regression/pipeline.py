"""pipeline.py
Default regression entry point with safety rails and standardized outputs.

This module provides a small, opinionated entry point for running a
regression model with:

- Outlier safety rails (winsorization/clipping/non-negativity)
- Standardized predictions schema (via :func:`build_predictions_frame`)
- Lightweight schema validation (via :func:`validate_predictions_schema`)

It is intentionally minimal and self-contained so tests can exercise the
core behavior (non-negative, bounded predictions + schema compliance)
without depending on the full CLI or notebook pipeline.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from finance_ml.ml_workflow.validation.splits import create_train_test_split
from .io import build_predictions_frame, validate_predictions_schema
from .robust import clip_predictions, enforce_non_negative


def run_default_regression_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "price_target",
    test_size: float = 0.2,
    random_state: int = 42,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Run a small default regression pipeline with safety rails applied.

    This helper is designed primarily for tests and simple batch runs. It:

    1. Splits the data into train/test partitions.
    2. Trains a ``HistGradientBoostingRegressor`` on the training set.
    3. Applies safety rails to the test predictions:
       - ``clip_predictions`` based on the training targets.
       - ``enforce_non_negative`` to guarantee non-negative outputs.
    4. Builds a standardized predictions dataframe via
       :func:`build_predictions_frame` using the original rows as the
       metadata source.
    5. Validates the resulting schema using
       :func:`validate_predictions_schema` and raises ``ValueError`` if the
       core contract is violated.

    Args:
        df: Source dataframe containing features, target, and metadata
            (e.g., ``ticker``, ``sector``, ``region``, ``last_price``).
        feature_cols: Names of feature columns to use for training.
        target_col: Name of the regression target column.
        test_size: Proportion of samples to reserve for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with standardized prediction columns and error metrics.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    # Extract features and target
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    # Use shared split utility to respect data split policy
    # We split on the full dataframe to preserve metadata (including dates).
    train_df, test_df = create_train_test_split(
        df,
        date_col=date_col,
        group_col=None,
        stratify_col=None,
        test_size=test_size,
        random_state=random_state,
    )

    # Derive X/y partitions from the policy-driven splits
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target_col].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target_col].astype(float)

    # Simple, robust tree-based regressor (handles basic non-linearities)
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    # Raw predictions on test set
    preds_raw = model.predict(X_test)

    # Apply safety rails: clip based on training distribution, then enforce non-negativity
    preds_clipped = clip_predictions(preds_raw, y_train.values, n_std=3.0)
    preds_safe = enforce_non_negative(preds_clipped, threshold=0.0)

    # Build standardized predictions frame using original df as metadata source
    df_source = df.loc[y_test.index]
    preds_df = build_predictions_frame(y_true=y_test, y_pred=preds_safe, df_source=df_source)

    # Validate schema; raise if core contract is violated
    validation = validate_predictions_schema(preds_df)
    if not validation["ok"]:
        raise ValueError(f"Predictions schema validation failed: {validation['errors']}")

    return preds_df
