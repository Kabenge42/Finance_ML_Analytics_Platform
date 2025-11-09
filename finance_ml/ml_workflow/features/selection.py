"""
finance_ml.ml_workflow.features.selection - Feature importance and selection

This module provides feature importance calculation and selection methods:
- Mutual information regression
- Random Forest feature importance
- SHAP values
- Recursive Feature Elimination (RFE)

Phase 9.3 refactor: Extracted from advanced_features.py for better modularity.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_feature_importance_mutual_info",
    "calculate_feature_importance_rf",
    "calculate_feature_importance_shap",
    "calculate_feature_importance_rfe",
]


def calculate_feature_importance_mutual_info(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None
) -> pd.DataFrame:
    """Calculate feature importance using mutual information.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    # Handle missing and invalid values
    X_clean = X.copy()

    # Replace infinite values with NaN first
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    X_clean = X_clean.fillna(X_clean.median())

    # For any remaining NaN, fill with 0
    X_clean = X_clean.fillna(0)

    # Clip extreme values
    for col in X_clean.columns:
        col_data = X_clean[col]
        if col_data.std() > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_clean[col] = np.clip(col_data, lower_bound, upper_bound)

    # Handle target variable
    y_clean = y.replace([np.inf, -np.inf], np.nan)
    y_clean = y_clean.fillna(y_clean.median())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(y_clean.mean())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(0)

    # Calculate mutual information
    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)

    # Create result DataFrame
    importance_df = pd.DataFrame({"feature": X.columns, "importance": mi_scores}).sort_values(
        "importance", ascending=False
    )

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated mutual information for {len(X.columns)} features")
    return importance_df


def calculate_feature_importance_rf(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None, n_estimators: int = 100
) -> pd.DataFrame:
    """Calculate feature importance using Random Forest.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    top_k : int, optional
        Number of top features to return. If None, returns all features.
    n_estimators : int, default=100
        Number of trees in the random forest

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by importance.
        Note: Only features present after data cleaning are included.

    Notes
    -----
    - Only numeric columns are used; non-numeric columns are automatically removed
    - Rows with NaN values in X or y are automatically cleaned (filled with median/0)
    - Feature importance is calculated only for features in the cleaned dataset
    - Returns empty DataFrame if no valid samples or features remain after cleaning
    """
    # Handle missing and invalid values
    X_clean = X.copy()

    # Select only numeric columns to avoid TypeError with median calculation
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    X_clean = X_clean[numeric_cols]

    # Replace infinite values with NaN first
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median (now safe because we only have numeric columns)
    X_clean = X_clean.fillna(X_clean.median())

    # For any remaining NaN (e.g., all-NaN columns), fill with 0
    X_clean = X_clean.fillna(0)

    # Clip extremely large values to prevent overflow
    # Use 3 standard deviations as threshold for each column
    for col in X_clean.columns:
        col_data = X_clean[col]
        if col_data.std() > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_clean[col] = np.clip(col_data, lower_bound, upper_bound)

    # Handle target variable
    y_clean = y.replace([np.inf, -np.inf], np.nan)
    y_clean = y_clean.fillna(y_clean.median())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(y_clean.mean())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(0)

    # If no valid samples or features remain, return empty DataFrame
    if len(X_clean) == 0 or len(X_clean.columns) == 0:
        return pd.DataFrame({"feature": [], "importance": []})

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_clean, y_clean)

    # Get feature importance
    importance_df = pd.DataFrame(
        {"feature": X_clean.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated Random Forest importance for {len(X_clean.columns)} features")
    return importance_df


def calculate_feature_importance_shap(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None, n_estimators: int = 50
) -> pd.DataFrame:
    """Calculate feature importance using SHAP values.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)
        n_estimators: Number of trees for tree-based model

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    # Clean data
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

    # Train a simple model for SHAP
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, max_depth=5, n_jobs=-1
    )
    model.fit(X_clean, y_clean)

    try:
        # Try to use SHAP if available
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean)

        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)
    except ImportError:
        shap = None  # Define in except block for type checker
        # Fallback to Random Forest feature importance if SHAP not available
        logger.warning("SHAP not available, falling back to Random Forest feature importance")
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated SHAP importance for {len(X.columns)} features")
    return importance_df


def calculate_feature_importance_rfe(
    X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 10, cv: Optional[int] = None
) -> List[str]:
    """Select features using Recursive Feature Elimination.

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features_to_select: Number of features to select
        cv: Number of cross-validation folds (uses RFECV if provided)

    Returns:
        List of selected feature names
    """
    # Clean data
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

    # Use Ridge regression as the base estimator (fast and stable)
    estimator = Ridge(alpha=1.0, random_state=42)

    if cv is not None and cv > 1:
        # Use RFECV for cross-validated selection
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            min_features_to_select=max(1, n_features_to_select // 2),
        )
    else:
        # Use RFE for simple selection
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features_to_select, len(X.columns)),
            step=1,
        )

    # Fit and get selected features
    selector.fit(X_clean, y_clean)
    selected_features = X.columns[selector.support_].tolist()

    logger.info(f"RFE selected {len(selected_features)} features from {len(X.columns)}")
    return selected_features
