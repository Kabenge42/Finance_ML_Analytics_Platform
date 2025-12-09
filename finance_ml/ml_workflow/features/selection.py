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
from typing import Optional, List, Union, Tuple, Dict

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
    "select_features_auto",
    "select_features_by_category",
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
    if isinstance(y, pd.Series):
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        y_clean = y_clean.fillna(y_clean.median())
        if y_clean.isna().any():
            y_clean = y_clean.fillna(y_clean.mean())
        if y_clean.isna().any():
            y_clean = y_clean.fillna(0)
    else:
        # Handle numpy array
        y_clean = np.copy(y)
        y_clean[np.isinf(y_clean)] = np.nan
        median = np.nanmedian(y_clean)
        y_clean[np.isnan(y_clean)] = median if not np.isnan(median) else 0

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
    if isinstance(y, pd.Series):
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        y_clean = y_clean.fillna(y_clean.median())
        if y_clean.isna().any():
            y_clean = y_clean.fillna(y_clean.mean())
        if y_clean.isna().any():
            y_clean = y_clean.fillna(0)
    else:
        # Handle numpy array
        y_clean = np.copy(y)
        y_clean[np.isinf(y_clean)] = np.nan
        median = np.nanmedian(y_clean)
        y_clean[np.isnan(y_clean)] = median if not np.isnan(median) else 0

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

    # Handle target variable
    if isinstance(y, pd.Series):
        y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())
    else:
        # Handle numpy array
        y_clean = np.copy(y)
        y_clean[np.isinf(y_clean)] = np.nan
        median = np.nanmedian(y_clean)
        y_clean[np.isnan(y_clean)] = median if not np.isnan(median) else 0

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


# ============================================================================
# Aliases for code_guidelines.md v1.10 API compliance
# ============================================================================


def select_features_rf(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: Optional[int] = None,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """
    Select features using Random Forest importance.

    Alias for calculate_feature_importance_rf() to match code_guidelines.md v1.10 API.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)
        n_estimators: Number of trees in the forest

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    return calculate_feature_importance_rf(X, y, top_k, n_estimators)


# ============================================================================
# Phase 9.3: Automated Feature Selection Pipeline
# ============================================================================


def select_features_auto(
    X: pd.DataFrame,
    y: pd.Series,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    method: str = "combined",
    preserve_columns: Optional[List[str]] = None,
    return_scores: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]:
    """
    Automated feature selection combining multiple methods.

    Implements Phase 9.3 Task 1: Automated Feature Selection Pipeline.
    Aligned with phase_9.3_implementation_plan.md and code_guidelines.md v1.10.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    importance_threshold : float, default=0.01
        Minimum importance score to retain feature
    correlation_threshold : float, default=0.95
        Maximum correlation to consider features redundant
    method : str, default='combined'
        Selection method: 'mutual_info', 'rf_importance', 'correlation', 'combined'
    preserve_columns : list, optional
        Columns to always preserve (defaults to PRICE_COLUMNS)
    return_scores : bool, default=False
        Whether to return importance scores

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Selected features, optionally with importance scores

    Notes
    -----
    - PRICE_COLUMNS are always preserved regardless of importance scores
    - Correlation-based deduplication keeps the feature with higher importance
    - Combined method applies both importance and correlation filtering
    - Only numeric columns are used for importance calculations
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

    # Default preserve columns to PRICE_COLUMNS
    if preserve_columns is None:
        preserve_columns = list(PRICE_COLUMNS)

    # Filter to only numeric columns for feature selection
    # Non-numeric columns (object, string, datetime) cannot be used in ML models
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].copy()

    if len(X_numeric.columns) == 0:
        logger.warning("No numeric columns found in feature matrix, returning empty selection")
        return pd.DataFrame()

    # Initialize selection mask (all numeric features selected initially)
    selected_features = set(X_numeric.columns)
    importance_scores = {}

    # Step 1: Importance-based selection
    if method in ["mutual_info", "rf_importance", "combined"]:
        if method in ["mutual_info", "combined"]:
            # Calculate mutual information scores (on numeric columns only)
            importance_df = calculate_feature_importance_mutual_info(X_numeric, y)
        else:
            # Calculate Random Forest importance (on numeric columns only)
            importance_df = calculate_feature_importance_rf(X_numeric, y)

        # Store scores
        importance_scores = dict(zip(importance_df["feature"], importance_df["importance"]))

        # Filter by threshold (but preserve protected columns)
        features_to_remove = set()
        for feature in X_numeric.columns:
            if feature not in preserve_columns:
                score = importance_scores.get(feature, 0.0)
                if score < importance_threshold:
                    features_to_remove.add(feature)

        selected_features -= features_to_remove
        logger.info(
            f"Importance filtering: removed {len(features_to_remove)} features below threshold {importance_threshold}"
        )

    # Step 2: Correlation-based deduplication
    if method in ["correlation", "combined"]:
        # Calculate correlation matrix for selected features
        X_selected = X_numeric[list(selected_features)]

        # Correlation matrix (already filtered to numeric columns above)
        if len(X_selected.columns) > 1:
            corr_matrix = X_selected.corr().abs()

            # Find highly correlated pairs
            redundant_features = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > correlation_threshold:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]

                        # Keep the feature with higher importance (or first one if no scores)
                        if importance_scores:
                            score_i = importance_scores.get(col_i, 0.0)
                            score_j = importance_scores.get(col_j, 0.0)
                            feature_to_remove = col_j if score_i >= score_j else col_i
                        else:
                            feature_to_remove = col_j  # Keep first by default

                        # Don't remove if it's a preserved column
                        if feature_to_remove not in preserve_columns:
                            redundant_features.add(feature_to_remove)

            selected_features -= redundant_features
            logger.info(
                f"Correlation filtering: removed {len(redundant_features)} redundant features above threshold {correlation_threshold}"
            )

    # Step 3: Always preserve protected columns (that are also numeric)
    for col in preserve_columns:
        if col in X_numeric.columns:
            selected_features.add(col)

    # Return selected DataFrame (from numeric columns only)
    X_selected = X_numeric[list(selected_features)]

    logger.info(
        f"Feature selection complete: {len(X_numeric.columns)} -> {len(X_selected.columns)} numeric features"
    )

    if return_scores:
        return X_selected, importance_scores
    else:
        return X_selected


def select_features_by_category(X: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """
    Select features by semantic category.

    Implements Phase 9.3 Task 1: Category-based feature selection.
    Supports Phase 9.3 feature categories: momentum, valuation, quality,
    profitability, cash_flow, growth.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    categories : list of str
        Categories to include (e.g., ['momentum', 'valuation'])

    Returns
    -------
    pd.DataFrame
        DataFrame with only features from specified categories

    Notes
    -----
    - Categories are inferred from feature name prefixes (e.g., 'momentum_rsi')
    - Features without category prefix are excluded
    """
    # Category prefix patterns
    category_patterns = {
        "momentum": "momentum_",
        "valuation": "valuation_",
        "quality": "quality_",
        "profitability": "profitability_",
        "cash_flow": "cash_flow_",
        "growth": "growth_",
    }

    # Build list of selected columns
    selected_columns = []
    for col in X.columns:
        col_lower = col.lower()
        for category in categories:
            pattern = category_patterns.get(category, f"{category}_")
            if col_lower.startswith(pattern):
                selected_columns.append(col)
                break

    logger.info(
        f"Category selection: selected {len(selected_columns)} features from categories {categories}"
    )

    return X[selected_columns]
