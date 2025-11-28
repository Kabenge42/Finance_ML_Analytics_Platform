"""Deprecation shim for advanced feature engineering (archived).

This module has been archived. Please import from the structured features
subpackage instead:
  - finance_ml.ml_workflow.features.advanced
  - finance_ml.ml_workflow.features.selection
  - finance_ml.ml_workflow.features.core

This shim re-exports the archived implementation to preserve behavior
while callers migrate to the new paths.
"""

from __future__ import annotations

import warnings
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.linear_model import Ridge

warnings.warn(
    "DEPRECATION NOTICE: 'finance_ml.ml_workflow.advanced_features' is archived. "
    "Use 'finance_ml.ml_workflow.features.*' modules instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export archived implementation only
from .archive.advanced_features import *  # noqa: F401,F403,E402

try:  # pragma: no cover
    from .archive.advanced_features import __all__ as _ALL  # type: ignore

    __all__ = list(_ALL)
except Exception:  # pragma: no cover
    __all__ = [name for name in globals().keys() if not name.startswith("_")]

# Minimal helpers to satisfy references inside archived content that some
# environments statically analyze. These are no-ops for the shim and will be
# overshadowed by the re-exported implementations when used.
logger = logging.getLogger(__name__)


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:  # pragma: no cover
    result = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce").replace(
        0, np.nan
    )
    return pd.Series(result).replace([np.inf, -np.inf], np.nan)


def engineer_temporal_features(
    df: pd.DataFrame, date_col: str = "next_earnings", reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Engineer temporal features from date columns.

    Args:
        df: Input DataFrame
        date_col: Name of date column to extract features from
        reference_date: Optional reference date for calculating days since

    Returns:
        DataFrame with temporal features added
    """
    result = df.copy()

    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found, skipping temporal features")
        return result

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
        try:
            result[date_col] = pd.to_datetime(result[date_col])
        except Exception as e:
            logger.warning(f"Could not convert {date_col} to datetime: {e}")
            return result

    # Extract fiscal quarter (1-4)
    result["fiscal_quarter"] = result[date_col].dt.quarter

    # Extract month (1-12)
    result["month"] = result[date_col].dt.month

    # Extract year
    result["year"] = result[date_col].dt.year

    # Days since reference date
    if reference_date is not None:
        result["days_since_reference"] = (result[date_col] - reference_date).dt.days

    logger.info(f"Engineered temporal features from {date_col}")
    return result


def engineer_market_microstructure_features(
    df: pd.DataFrame,
    price_col: str = "last_price",
    high_col: str = "high_price",
    low_col: str = "low_price",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Engineer market microstructure features (volatility, momentum, moving averages).

    Args:
        df: Input DataFrame
        price_col: Name of price column
        high_col: Name of high price column (for range calculation)
        low_col: Name of low price column (for range calculation)
        group_col: Optional grouping column (e.g., ticker) for time-series features

    Returns:
        DataFrame with market microstructure features added
    """
    result = df.copy()

    if price_col not in df.columns:
        logger.warning(
            f"Price column '{price_col}' not found, skipping market microstructure features"
        )
        return result

    # Price range indicator (requires high and low prices)
    if high_col in df.columns and low_col in df.columns:
        price_range = df[high_col] - df[low_col]
        result["price_range_pct"] = _safe_div(price_range, df[price_col]) * 100

    # Time-series features (volatility, momentum, moving averages)
    if group_col and group_col in df.columns:
        # Historical volatility (30, 60, 90 day rolling windows)
        for window in [30, 60, 90]:
            result[f"volatility_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.pct_change()
                .rolling(window=window, min_periods=max(1, window // 2))
                .std()
                * 100
            )

        # Momentum (rate of change over 20 days)
        result["momentum_20d"] = df.groupby(group_col)[price_col].transform(
            lambda x: x.pct_change(periods=20) * 100
        )

        # Moving averages (20, 50 day)
        for window in [20, 50]:
            result[f"ma_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).mean()
            )
    else:
        # Without grouping, calculate simple rolling features if enough data
        if len(df) >= 30:
            for window in [30, 60, 90]:
                if len(df) >= window:
                    result[f"volatility_{window}d"] = (
                        df[price_col]
                        .pct_change()
                        .rolling(window=window, min_periods=window // 2)
                        .std()
                        * 100
                    )

            if len(df) >= 20:
                result["momentum_20d"] = df[price_col].pct_change(periods=20) * 100

            for window in [20, 50]:
                if len(df) >= window:
                    result[f"ma_{window}d"] = (
                        df[price_col].rolling(window=window, min_periods=window // 2).mean()
                    )

    logger.info("Engineered market microstructure features")
    return result


def engineer_nonlinear_transforms(
    df: pd.DataFrame,
    log_features: Optional[List[str]] = None,
    sqrt_features: Optional[List[str]] = None,
    inverse_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply non-linear transformations to features.

    Args:
        df: Input DataFrame
        log_features: Features to apply natural log transformation (for skewed distributions)
        sqrt_features: Features to apply square root transformation
        inverse_features: Features to apply inverse transformation (1/x)

    Returns:
        DataFrame with non-linear transformed features added
    """
    result = df.copy()

    # Log transformation (natural log)
    if log_features:
        for feature in log_features:
            if feature in df.columns:
                # Only apply log to positive values
                result[f"log_{feature}"] = df[feature].apply(
                    lambda x: np.log(x) if x > 0 else np.nan
                )

    # Square root transformation
    if sqrt_features:
        for feature in sqrt_features:
            if feature in df.columns:
                # Only apply sqrt to non-negative values
                result[f"sqrt_{feature}"] = df[feature].apply(
                    lambda x: np.sqrt(x) if x >= 0 else np.nan
                )

    # Inverse transformation (1/x)
    if inverse_features:
        for feature in inverse_features:
            if feature in df.columns:
                result[f"inv_{feature}"] = _safe_div(pd.Series([1.0] * len(df)), df[feature])

    logger.info(f"Applied non-linear transforms")
    return result


def create_feature_interactions(
    df: pd.DataFrame, features: Optional[List[str]] = None, max_degree: int = 2
) -> pd.DataFrame:
    """Create polynomial and interaction features.

    Args:
        df: Input DataFrame
        features: Features to create interactions for (default: key financial metrics)
        max_degree: Maximum polynomial degree (default: 2)

    Returns:
        DataFrame with interaction features added
    """
    result = df.copy()

    if features is None:
        # Default key features for interactions
        features = ["market_cap", "p_e_ratio", "roe", "debt_to_equity", "revenue_growth_yoy"]
        features = [f for f in features if f in df.columns]

    if len(features) == 0:
        logger.warning("No features available for interactions")
        return result

    # Create pairwise interactions (requires at least 2 features)
    if len(features) >= 2:
        for i, feat1 in enumerate(features):
            for feat2 in features[i + 1 :]:
                interaction_name = f"{feat1}_x_{feat2}"
                result[interaction_name] = df[feat1] * df[feat2]
    else:
        logger.warning("Not enough features for pairwise interactions (need 2+)")

    # Create polynomial features if degree > 1
    if max_degree >= 2:
        for feat in features:
            result[f"{feat}_squared"] = df[feat] ** 2

    logger.info(f"Created {len(result.columns) - len(df.columns)} interaction features")
    return result


def create_relative_value_features(
    df: pd.DataFrame, sector_col: str = "sector", metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create relative value features (deviations from sector median).

    Args:
        df: Input DataFrame
        sector_col: Name of sector column
        metrics: Metrics to compute relative values for

    Returns:
        DataFrame with relative value features added
    """
    result = df.copy()

    if sector_col not in df.columns:
        logger.warning(f"Sector column '{sector_col}' not found")
        return result

    if metrics is None:
        metrics = ["p_e_ratio", "p_b_ratio", "roe", "net_margin_pct", "debt_to_equity"]
        metrics = [m for m in metrics if m in df.columns]

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Calculate sector median
        sector_median = df.groupby(sector_col)[metric].transform("median")

        # Z-score relative to sector
        sector_mean = df.groupby(sector_col)[metric].transform("mean")
        sector_std = df.groupby(sector_col)[metric].transform("std")

        result[f"{metric}_vs_sector_median"] = df[metric] - sector_median
        result[f"{metric}_sector_zscore"] = _safe_div(df[metric] - sector_mean, sector_std)

        # Percentile rank within sector
        result[f"{metric}_sector_percentile"] = df.groupby(sector_col)[metric].rank(pct=True) * 100

    logger.info(f"Created relative value features for {len(metrics)} metrics")
    return result


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


def build_comprehensive_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_relative_values: bool = True,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build comprehensive feature set with all advanced features.

    This is the main function that orchestrates all feature engineering steps.

    Args:
        df: Input DataFrame
        include_interactions: Whether to include interaction features
        include_relative_values: Whether to include relative value features
        sector_col: Name of sector column

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Building comprehensive feature set...")

    result = df.copy()

    # Step 1: Core financial ratios
    result = engineer_valuation_ratios(result)
    result = engineer_profitability_ratios(result)
    result = engineer_leverage_ratios(result)
    result = engineer_liquidity_ratios(result)
    result = engineer_efficiency_ratios(result)
    result = engineer_growth_metrics(result)

    # Step 2: Sector-specific features
    result = engineer_sector_specific_features(result, sector_col=sector_col)

    # Step 3: Interaction features (optional)
    if include_interactions:
        result = create_feature_interactions(result)

    # Step 4: Relative value features (optional)
    if include_relative_values:
        result = create_relative_value_features(result, sector_col=sector_col)

    n_new_features = len(result.columns) - len(df.columns)
    logger.info(f"✓ Feature engineering complete: Added {n_new_features} new features")

    return result
