"""
Classification Models Module - Phase 9.4.1

Comprehensive model training, data preparation, and sampling functions for classification tasks.
Extracted from classification.py with Phase 9.3 feature integration.

Functions:
- Data preparation: prepare_classification_data, _prepare_categorical_features
- Model training: train_xgboost_classifier, train_lightgbm_classifier, train_catboost_classifier,
  train_svm_classifier, train_neural_network_classifier
- Ensemble methods: train_voting_classifier, train_stacking_classifier
- Sampling: apply_smote, apply_adasyn, apply_undersampling, apply_combined_sampling
- Utilities: export_classification_features, clean_extreme_values, validate_data_quality
- Comparison: compare_classifiers
- Orchestrator: fit_classifier (high-level API)

Author: Finance ML Team
Date: 2025-11-09
Version: Phase 9.4.1
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Optional imports with fallback handling
try:
    import xgboost as xgb

    HAVE_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore
    HAVE_XGBOOST = False

try:
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore
    HAVE_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier

    HAVE_CATBOOST = True
except ImportError:
    CatBoostClassifier = None  # type: ignore
    HAVE_CATBOOST = False

try:
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    HAVE_IMBLEARN = True
except ImportError:
    SMOTE = None  # type: ignore
    ADASYN = None  # type: ignore
    RandomUnderSampler = None  # type: ignore
    ImbPipeline = None  # type: ignore
    HAVE_IMBLEARN = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAVE_TENSORFLOW = True
except ImportError:
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    HAVE_TENSORFLOW = False

logger = logging.getLogger(__name__)


# ============================================================================
# Data Preparation Functions
# ============================================================================


def _ensure_numeric_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Ensure all columns in DataFrame have numeric dtypes for compatibility with gradient boosting models.

    LightGBM, XGBoost, and CatBoost require numeric dtypes (int, float, bool).
    This function handles object/string columns by attempting conversion and validates the result.

    Args:
        df: Input DataFrame
        verbose: Print information about transformations

    Returns:
        DataFrame with all numeric dtypes

    Raises:
        ValueError: If non-numeric columns remain after conversion
    """
    df_processed = df.copy()

    # Identify non-numeric columns
    object_cols = df_processed.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = df_processed.select_dtypes(include=["datetime64"]).columns.tolist()

    if verbose and (object_cols or datetime_cols):
        logger.info(f"Found {len(object_cols)} object columns requiring conversion")
        logger.info(f"Found {len(datetime_cols)} datetime columns requiring conversion")

    # Drop datetime columns (should have been handled earlier, but safety check)
    if datetime_cols:
        logger.warning(f"Dropping {len(datetime_cols)} datetime columns: {datetime_cols}")
        df_processed = df_processed.drop(columns=datetime_cols)

    # Handle object columns - should already be one-hot encoded, but validate
    if object_cols:
        logger.warning(f"Found unexpected object columns after encoding: {object_cols[:5]}")
        # Attempt to convert to numeric
        for col in object_cols:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
            except Exception as e:
                logger.error(f"Failed to convert column {col} to numeric: {e}")
                # Drop the column as last resort
                df_processed = df_processed.drop(columns=[col])

    # Ensure all columns are numeric (int, float, bool)
    df_processed = df_processed.astype(float)

    # Handle infinite values
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with 0 (or median could be used)
    df_processed = df_processed.fillna(0.0)

    # Final validation
    non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns remain after conversion: {non_numeric}")

    if verbose:
        logger.info(f"Dtype validation passed. Final shape: {df_processed.shape}")

    return df_processed


def _prepare_categorical_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare categorical features using one-hot encoding (pd.get_dummies).

    This function replaces LabelEncoder with one-hot encoding for robust handling
    of categorical variables, including unseen categories in test set.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        categorical_cols: List of categorical column names

    Returns:
        Tuple of (X_train_encoded, X_test_encoded) with one-hot encoded categoricals
    """
    if not categorical_cols:
        return X_train.copy(), X_test.copy()

    # Separate numeric and categorical columns
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # One-hot encode training set
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)

    # One-hot encode test set
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

    # Align test set columns with training set
    # Add missing columns (fill with 0)
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0

    # Remove extra columns in test set that weren't in training
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    return X_train_encoded, X_test_encoded


def prepare_classification_data(
    df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_groups: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """Prepare data for classification with train/test split.

    Phase 9.3 Integration: Supports optional feature group selection for Phase 9.3 features.

    Args:
        df: DataFrame with features
        labels: Target labels
        test_size: Proportion of test set
        random_state: Random seed
        feature_groups: Optional list of Phase 9.3 feature groups to include:
            - 'analyst_quality': Analyst coverage and rating features
            - 'accounting_quality': Accounting quality indicators
            - 'employee_productivity': Employee productivity metrics
            If None, includes all available features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)
    """
    # Drop non-feature columns
    drop_cols = [
        "ticker",
        "isin",
        "name",
        "description",
        "price_target",
        "last_updated",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Phase 9.3: Detect feature groups if feature selection requested
    if feature_groups is not None:
        # Define Phase 9.3 feature patterns
        analyst_cols = [
            c
            for c in X.columns
            if any(
                k in c.lower()
                for k in [
                    "analyst_coverage",
                    "analyst_consensus",
                    "price_target_spread",
                    "rating_buy_ratio",
                    "rating_sell_ratio",
                ]
            )
        ]

        accounting_cols = [
            c
            for c in X.columns
            if any(
                k in c.lower()
                for k in [
                    "exceptional_items_intensity",
                    "goodwill_intensity",
                    "intangibles_ratio",
                    "accounting_quality_score",
                ]
            )
        ]

        employee_cols = [
            c
            for c in X.columns
            if any(
                k in c.lower()
                for k in [
                    "revenue_per_employee",
                    "profit_per_employee",
                    "assets_per_employee",
                    "employee_growth_rate",
                ]
            )
        ]

        # Build list of columns to keep based on feature_groups
        phase93_cols = []
        if "analyst_quality" in feature_groups:
            phase93_cols.extend(analyst_cols)
        if "accounting_quality" in feature_groups:
            phase93_cols.extend(accounting_cols)
        if "employee_productivity" in feature_groups:
            phase93_cols.extend(employee_cols)

        # Keep only selected Phase 9.3 features plus base features
        all_phase93 = analyst_cols + accounting_cols + employee_cols
        keep_cols = [c for c in X.columns if c not in all_phase93 or c in phase93_cols]
        X = X[keep_cols]

        logger.info(
            f"Phase 9.3 feature selection: kept {len(phase93_cols)} features from groups {feature_groups}"
        )

    # Handle duplicate columns
    if X.columns.duplicated().any():
        logger.warning(f"Removing {X.columns.duplicated().sum()} duplicate columns")
        X = X.loc[:, ~X.columns.duplicated(keep="first")]

    # Identify numeric and categorical columns using dtype-aware logic
    # Categorical: object or category dtypes
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric: number / bool dtypes; ensure they are not in categorical_cols
    numeric_cols = (
        X.select_dtypes(include=[np.number, "bool"]).columns.difference(categorical_cols).tolist()
    )

    # Fill NaN values
    for col in numeric_cols:
        # Use median imputation for numeric columns
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        # Robust handling for categorical features:
        # - If dtype is pandas Categorical, add "Unknown" to categories before fillna.
        # - Otherwise, coerce to string dtype and fill missing with "Unknown".
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            # Ensure "Unknown" is a valid category before using it as fill value
            if "Unknown" not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories(["Unknown"])
            X[col] = X[col].fillna("Unknown")
        else:
            # Coerce non-categorical types (e.g., object or StringDtype) to string,
            # then safely fill missing values with "Unknown"
            X[col] = X[col].astype("string").fillna("Unknown")

    # ------------------------------------------------------------------
    # Train-test split with shared Phase 9.9 policy where possible.
    #
    # For temporal / grouped data the policy prioritizes:
    #   1) time-aware split by snapshot_date
    #   2) grouped split by ticker
    #   3) stratified split by sector
    #   4) random split as a final fallback
    #
    # When none of these columns are present, we fall back to a pure
    # label-stratified split using sklearn.train_test_split to preserve
    # class balance semantics from earlier versions.
    # ------------------------------------------------------------------
    from finance_ml.ml_workflow.validation.splits import create_train_test_split

    # Attach labels so the helper can operate on the full dataframe
    df_for_split = X.copy()
    df_for_split["__labels__"] = labels

    date_col = "snapshot_date" if "snapshot_date" in df_for_split.columns else None
    group_col = "ticker" if "ticker" in df.columns else None
    stratify_col = "sector" if "sector" in df.columns else None

    if date_col or group_col or stratify_col:
        train_df, test_df = create_train_test_split(
            df_for_split,
            date_col=date_col,
            group_col=group_col,
            stratify_col=stratify_col,
            test_size=test_size,
            random_state=random_state,
        )

        X_train = train_df.drop(columns=["__labels__"])
        X_test = test_df.drop(columns=["__labels__"])
        y_train = train_df["__labels__"].to_numpy()
        y_test = test_df["__labels__"].to_numpy()
    else:
        # Fallback: label-stratified split as in the original implementation
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

    logger.info(
        f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples, "
        f"Features: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical"
    )

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


# ============================================================================
# Utility Functions
# ============================================================================


def export_classification_features(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Export classification probabilities as meta-features for regression.

    Args:
        df: Original DataFrame
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names for classes

    Returns:
        DataFrame with added classification meta-features
    """
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]

    # Validate input shapes for safer usage in edge cases
    if y_proba is None:
        raise ValueError("y_proba must not be None")
    if y_proba.ndim != 2:
        raise ValueError(
            f"y_proba must be 2D array of shape (n_samples, n_classes); got ndim={y_proba.ndim}"
        )
    if len(df) != y_proba.shape[0]:
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but y_proba has {y_proba.shape[0]} samples"
        )

    df_with_features = df.copy()

    # Add probability columns (multiple naming conventions for compatibility)
    for i, class_name in enumerate(class_names):
        if i < y_proba.shape[1]:
            df_with_features[f"event_prob_{class_name.lower()}"] = y_proba[:, i]
            df_with_features[f"prob_class_{i}"] = y_proba[:, i]
            df_with_features[f"class_{i}_proba"] = y_proba[:, i]  # Test compatibility

    # Add predicted class
    df_with_features["event_class_predicted"] = np.argmax(y_proba, axis=1)

    # Add confidence (max probability)
    df_with_features["event_confidence"] = np.max(y_proba, axis=1)

    logger.info(
        f"Added {min(len(class_names), y_proba.shape[1]) * 3 + 2} classification meta-features"
    )

    return df_with_features


def clean_extreme_values(df: pd.DataFrame, clip_threshold: float = 1e8) -> pd.DataFrame:
    """Remove infinities and clip extreme values for numerical stability.

    This function provides robust preprocessing for financial data that may contain
    infinities (from division by zero) or extreme outliers.

    Args:
        df: DataFrame to clean
        clip_threshold: Maximum absolute value threshold (default: 1e8)

    Returns:
        Cleaned DataFrame with no infinities and clipped extreme values
    """
    df_clean = df.copy()

    # Replace infinities with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Clip extreme values for numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Calculate 99th percentile for adaptive clipping
            col_abs_max = df_clean[col].abs().quantile(0.99)
            if col_abs_max > clip_threshold:
                clip_value = clip_threshold
            else:
                clip_value = col_abs_max * 10  # Allow 10x the 99th percentile

            df_clean[col] = df_clean[col].clip(-clip_value, clip_value)

    # Impute remaining NaN values with median
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean


def validate_data_quality(
    X: pd.DataFrame, feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate data quality and report issues.

    Args:
        X: Input dataframe to validate
        feature_names: Optional list of feature names to validate (defaults to all columns)

    Returns:
        Dict with validation results:
            - 'has_nulls': bool
            - 'has_inf': bool
            - 'has_extreme_values': bool
            - 'issues': list of issue descriptions
            - 'is_valid': bool (True if no issues)
    """
    issues = []
    has_nulls = False
    has_inf = False
    has_extreme_values = False

    cols_to_check = feature_names if feature_names is not None else X.columns

    for col in cols_to_check:
        if col not in X.columns:
            continue

        col_data = X[col]

        # Check for NaN values
        if col_data.isna().any():
            has_nulls = True
            nan_count = col_data.isna().sum()
            issues.append(f"Column {col}: {nan_count} NaN values")

        # Check for infinities
        if np.any(np.isinf(col_data)):
            has_inf = True
            inf_count = np.sum(np.isinf(col_data))
            issues.append(f"Column {col}: {inf_count} infinite values")

        # Check for extremely large values
        max_val = np.nanmax(np.abs(col_data))
        if max_val > 1e10:
            has_extreme_values = True
            issues.append(f"Column {col}: extremely large values (max={max_val:.2e})")

    if issues:
        logger.warning("⚠️ Data Quality Issues Detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return {
        "has_nulls": has_nulls,
        "has_inf": has_inf,
        "has_extreme_values": has_extreme_values,
        "issues": issues,
        "is_valid": len(issues) == 0,
    }


# ============================================================================
# Sampling Methods for Imbalanced Data
# ============================================================================


def apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    sampling_strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE for class imbalance.

    Args:
        X_train: Training features
        y_train: Training labels
        numeric_cols: Numeric columns (SMOTE requires numeric data)
        sampling_strategy: Sampling strategy ('auto', 'minority', or dict)
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not HAVE_IMBLEARN:
        logger.warning("imbalanced-learn not available, returning original data")
        return X_train, y_train

    # SMOTE requires numeric data
    X_numeric = X_train[numeric_cols].copy()

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_numeric, y_train)

    logger.info(
        f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples. "
        f"Class distribution: {np.bincount(y_resampled)}"
    )

    return pd.DataFrame(X_resampled, columns=numeric_cols), y_resampled


def apply_adasyn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    sampling_strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply ADASYN (Adaptive Synthetic Sampling) for class imbalance.

    ADASYN adaptively generates synthetic samples based on density distribution.

    Args:
        X_train: Training features
        y_train: Training labels
        numeric_cols: Numeric columns (ADASYN requires numeric data)
        sampling_strategy: Sampling strategy ('auto', 'minority', or dict)
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not HAVE_IMBLEARN:
        logger.warning("imbalanced-learn not available, returning original data")
        return X_train, y_train

    # ADASYN requires numeric data
    X_numeric = X_train[numeric_cols].copy()

    try:
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X_numeric, y_train)

        logger.info(
            f"ADASYN applied: {len(y_train)} -> {len(y_resampled)} samples. "
            f"Class distribution: {np.bincount(y_resampled)}"
        )

        return pd.DataFrame(X_resampled, columns=numeric_cols), y_resampled
    except ValueError as e:
        logger.warning(f"ADASYN failed: {e}. Returning original data.")
        return X_train[numeric_cols], y_train


def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    strategy: str = "random",
    sampling_strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply under-sampling for majority class reduction.

    Args:
        X_train: Training features
        y_train: Training labels
        numeric_cols: Numeric columns
        strategy: Under-sampling strategy ('random', 'tomek', 'nearmiss')
        sampling_strategy: Sampling strategy ('auto', 'majority', or dict)
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not HAVE_IMBLEARN:
        logger.warning("imbalanced-learn not available, returning original data")
        return X_train, y_train

    X_numeric = X_train[numeric_cols].copy()

    try:
        if strategy == "random":
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        elif strategy == "tomek":
            from imblearn.under_sampling import TomekLinks

            sampler = TomekLinks(sampling_strategy=sampling_strategy)
        elif strategy == "nearmiss":
            from imblearn.under_sampling import NearMiss

            sampler = NearMiss(sampling_strategy=sampling_strategy)
        else:
            logger.warning(f"Unknown under-sampling strategy: {strategy}. Using random.")
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )

        X_resampled, y_resampled = sampler.fit_resample(X_numeric, y_train)

        logger.info(
            f"Under-sampling ({strategy}) applied: {len(y_train)} -> {len(y_resampled)} samples. "
            f"Class distribution: {np.bincount(y_resampled)}"
        )

        return pd.DataFrame(X_resampled, columns=numeric_cols), y_resampled
    except Exception as e:
        logger.warning(f"Under-sampling failed: {e}. Returning original data.")
        return X_train[numeric_cols], y_train


def apply_combined_sampling(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    over_strategy: str = "smote",
    under_strategy: str = "random",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply combined over-sampling and under-sampling.

    Args:
        X_train: Training features
        y_train: Training labels
        numeric_cols: Numeric columns
        over_strategy: Over-sampling strategy ('smote', 'adasyn')
        under_strategy: Under-sampling strategy ('random', 'tomek', 'nearmiss')
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not HAVE_IMBLEARN:
        logger.warning("imbalanced-learn not available, returning original data")
        return X_train, y_train

    X_numeric = X_train[numeric_cols].copy()

    try:
        # Select over-sampler
        if over_strategy == "smote":
            over_sampler = SMOTE(random_state=random_state)
        elif over_strategy == "adasyn":
            over_sampler = ADASYN(random_state=random_state)
        else:
            logger.warning(f"Unknown over-sampling strategy: {over_strategy}. Using SMOTE.")
            over_sampler = SMOTE(random_state=random_state)

        # Select under-sampler
        if under_strategy == "random":
            from imblearn.under_sampling import RandomUnderSampler

            under_sampler = RandomUnderSampler(random_state=random_state)
        elif under_strategy == "tomek":
            from imblearn.under_sampling import TomekLinks

            under_sampler = TomekLinks()
        elif under_strategy == "nearmiss":
            from imblearn.under_sampling import NearMiss

            under_sampler = NearMiss()
        else:
            logger.warning(f"Unknown under-sampling strategy: {under_strategy}. Using random.")
            from imblearn.under_sampling import RandomUnderSampler

            under_sampler = RandomUnderSampler(random_state=random_state)

        # Create pipeline
        pipeline = ImbPipeline([("over", over_sampler), ("under", under_sampler)])
        X_resampled, y_resampled = pipeline.fit_resample(X_numeric, y_train)

        logger.info(
            f"Combined sampling ({over_strategy}+{under_strategy}) applied: "
            f"{len(y_train)} -> {len(y_resampled)} samples. "
            f"Class distribution: {np.bincount(y_resampled)}"
        )

        return pd.DataFrame(X_resampled, columns=numeric_cols), y_resampled
    except Exception as e:
        logger.warning(f"Combined sampling failed: {e}. Returning original data.")
        return X_train[numeric_cols], y_train


def balance_classes(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "auto",
    imbalance_threshold: float = 10.0,
    sampling_strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Automatically remediate class imbalance through resampling.

    Implements Phase 9.4 Task 5: Classification Class Balance Auto-Remediation.
    Aligned with phase_9.4_implementation_plan.md and code_guidelines.md v1.10.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    method : str, default='auto'
        Resampling method:
        - 'auto': Automatically select based on imbalance severity
        - 'smote': SMOTE oversampling
        - 'adasyn': ADASYN oversampling
        - 'undersample': Random undersampling
        - 'combined': Combined over+undersampling
        - 'none': No resampling (passthrough)
    imbalance_threshold : float, default=10.0
        Imbalance ratio threshold to trigger auto-remediation
        (majority_class_count / minority_class_count)
    sampling_strategy : str, default='auto'
        Sampling strategy for imblearn
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_balanced : pd.DataFrame
        Resampled feature matrix with all original columns (numeric and categorical)
    y_balanced : pd.Series
        Resampled target labels

    Notes
    -----
    - Auto method selects SMOTE for moderate imbalance (10:1 to 20:1)
    - Auto method selects combined sampling for severe imbalance (>20:1)
    - Categorical columns are encoded before resampling and restored after
    - Original indices are not preserved (new integer index assigned)

    Examples
    --------
    >>> X_balanced, y_balanced = balance_classes(
    ...     X, y,
    ...     method='auto',
    ...     imbalance_threshold=10
    ... )
    """
    # Ensure y is a pandas Series (handle numpy array input)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target")

    # Calculate class distribution
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        logger.warning("Only one class present, skipping balancing")
        return X, y

    majority_count = class_counts.max()
    minority_count = class_counts.min()
    imbalance_ratio = majority_count / minority_count

    logger.info(
        f"Class balance analysis: {len(class_counts)} classes, "
        f"imbalance ratio: {imbalance_ratio:.2f}:1"
    )

    # Check if balancing is needed
    if imbalance_ratio < imbalance_threshold:
        logger.info(
            f"Imbalance ratio {imbalance_ratio:.2f} below threshold {imbalance_threshold}, "
            f"skipping balancing"
        )
        return X, y

    # Identify categorical columns (object, string, category dtypes)
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Store original column order for reconstruction
    original_columns = X.columns.tolist()

    # Encode categorical columns before resampling
    # Store mappings for restoration after SMOTE
    categorical_encodings: Dict[str, pd.Index] = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        # Use pd.factorize to convert strings to integers
        # Returns (codes, uniques) where codes are integer indices
        codes, uniques = pd.factorize(X_encoded[col], sort=False)
        X_encoded[col] = codes
        categorical_encodings[col] = uniques
        logger.debug(f"Encoded categorical column '{col}': {len(uniques)} unique values")

    if categorical_encodings:
        logger.info(
            f"Encoded {len(categorical_encodings)} categorical columns for resampling: "
            f"{list(categorical_encodings.keys())}"
        )

    # Select all numeric columns (including newly encoded categoricals)
    numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric columns found for resampling, returning original data")
        return X, y

    # Auto-select method based on imbalance severity
    if method == "auto":
        if imbalance_ratio < 20:
            method = "smote"
            logger.info(f"Auto-selected SMOTE for moderate imbalance ({imbalance_ratio:.2f}:1)")
        else:
            method = "combined"
            logger.info(
                f"Auto-selected combined sampling for severe imbalance ({imbalance_ratio:.2f}:1)"
            )

    # Apply selected resampling method
    try:
        if method == "smote":
            X_balanced, y_balanced = apply_smote(
                X_encoded,
                y,
                numeric_cols,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        elif method == "adasyn":
            X_balanced, y_balanced = apply_adasyn(
                X_encoded,
                y,
                numeric_cols,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        elif method == "undersample":
            X_balanced, y_balanced = apply_undersampling(
                X_encoded,
                y,
                numeric_cols,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        elif method == "combined":
            X_balanced, y_balanced = apply_combined_sampling(
                X_encoded,
                y,
                numeric_cols,
                over_strategy="smote",
                under_strategy="random",
                random_state=random_state,
            )
        elif method == "none":
            logger.info("Method='none', skipping balancing")
            return X, y
        else:
            logger.warning(f"Unknown method '{method}', skipping balancing")
            return X, y

        # Restore categorical columns from encoded values
        # SMOTE may produce fractional values, so round to nearest integer index
        for col, uniques in categorical_encodings.items():
            if col in X_balanced.columns:
                # Round to nearest integer and clip to valid range
                encoded_values = X_balanced[col].values
                rounded_indices = np.clip(np.round(encoded_values).astype(int), 0, len(uniques) - 1)
                # Map back to original string values
                X_balanced[col] = uniques[rounded_indices]
                logger.debug(f"Restored categorical column '{col}' to string dtype")

        if categorical_encodings:
            logger.info(
                f"Restored {len(categorical_encodings)} categorical columns to original dtypes"
            )

        # Ensure column order matches original (only include columns that exist)
        final_columns = [col for col in original_columns if col in X_balanced.columns]
        X_balanced = X_balanced[final_columns]

        # Convert to Series if numpy array returned
        if isinstance(y_balanced, np.ndarray):
            y_balanced = pd.Series(y_balanced, name=y.name if hasattr(y, "name") else "target")

        logger.info(
            f"Class balancing complete: {len(y)} -> {len(y_balanced)} samples, "
            f"new ratio: {y_balanced.value_counts().max() / y_balanced.value_counts().min():.2f}:1"
        )

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Class balancing failed: {e}, returning original data")
        return X, y


def determine_cv_strategy(
    df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    n_splits: int = 5,
    date_column: str = "snapshot_date",
    group_column: str = "ticker",
    random_state: int = 42,
) -> Tuple[str, Any]:
    """
    Determine appropriate CV strategy based on data characteristics.

    Implements Phase 9.4 Task 4: Cross-Validation Policy Enforcement.
    Aligned with phase_9.4_implementation_plan.md and code_guidelines.md v1.10.

    Hierarchy:
    1. time_series: if date_column exists → TimeSeriesSplit
    2. grouped: if group_column exists → GroupKFold
    3. stratified: if target is categorical → StratifiedKFold
    4. kfold: fallback → KFold

    Parameters
    ----------
    df : pd.DataFrame
        Data with potential date/group columns
    target : pd.Series, optional
        Target variable for stratification
    n_splits : int, default=5
        Number of CV splits
    date_column : str, default='snapshot_date'
        Column name for time-series ordering
    group_column : str, default='ticker'
        Column name for group-based splitting
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    cv_strategy : str
        Selected strategy name ('time_series', 'grouped', 'stratified', 'kfold')
    cv_object : sklearn CV splitter
        Configured cross-validator object

    Notes
    -----
    - Prevents look-ahead bias in backtesting
    - Grouped CV prevents data leakage across same entity (e.g., ticker)
    - Falls back gracefully if preferred strategy unavailable

    Examples
    --------
    >>> cv_strategy, cv_obj = determine_cv_strategy(
    ...     df,
    ...     n_splits=5
    ... )
    >>> for train_idx, test_idx in cv_obj.split(df):
    ...     # Train/test split without leakage
    ...     pass
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    # Priority 1: Time-series split if date column exists
    if date_column in df.columns:
        logger.info(
            f"CV Strategy: time_series (detected '{date_column}' column for temporal ordering)"
        )
        return "time_series", TimeSeriesSplit(n_splits=n_splits)

    # Priority 2: Grouped split if group column exists (prevents ticker leakage)
    if group_column in df.columns:
        unique_groups = df[group_column].nunique()
        if unique_groups >= n_splits:
            logger.info(
                f"CV Strategy: grouped (detected '{group_column}' column with "
                f"{unique_groups} unique groups, preventing entity leakage)"
            )
            return "grouped", GroupKFold(n_splits=n_splits)
        else:
            logger.warning(
                f"Only {unique_groups} unique groups in '{group_column}', "
                f"cannot create {n_splits} splits. Falling back."
            )

    # Priority 3: Stratified split if target is categorical
    if target is not None:
        try:
            # Check if target is categorical (discrete classes)
            unique_classes = target.nunique()
            if unique_classes < len(target) / 2:  # Heuristic: likely categorical
                min_class_count = target.value_counts().min()
                if min_class_count >= n_splits:
                    logger.info(
                        f"CV Strategy: stratified (detected {unique_classes} classes, "
                        f"maintaining class balance)"
                    )
                    return "stratified", StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=random_state
                    )
                else:
                    logger.warning(
                        f"Smallest class has only {min_class_count} samples, "
                        f"cannot stratify with {n_splits} splits. Falling back to KFold."
                    )
        except Exception as e:
            logger.warning(f"Stratification check failed: {e}. Falling back to KFold.")

    # Priority 4: Simple KFold fallback
    logger.info("CV Strategy: kfold (fallback, no date/group/stratification available)")
    return "kfold", KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ============================================================================
# Model Training Functions
# ============================================================================


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional XGBoost parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_XGBOOST:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")

    # Default parameters
    default_params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
    }
    if params:
        default_params.update(params)

    # Prepare data: encode categoricals and scale numerics
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Clean extreme values before scaling
    logger.info("Cleaning extreme values and infinities...")
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    # Ensure numeric dtypes for XGBoost compatibility
    X_train_proc = _ensure_numeric_dtypes(X_train_proc, verbose=False)
    X_test_proc = _ensure_numeric_dtypes(X_test_proc, verbose=False)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_proc),
        columns=X_train_proc.columns,
        index=X_train_proc.index,
    )
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_proc),
        columns=X_test_proc.columns,
        index=X_test_proc.index,
    )

    # Train model
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)], verbose=False)

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,  # Backward compatibility
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train_proc.columns, model.feature_importances_)),
    }


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train LightGBM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional LightGBM parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_LIGHTGBM:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")

    # Default parameters
    default_params = {
        "objective": "multiclass",
        "num_class": 5,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    # Prepare data
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Clean extreme values before scaling
    logger.info("Cleaning extreme values and infinities...")
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    # Ensure numeric dtypes for LightGBM compatibility (critical fix for dtype errors)
    X_train_proc = _ensure_numeric_dtypes(X_train_proc, verbose=False)
    X_test_proc = _ensure_numeric_dtypes(X_test_proc, verbose=False)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_proc),
        columns=X_train_proc.columns,
        index=X_train_proc.index,
    )
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_proc),
        columns=X_test_proc.columns,
        index=X_test_proc.index,
    )

    # Train model
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)])

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"LightGBM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train_proc.columns, model.feature_importances_)),
    }


def train_catboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train CatBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional CatBoost parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_CATBOOST:
        raise ImportError("CatBoost not available. Install with: pip install catboost")

    # Default parameters
    default_params = {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "MultiClass",
        "random_seed": 42,
        "verbose": False,
    }
    if params:
        default_params.update(params)

    # Prepare data
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Clean extreme values before scaling
    logger.info("Cleaning extreme values and infinities...")
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    # Ensure numeric dtypes for CatBoost compatibility
    X_train_proc = _ensure_numeric_dtypes(X_train_proc, verbose=False)
    X_test_proc = _ensure_numeric_dtypes(X_test_proc, verbose=False)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_proc),
        columns=X_train_proc.columns,
        index=X_train_proc.index,
    )
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_proc),
        columns=X_test_proc.columns,
        index=X_test_proc.index,
    )

    # Train model
    model = CatBoostClassifier(**default_params)
    model.fit(X_train_proc, y_train, eval_set=(X_test_proc, y_test))

    # Predictions
    y_pred = model.predict(X_test_proc).flatten().astype(int)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"CatBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train_proc.columns, model.feature_importances_)),
    }


def train_svm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    kernel: str = "rbf",
    **kwargs,
) -> Dict[str, Any]:
    """Train SVM classifier (simplified implementation).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        kernel: Kernel type
        **kwargs: Additional SVM parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Prepare data
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    # Train
    model = SVC(kernel=kernel, probability=True, random_state=42, **kwargs)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"SVM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_neural_network_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train Neural Network classifier (simplified implementation).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional NN parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_TENSORFLOW:
        logger.warning("TensorFlow not available, using RandomForest as fallback")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)
        X_train_proc = clean_extreme_values(X_train_proc)
        X_test_proc = clean_extreme_values(X_test_proc)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_proc)
        X_test_scaled = scaler.transform(X_test_proc)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        return {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # TensorFlow implementation (simplified)
    from sklearn.preprocessing import StandardScaler

    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    # Build simple model
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_scaled,
        y_train,
        epochs=30,
        batch_size=32,
        verbose=0,
        validation_split=0.2,
    )

    y_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Neural Network - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_voting_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    voting: Literal["hard", "soft"] = "soft",
) -> Dict[str, Any]:
    """Train Voting ensemble classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        voting: Voting strategy ('hard' or 'soft')

    Returns:
        Dictionary with model, predictions, and metrics
    """
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    # Create base estimators
    estimators = []
    if HAVE_XGBOOST:
        estimators.append(
            ("xgb", xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0))
        )
    if HAVE_LIGHTGBM:
        estimators.append(
            ("lgbm", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        )
    estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))

    # Train voting classifier
    model = VotingClassifier(estimators=estimators, voting=voting)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled) if voting == "soft" else None

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Voting Classifier - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_stacking_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Train Stacking ensemble classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names

    Returns:
        Dictionary with model, predictions, and metrics
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)
    X_train_proc = clean_extreme_values(X_train_proc)
    X_test_proc = clean_extreme_values(X_test_proc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    # Create base estimators
    estimators = []
    if HAVE_XGBOOST:
        estimators.append(
            ("xgb", xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0))
        )
    if HAVE_LIGHTGBM:
        estimators.append(
            ("lgbm", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        )
    estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))

    # Train stacking classifier
    model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=3
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Stacking Classifier - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compare_classifiers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple classifiers.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names

    Returns:
        Dictionary mapping model names to their results with consistent metric schema.
        Each result dict is guaranteed to have keys: 'model', 'accuracy', 'f1_macro',
        'f1_weighted', 'precision_macro', 'recall_macro'
    """
    results = {}

    # Train available models
    if HAVE_XGBOOST:
        try:
            results["XGBoost"] = train_xgboost_classifier(
                X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
            )
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")

    if HAVE_LIGHTGBM:
        try:
            results["LightGBM"] = train_lightgbm_classifier(
                X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
            )
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")

    if HAVE_CATBOOST:
        try:
            results["CatBoost"] = train_catboost_classifier(
                X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
            )
        except Exception as e:
            logger.warning(f"CatBoost training failed: {e}")

    # Standardize metric keys to ensure consistent schema across all models
    # FIX 4: Add missing f1_macro, f1_weighted, precision_macro, recall_macro keys
    for model_name, result in results.items():
        # Add f1_macro (alias for f1_score which is computed with average='macro')
        if "f1_macro" not in result and "f1_score" in result:
            result["f1_macro"] = result["f1_score"]

        # Compute f1_weighted if not present
        if "f1_weighted" not in result and "y_pred" in result:
            from sklearn.metrics import f1_score

            result["f1_weighted"] = f1_score(
                y_test, result["y_pred"], average="weighted", zero_division=0
            )

        # Add precision_macro (alias for precision which is computed with average='macro')
        if "precision_macro" not in result and "precision" in result:
            result["precision_macro"] = result["precision"]

        # Add recall_macro (alias for recall which is computed with average='macro')
        if "recall_macro" not in result and "recall" in result:
            result["recall_macro"] = result["recall"]

    logger.info(f"Compared {len(results)} classifiers with standardized metrics")
    return results


# ============================================================================
# High-Level Orchestrator (Phase 9.4.1)
# ============================================================================


def fit_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[np.ndarray] = None,
    model: Union[str, List[str]] = "xgboost",
    params: Optional[Dict[str, Any]] = None,
    tuning: Optional[Dict[str, Any]] = None,
    cv: Optional[Dict[str, Any]] = None,
    class_weighting: Optional[str] = None,
    feature_groups: Optional[List[str]] = None,
    compare: bool = False,
) -> Dict[str, Any]:
    """High-level classifier training orchestrator with integrated tuning and CV.

    This is the unified API for classification model training introduced in Phase 9.4.1.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features
        y_test: Optional test labels
        model: Model type ('xgboost', 'lightgbm', 'catboost', 'svm', 'neural_network') or list for comparison
        params: Optional model parameters
        tuning: Optional hyperparameter tuning config (e.g., {'n_trials': 50, 'cv_folds': 3})
        cv: Optional cross-validation config (e.g., {'sector_stratified': True, 'cv_folds': 5})
        class_weighting: Optional class weighting strategy ('balanced', 'auto')
        feature_groups: Optional Phase 9.3 feature groups to select
        compare: If True and model is a list, compare multiple models

    Returns:
        Standardized dict with keys: {model, metrics, y_pred, y_proba, artifacts}
    """
    # Phase 9.3: Prepare data with feature group selection
    if feature_groups is not None:
        # Use prepare_classification_data's feature_groups parameter

        # Reconstruct full dataset temporarily
        if X_test is not None and y_test is not None:
            X_full = pd.concat([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test])
        else:
            X_full = X_train
            y_full = y_train

        (
            X_train_prep,
            X_test_prep,
            y_train_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
        ) = prepare_classification_data(
            X_full,
            y_full,
            test_size=0.2 if X_test is None else len(X_test) / len(X_full),
            random_state=42,
            feature_groups=feature_groups,
        )
    else:
        # Standard preparation
        if X_test is None or y_test is None:
            # Need to split
            (
                X_train_prep,
                X_test_prep,
                y_train_prep,
                y_test_prep,
                numeric_cols,
                categorical_cols,
            ) = prepare_classification_data(X_train, y_train, test_size=0.2, random_state=42)
        else:
            # Already split
            numeric_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
            categorical_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
            X_train_prep, X_test_prep, y_train_prep, y_test_prep = (
                X_train,
                X_test,
                y_train,
                y_test,
            )

    # Handle class weighting
    if class_weighting == "balanced":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train_prep)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train_prep)
        if params is None:
            params = {}
        # Different models use different parameter names
        if isinstance(model, str):
            if model in ["xgboost", "lightgbm", "catboost"]:
                params["class_weight"] = dict(zip(classes, class_weights))

    # Hyperparameter tuning
    tuning_result = None
    tuned_model = None
    if tuning is not None:
        from finance_ml.ml_workflow.classification.tuning import (
            optimize_classifier_hyperparameters,
        )

        tuning_result = optimize_classifier_hyperparameters(
            X_train_prep,
            y_train_prep,
            classifier_type=model if isinstance(model, str) else "xgboost",
            n_trials=tuning.get("n_trials", 50),
            cv_folds=tuning.get("cv_folds", 5),
            random_state=42,
            verbose=tuning.get("verbose", True),
        )
        params = tuning_result["best_params"]
        tuned_model = tuning_result["model"]

    # Cross-validation
    cv_results = None
    if cv is not None:
        from finance_ml.ml_workflow.classification.tuning import (
            cross_validate_with_sector_stratification,
        )

        # Build model for CV
        if tuning is None:
            if isinstance(model, str) and model == "xgboost" and HAVE_XGBOOST:
                cv_model = xgb.XGBClassifier(**(params or {}))
            elif isinstance(model, str) and model == "lightgbm" and HAVE_LIGHTGBM:
                cv_model = lgb.LGBMClassifier(**(params or {}))
            else:
                cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            cv_model = tuned_model

        cv_results = cross_validate_with_sector_stratification(
            X_train_prep,
            y_train_prep,
            cv_model,
            sector_col=cv.get("sector_col", "sector") if "sector" in X_train_prep.columns else None,
            cv_folds=cv.get("cv_folds", 5),
            scoring="f1_macro",
            random_state=42,
        )

    # Model comparison mode
    if compare and isinstance(model, list):
        comparison = compare_classifiers(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
        )
        # Return best model
        best_name = max(comparison.keys(), key=lambda k: comparison[k]["metrics"]["f1_score"])
        result = comparison[best_name]
        result["artifacts"] = {
            "comparison": comparison,
            "best_model_name": best_name,
            "feature_groups": feature_groups,
        }
        if cv_results:
            result["artifacts"]["cv_results"] = cv_results
        if tuning:
            result["artifacts"]["tuning_results"] = tuning_result if tuning else None
        return result

    # Single model training
    model_str = (
        model if isinstance(model, str) else model[0] if isinstance(model, list) else "xgboost"
    )

    if model_str == "xgboost" and HAVE_XGBOOST:
        result = train_xgboost_classifier(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
            params,
        )
    elif model_str == "lightgbm" and HAVE_LIGHTGBM:
        result = train_lightgbm_classifier(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
            params,
        )
    elif model_str == "catboost" and HAVE_CATBOOST:
        result = train_catboost_classifier(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
            params,
        )
    elif model_str == "svm":
        result = train_svm_classifier(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
        )
    elif model_str == "neural_network":
        result = train_neural_network_classifier(
            X_train_prep,
            y_train_prep,
            X_test_prep,
            y_test_prep,
            numeric_cols,
            categorical_cols,
            params,
        )
    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Add artifacts
    result["artifacts"] = {
        "feature_groups": feature_groups,
    }
    if cv_results:
        result["artifacts"]["cv_results"] = cv_results
    if tuning:
        result["artifacts"]["tuning_results"] = tuning_result if tuning else None

    return result


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Data preparation
    "prepare_classification_data",
    "_prepare_categorical_features",
    # Utilities
    "export_classification_features",
    "clean_extreme_values",
    "validate_data_quality",
    # Sampling
    "apply_smote",
    "apply_adasyn",
    "apply_undersampling",
    "apply_combined_sampling",
    "balance_classes",
    # Cross-validation
    "determine_cv_strategy",
    # Model training
    "train_xgboost_classifier",
    "train_lightgbm_classifier",
    "train_catboost_classifier",
    "train_svm_classifier",
    "train_neural_network_classifier",
    "train_voting_classifier",
    "train_stacking_classifier",
    # Comparison and orchestration
    "compare_classifiers",
    "fit_classifier",
]
