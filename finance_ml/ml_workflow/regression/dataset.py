"""
Dataset preparation and feature engineering for regression models.

This module provides functions for preparing data for regression training,
including classification feature integration, validation, and sector-specific
preprocessing.

Phase 9.5 - Regression Refactor
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


# ==============================================================================
# Classification Feature Integration
# ==============================================================================


def extract_classification_features(probabilities: np.ndarray) -> pd.DataFrame:
    """
    Extract classification features from predicted probabilities.

    This function converts raw classifier probabilities into structured features
    that can be used as inputs for regression models. The classification features
    provide meta-information about market sentiment and event likelihood.

    Creates DataFrame with 5 columns:
    - event_prob_neutral: Probability of neutral class (class 0, -10% to +10% price change)
    - event_prob_positive: Probability of positive class (class 1, >= +10% upside)
    - event_prob_negative: Probability of negative class (class 2, >= -10% downside)
    - event_class_predicted: Predicted class (0, 1, or 2 based on argmax)
    - event_confidence: Confidence score (max probability across classes)

    Args:
        probabilities: Array of shape (n_samples, 3) with class probabilities
                      from a trained 3-class event classifier

    Returns:
        DataFrame with classification features (n_samples rows, 5 columns)

    Raises:
        ValueError: If probabilities array doesn't have exactly 3 classes

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>>
        >>> # Train event classifier
        >>> classifier = RandomForestClassifier()
        >>> classifier.fit(X_train, y_train)
        >>>
        >>> # Extract classification features for regression
        >>> probs = classifier.predict_proba(X_test)
        >>> features = extract_classification_features(probs)
        >>>
        >>> # Use in regression
        >>> X_regression = pd.concat([X_test, features], axis=1)

    Phase 9.5 Implementation:
        This function enables integration of classification meta-features into
        regression models, as specified in the Phase 9.5 requirements for
        sector-optimized regression with classification feature enhancement.
    """
    if probabilities.shape[1] != 3:
        raise ValueError(f"Expected 3 classes, got {probabilities.shape[1]}")

    n_samples = probabilities.shape[0]
    logger.debug(f"Extracting classification features for {n_samples} samples")

    features = pd.DataFrame(
        {
            "event_prob_neutral": probabilities[:, 0],
            "event_prob_positive": probabilities[:, 1],
            "event_prob_negative": probabilities[:, 2],
            "event_class_predicted": probabilities.argmax(axis=1),
            "event_confidence": probabilities.max(axis=1),
        }
    )

    # Log summary statistics
    avg_confidence = features["event_confidence"].mean()
    class_distribution = features["event_class_predicted"].value_counts()
    logger.debug(
        f"Average classification confidence: {avg_confidence:.3f}, "
        f"Class distribution: {dict(class_distribution)}"
    )

    return features


def integrate_classification_features_into_dataframe(
    df: pd.DataFrame, classification_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrate classification features into main DataFrame.

    This function combines the original stock data DataFrame with the
    classification meta-features, creating a unified dataset suitable for
    training regression models with classification feature enhancement.

    The function:
    1. Resets indices on both DataFrames to ensure proper row alignment
    2. Concatenates horizontally (axis=1)
    3. Returns combined DataFrame with all columns

    Args:
        df: Original DataFrame with stock data (ticker, sector, price_target, etc.)
        classification_features: DataFrame with classification features from
                                extract_classification_features()

    Returns:
        Combined DataFrame with both original and classification features.
        Row count equals len(df), column count equals len(df.columns) + 5

    Raises:
        ValueError: If DataFrames have different row counts (implicit via concat)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Original stock data
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        ...     'sector': ['Tech', 'Tech', 'Tech'],
        ...     'last_price': [150.0, 300.0, 2500.0],
        ...     'price_target': [180.0, 350.0, 2800.0]
        ... })
        >>>
        >>> # Classification features from trained classifier
        >>> probs = np.array([[0.2, 0.7, 0.1], [0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])
        >>> class_features = extract_classification_features(probs)
        >>>
        >>> # Combine for regression
        >>> df_enhanced = integrate_classification_features_into_dataframe(df, class_features)
        >>> print(df_enhanced.columns)
        # ['ticker', 'sector', 'last_price', 'price_target',
        #  'event_prob_neutral', 'event_prob_positive', 'event_prob_negative',
        #  'event_class_predicted', 'event_confidence']

    Phase 9.5 Integration:
        This function is part of the classification feature enhancement pipeline,
        enabling sector-optimized regression models to leverage event classifier
        outputs as meta-features for improved price target prediction.

    Note:
        Both DataFrames must have the same number of rows. The function resets
        indices to avoid alignment issues, so original index values are not preserved.
    """
    # Validate input
    if len(df) != len(classification_features):
        raise ValueError(
            f"DataFrame length mismatch: df has {len(df)} rows, "
            f"classification_features has {len(classification_features)} rows"
        )

    logger.debug(
        f"Integrating {len(classification_features.columns)} classification features "
        f"into DataFrame with {len(df.columns)} original columns"
    )

    # Reset indices to ensure proper alignment
    df_reset = df.reset_index(drop=True)
    features_reset = classification_features.reset_index(drop=True)

    # Concatenate horizontally
    result = pd.concat([df_reset, features_reset], axis=1)

    logger.debug(f"Integration complete: {len(result)} rows, {len(result.columns)} total columns")

    return result


def create_classification_interactions(
    df: pd.DataFrame, classification_cols: List[str], valuation_cols: List[str]
) -> pd.DataFrame:
    """
    Create interaction features between classification probabilities and valuation metrics.

    Args:
        df: Input DataFrame
        classification_cols: Classification feature columns (e.g., event probabilities)
        valuation_cols: Valuation metric columns (e.g., P/E, P/B ratios)

    Returns:
        DataFrame with additional interaction features
    """
    df_enhanced = df.copy()

    # Create pairwise interactions
    for class_col in classification_cols:
        for val_col in valuation_cols:
            interaction_name = f"{class_col}_x_{val_col}"
            df_enhanced[interaction_name] = df[class_col] * df[val_col]

    return df_enhanced


# ==============================================================================
# Regression Data Preparation
# ==============================================================================


def prepare_regression_data(
    df: pd.DataFrame,
    target_col: str = "price_target",
    exclude_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Prepare regression data with classification meta-features.

    Args:
        df: Input DataFrame with features and target
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test, feature_info
    """
    if exclude_cols is None:
        exclude_cols = [target_col, "last_price"]

    # Identify feature types
    classification_features = [
        col
        for col in df.columns
        if col.startswith("event_prob_") or col in ["event_class_predicted", "event_confidence"]
    ]

    # Get all feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Separate numeric and categorical
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if col in classification_features:
            numeric_features.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # Prepare X and y (only numeric features for now)
    X = df[numeric_features].copy()
    y = df[target_col].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature info - CRITICAL FIX: 'all_features' should only contain numeric features
    # to prevent passing non-numeric columns to model training
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classification_features": classification_features,
        "all_features": numeric_features,  # ✓ Only numeric features for training
    }

    return X_train, X_test, y_train, y_test, feature_info


# ==============================================================================
# Data Validation and Preprocessing
# ==============================================================================


def validate_training_data(X: pd.DataFrame, y: pd.Series, strict: bool = True) -> Dict[str, Any]:
    """
    Validate training data before model fitting.

    This function implements Priority 1 from ML Workflow Improvement Plan:
    comprehensive validation gates to prevent NaN/Inf values from reaching model training.

    Args:
        X: Feature matrix
        y: Target vector
        strict: If True, raise exceptions on validation failures

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if data passed all checks
        - nan_features: count of NaN values in features
        - nan_target: count of NaN values in target
        - inf_features: count of infinite values in features
        - inf_target: count of infinite values in target
        - zero_var_columns: list of zero-variance column names
        - issues: list of issue descriptions

    Raises:
        ValueError: If validation fails and strict=True

    Example:
        >>> X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y_train = pd.Series([10, 20, 30])
        >>> result = validate_training_data(X_train, y_train, strict=True)
        >>> assert result['valid'] == True
    """
    issues = []

    # Check for empty data
    if len(X) == 0 or len(y) == 0:
        msg = "Feature matrix X or target vector y is empty"
        if strict:
            raise ValueError(f"{msg}. Cannot train on empty data.")
        issues.append(msg)

    # Check for NaN in features
    nan_count_X = X.isnull().sum().sum()
    if nan_count_X > 0:
        msg = f"Feature matrix X contains {nan_count_X} NaN values"
        if strict:
            raise ValueError(
                f"{msg}. Apply imputation before training. "
                f"Use finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step()"
            )
        issues.append(msg)

    # Check for NaN in target
    nan_count_y = y.isnull().sum()
    if nan_count_y > 0:
        msg = f"Target vector y contains {nan_count_y} NaN values"
        if strict:
            raise ValueError(f"{msg}. Remove or impute target NaN before training.")
        issues.append(msg)

    # Check for infinite values in features
    inf_count_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count_X > 0:
        msg = f"Feature matrix X contains {inf_count_X} infinite values"
        if strict:
            raise ValueError(f"{msg}. Replace infinite values before training.")
        issues.append(msg)

    # Check for infinite values in target
    inf_count_y = np.isinf(y).sum()
    if inf_count_y > 0:
        msg = f"Target vector y contains {inf_count_y} infinite values"
        if strict:
            raise ValueError(f"{msg}. Replace infinite values in target.")
        issues.append(msg)

    # Check for zero-variance columns (warning, not blocker)
    zero_var_cols = X.columns[X.var() == 0].tolist()
    if len(zero_var_cols) > 0:
        msg = f"Feature matrix X contains {len(zero_var_cols)} zero-variance columns: {zero_var_cols[:5]}"
        issues.append(msg)

    return {
        "valid": len(issues) == 0 or (len(issues) == 1 and len(zero_var_cols) > 0),
        "nan_features": nan_count_X,
        "nan_target": nan_count_y,
        "inf_features": inf_count_X,
        "inf_target": inf_count_y,
        "zero_var_columns": zero_var_cols,
        "issues": issues,
    }


def prepare_features_for_training(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    apply_imputation: bool = True,
    sector_column: str = "sector",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training with final imputation checkpoint.

    This function implements Priority 3 from ML Workflow Improvement Plan:
    pre-model training imputation checkpoint to ensure zero NaN values.

    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        apply_imputation: If True, apply 6-step imputation before extraction
        sector_column: Sector column for KNN imputation

    Returns:
        Tuple of (X, y) ready for model training with zero NaN

    Example:
        >>> df = pd.DataFrame({
        ...     'sector': ['Tech', 'Finance'],
        ...     'market_cap': [1e9, np.nan],
        ...     'last_price': [100, 150],
        ...     'price_target': [110, 160]
        ... })
        >>> X, y = prepare_features_for_training(
        ...     df, ['market_cap'], 'price_target',
        ...     apply_imputation=True, sector_column='sector'
        ... )
        >>> assert X.isnull().sum().sum() == 0
    """
    # Import here to avoid circular dependency
    apply_imputation_func = None
    if apply_imputation:
        try:
            from finance_ml.ml_workflow.preprocessing.imputation import (
                apply_enhanced_imputation_strategy_6step,
            )

            apply_imputation_func = apply_enhanced_imputation_strategy_6step
        except ImportError:
            # Fallback to old location (legacy 6-step name)
            try:
                from finance_ml.advanced_preprocessing import (
                    apply_enhanced_imputation_strategy_6step,
                )

                apply_imputation_func = apply_enhanced_imputation_strategy_6step
            except ImportError:
                logger.warning("Could not import imputation function, skipping imputation")
                apply_imputation = False

    # Extract target BEFORE imputation to preserve NaN for removal
    y = df[target_col].copy()

    # Drop rows with NaN in target
    valid_mask = ~y.isnull()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        logger.warning(f"Dropping {n_dropped} rows with NaN target values")
        df = df[valid_mask].copy()
        y = y[valid_mask]

    # Apply final imputation if requested (only on features, target already extracted)
    if apply_imputation and apply_imputation_func is not None:
        logger.info("Applying final imputation before feature extraction...")
        df = apply_imputation_func(
            df,
            sector_column=sector_column,
            n_neighbors=5,
            price_column="last_price" if "last_price" in df.columns else None,
        )

    # Extract features after imputation
    X = df[feature_cols].copy()

    # Final validation - handle any residual NaN/Inf
    nan_X = X.isnull().sum().sum()
    nan_y = y.isnull().sum()
    inf_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    inf_y = np.isinf(y).sum()

    if nan_X > 0 or nan_y > 0 or inf_X > 0 or inf_y > 0:
        logger.error(
            f"Features have {nan_X} NaN, {inf_X} Inf; target has {nan_y} NaN, {inf_y} Inf after preparation"
        )
        # Emergency fallback: fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median() if pd.notna(y.median()) else 0)
        logger.warning("Applied emergency fillna(0) to ensure training can proceed")

    logger.info(f"✓ Features prepared: {X.shape}, target: {y.shape}, zero NaN confirmed")

    return X, y


def extract_numeric_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract numeric feature columns from DataFrame, excluding targets and metadata.

    This utility function identifies all numeric columns in a DataFrame and filters
    out common non-feature columns like identifiers, targets, and event labels.

    Args:
        df: Input DataFrame
        exclude_cols: Explicit list of column names to exclude (default: None)
        exclude_patterns: List of substring patterns to match for exclusion
            (default: ['event_proba_', 'event_label'])

    Returns:
        List of numeric column names suitable for model training

    Default Exclusions:
        - Identifier columns: 'ticker', 'isin', 'name', 'description'
        - Categorical columns: 'sector', 'region', 'industry', 'country'
        - Target columns: 'price_target', 'analyst_target_price'
        - Event-related columns: 'event_label', 'event_proba_*'
        - Any custom columns in exclude_cols parameter

    Examples:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'sector': ['Tech', 'Tech'],
        ...     'last_price': [150.0, 300.0],
        ...     'market_cap': [2.5e12, 2.3e12],
        ...     'price_target': [180.0, 350.0]
        ... })
        >>> features = extract_numeric_feature_columns(df)
        >>> # Returns: ['last_price', 'market_cap']
        >>> # (excludes ticker, sector, price_target)

        >>> # Custom exclusions
        >>> features = extract_numeric_feature_columns(
        ...     df, exclude_cols=['last_price', 'price_target']
        ... )
        >>> # Returns: ['market_cap']
    """
    if df.empty:
        logger.info("DataFrame is empty, returning empty feature list")
        return []

    # Default exclusion set
    default_exclude = {
        # Identifiers
        "ticker",
        "isin",
        "name",
        "description",
        # Categorical grouping columns (even if accidentally numeric)
        "sector",
        "region",
        "industry",
        "country",
        "trading_country",
        # Common target columns
        "price_target",
        "analyst_target_price",
        "price_target_median",
        # Event classification outputs
        "event_label",
    }

    # Combine with user-provided exclusions
    if exclude_cols:
        default_exclude.update(exclude_cols)

    # Default patterns to exclude
    if exclude_patterns is None:
        exclude_patterns = ["event_proba_"]

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"DataFrame analysis: {len(df.columns)} total columns, {len(numeric_cols)} numeric")

    # Filter out excluded columns and patterns
    feature_cols = []
    for col in numeric_cols:
        # Check explicit exclusions
        if col in default_exclude:
            continue

        # Check pattern exclusions
        if any(pattern in col for pattern in exclude_patterns):
            continue

        feature_cols.append(col)

    logger.info(
        f"Extracted {len(feature_cols)} numeric feature columns "
        f"(excluded {len(numeric_cols) - len(feature_cols)} columns)"
    )

    if len(feature_cols) == 0:
        logger.warning("No numeric feature columns found after exclusions")
    else:
        logger.debug(
            f"Feature columns: {feature_cols[:10]}"
            + (f" ... and {len(feature_cols) - 10} more" if len(feature_cols) > 10 else "")
        )

    return feature_cols


# ==============================================================================
# Sector-Specific Training
# ==============================================================================


def train_sector_specific_models(
    df: pd.DataFrame,
    feature_cols: Union[List[str], Dict[str, List[str]]],
    target_col: str,
    sector_col: str = "sector",
    model_type: str = "random_forest",
    random_state: int = 42,
    min_samples: int = 20,
    ensure_nonnegative: bool = False,
    auto_extract_fallback: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train separate regression models for each sector.

    Args:
        df: Input DataFrame
        feature_cols: Feature column names. Accepts a list of column names or a dict
            with keys like 'all_features', 'numeric_features', 'categorical_features',
            and 'classification_features'. If a dict is provided, the function will
            try 'all_features' first, otherwise combine available groups.
        target_col: Target column name
        sector_col: Sector column name
        model_type: Model type to train ('random_forest' or 'ridge')
        random_state: Random seed
        min_samples: Minimum samples required per sector (default: 20)
        ensure_nonnegative: If True, wrap models with NonNegativeRegressionWrapper
                           to ensure predictions >= 0
        auto_extract_fallback: If True, automatically extract numeric features from
                              DataFrame when provided feature_cols are invalid or missing.
                              Uses extract_numeric_feature_columns() to identify suitable
                              features (default: False)

    Returns:
        Tuple of (sector_models, results):
        - sector_models: Dictionary mapping sector names to trained models
        - results: Dictionary with metrics and metadata

    Raises:
        ValueError: If no valid features remain after validation against df and
                   auto_extract_fallback is False
    """
    # Import model training functions here to avoid circular dependency
    try:
        from finance_ml.ml_workflow.regression.models import (
            train_random_forest_regressor,
            train_ridge_regressor,
        )
    except ImportError:
        # Fallback to old location
        from finance_ml.ml_workflow.advanced_models import (
            train_random_forest_regressor,
            train_ridge_regressor,
        )

    # DataFrame structure diagnostics
    logger.info("=" * 60)
    logger.info("TRAIN SECTOR-SPECIFIC MODELS - DataFrame Diagnostics")
    logger.info("=" * 60)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Total columns: {len(df.columns)}")

    # Analyze column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"  Numeric columns: {len(numeric_cols)}")
    logger.info(f"  Object columns: {len(object_cols)}")

    # Check for target and sector columns
    if target_col in df.columns:
        logger.info(f"  ✓ Target column '{target_col}' present")
    else:
        logger.warning(f"  ⚠ Target column '{target_col}' NOT FOUND")

    if sector_col in df.columns:
        n_sectors = df[sector_col].nunique()
        logger.info(f"  ✓ Sector column '{sector_col}' present ({n_sectors} unique sectors)")
    else:
        logger.warning(f"  ⚠ Sector column '{sector_col}' NOT FOUND")

    logger.info("=" * 60)

    # Smart handling of feature_cols
    actual_feature_cols: List[str]
    if isinstance(feature_cols, dict):
        logger.info("feature_cols is a dict, extracting feature list...")
        all_key = feature_cols.get("all_features")
        if all_key:
            actual_feature_cols = list(all_key)
            logger.info(f"  Using 'all_features' key: {len(actual_feature_cols)} features")
        else:
            combined: List[str] = []
            for key in ["numeric_features", "categorical_features", "classification_features"]:
                vals = feature_cols.get(key, [])
                if vals:
                    combined.extend(list(vals))
            actual_feature_cols = combined
            logger.info(f"  Combined feature types: {len(actual_feature_cols)} features")
        # Deduplicate while preserving order
        before = len(actual_feature_cols)
        actual_feature_cols = list(dict.fromkeys(actual_feature_cols))
        if len(actual_feature_cols) != before:
            logger.info(f"  After deduplication: {len(actual_feature_cols)} features")
    elif isinstance(feature_cols, list):
        actual_feature_cols = feature_cols
        logger.info(f"feature_cols is already a list: {len(actual_feature_cols)} features")
    else:
        # Attempt a graceful conversion (e.g., pandas Index or numpy array)
        try:
            actual_feature_cols = list(feature_cols)  # type: ignore[arg-type]
            logger.info(
                f"feature_cols provided as {type(feature_cols).__name__}; converted to list with "
                f"{len(actual_feature_cols)} features"
            )
        except Exception as e:
            raise TypeError(
                f"feature_cols must be a list or dict of lists; got {type(feature_cols).__name__}"
            ) from e

    # Basic empty check
    if len(actual_feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty")

    # Validate that feature columns exist in the DataFrame; skip missing with warning
    available_features = [c for c in actual_feature_cols if c in df.columns]
    missing_features = [c for c in actual_feature_cols if c not in df.columns]

    if missing_features:
        msg = (
            f"⚠ Warning: {len(missing_features)} features not in DataFrame (will be skipped). "
            f"Missing: {missing_features[:5]}..."
            if len(missing_features) > 5
            else f"⚠ Warning: {len(missing_features)} features not in DataFrame (will be skipped): {missing_features}"
        )
        logger.warning(msg)

    actual_feature_cols = available_features

    if len(actual_feature_cols) == 0:
        # Try auto-extraction fallback if enabled
        if auto_extract_fallback:
            logger.warning("No valid features from input, attempting auto-extraction...")
            actual_feature_cols = extract_numeric_feature_columns(
                df, exclude_cols=[target_col, sector_col]
            )

            if len(actual_feature_cols) > 0:
                logger.info(
                    f"✓ Auto-extracted {len(actual_feature_cols)} numeric features from DataFrame"
                )
                logger.info(f"  First 10 features: {actual_feature_cols[:10]}")
            else:
                error_msg = (
                    "❌ No valid feature columns found even after auto-extraction.\n"
                    f"  DataFrame has {len(df.columns)} columns total:\n"
                    f"    - {len(numeric_cols)} numeric columns\n"
                    f"    - {len(object_cols)} object/categorical columns\n"
                    f"  Tried to exclude: {target_col}, {sector_col}\n"
                    f"  Available columns: {list(df.columns)[:20]}"
                    + ("..." if len(df.columns) > 20 else "")
                )
                raise ValueError(error_msg)
        else:
            # Auto-extraction not enabled, provide detailed error
            sample_cols = list(df.columns)[:20]
            error_msg = (
                "❌ No valid feature columns remain after validation against DataFrame.\n"
                f"  Requested features: {len(actual_feature_cols + missing_features)} "
                f"(0 valid, {len(missing_features)} missing)\n"
                f"  DataFrame columns ({len(df.columns)} total): {sample_cols}"
                + ("..." if len(df.columns) > 20 else "")
                + f"\n  Missing features: {missing_features[:10]}"
                + ("..." if len(missing_features) > 10 else "")
                + "\n\n💡 Tip: Set auto_extract_fallback=True to automatically extract "
                "numeric features from the DataFrame."
            )
            raise ValueError(error_msg)

    logger.info(f"✓ Final feature count for sector regression: {len(actual_feature_cols)}")

    # ============================================================================
    # VALIDATE AND CLEAN TARGET COLUMN
    # ============================================================================
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Drop rows with NaN in target before training
    nan_target_count = df[target_col].isna().sum()
    if nan_target_count > 0:
        logger.warning(
            f"⚠ Target column '{target_col}' contains {nan_target_count} NaN values. "
            f"Dropping these rows before training."
        )
        df = df[df[target_col].notna()].copy()
        logger.info(f"✓ After dropping NaN targets: {len(df)} rows remaining")

    sector_models: Dict[str, Any] = {}
    sector_metrics: Dict[str, Any] = {}

    sectors = df[sector_col].unique()

    for sector in sectors:
        # Filter data for sector
        sector_df = df[df[sector_col] == sector]

        if len(sector_df) < min_samples:
            logger.info(
                f"Skipping sector '{sector}': only {len(sector_df)} samples (min: {min_samples})"
            )
            continue

        # Apply preprocessing with imputation to handle NaN values
        try:
            X_sector, y_sector = prepare_features_for_training(
                df=sector_df,
                feature_cols=actual_feature_cols,
                target_col=target_col,
                apply_imputation=True,
                sector_column=sector_col,
            )
        except Exception as e:
            logger.warning(
                f"⚠ Failed to prepare features for sector '{sector}': {e}. Skipping this sector."
            )
            continue

        # Train model
        try:
            if model_type == "random_forest":
                # Code Guidelines Section 1.1: Training functions return (model, results_dict) tuple
                model, result = train_random_forest_regressor(
                    X_sector, y_sector, n_estimators=50, random_state=random_state
                )
                metrics = {
                    "train_score": result.get("train_score", 0),
                    "cv_mean": result.get("cv_mean", 0),
                    "cv_std": result.get("cv_std", 0),
                }
            else:  # ridge
                # Code Guidelines Section 1.1: Training functions return (model, results_dict) tuple
                model, result = train_ridge_regressor(
                    X_sector,
                    y_sector,
                    cv=5,
                    random_state=random_state,
                    ensure_nonnegative=ensure_nonnegative,
                )
                metrics = {
                    "train_score": result.get("train_score", 0),
                    "cv_mean": result.get("cv_mean", 0),
                    "cv_std": result.get("cv_std", 0),
                    "best_alpha": result.get("best_alpha", 1.0),
                }

            sector_models[sector] = model
            sector_metrics[sector] = metrics
            logger.info(
                f"✓ Trained {model_type} for sector '{sector}': "
                f"{len(sector_df)} samples, CV R²={metrics['cv_mean']:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to train model for sector '{sector}': {e}")
            continue

    # Summary
    logger.info("=" * 60)
    logger.info(f"✓ Sector-specific training complete: {len(sector_models)}/{len(sectors)} sectors")
    logger.info("=" * 60)

    results = {
        "n_sectors_trained": len(sector_models),
        "n_sectors_total": len(sectors),
        "metrics": sector_metrics,
        "feature_count": len(actual_feature_cols),
    }

    return sector_models, results
