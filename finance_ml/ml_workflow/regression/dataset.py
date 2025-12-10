"""
Dataset preparation and feature engineering for regression models.

This module provides functions for preparing data for regression training,
including classification feature integration, validation, and sector-specific
preprocessing.

Phase 9.5 - Regression Refactor
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Phase 9.9 Gap 3: shared split policy (time-series → grouped → stratified)
from finance_ml.ml_workflow.validation.splits import create_train_test_split
from pathlib import Path
import os

# Phase 16.4: Feature validation/pruning utilities (Phase 9.3 review)
try:
    from finance_ml.ml_workflow.features.validation import (
        validate_feature_coverage,
        prune_low_importance_features,
    )
except Exception:  # pragma: no cover - optional at runtime
    validate_feature_coverage = None
    prune_low_importance_features = None

logger = logging.getLogger(__name__)


# ==============================================================================
# Classification Feature Integration
# ==============================================================================


def extract_classification_features(probabilities: np.ndarray) -> pd.DataFrame:
    """Extract classification features from predicted probabilities.

    Converts raw classifier probabilities into structured features that
    can be used as inputs for regression models. The classification
    features provide meta-information about market sentiment and event
    likelihood.

    This implementation is aligned with the **5-class event labeling
    system** used throughout Phase 9.4 / 9.9, where labels are:

    - 0 → Strong Negative
    - 1 → Negative
    - 2 → Neutral
    - 3 → Positive
    - 4 → Strong Positive

    It expects an array of shape ``(n_samples, 5)`` with columns ordered
    exactly as above and returns a DataFrame with 7 columns:

    - ``event_prob_strong_negative``
    - ``event_prob_negative``
    - ``event_prob_neutral``
    - ``event_prob_positive``
    - ``event_prob_strong_positive``
    - ``event_class_predicted`` (argmax over the 5 classes)
    - ``event_confidence`` (max probability across classes)

    Args:
        probabilities: Array of shape (n_samples, 5) with class
            probabilities from a trained 5-class event classifier.

    Returns:
        DataFrame with classification features (n_samples rows, 7 columns).

    Raises:
        ValueError: If probabilities array doesn't have exactly 5 classes.
    """

    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError(f"Expected 5 classes, got shape {probabilities.shape}")

    n_samples = probabilities.shape[0]
    logger.debug(f"Extracting 5-class classification features for {n_samples} samples")

    features = pd.DataFrame(
        {
            "event_prob_strong_negative": probabilities[:, 0],
            "event_prob_negative": probabilities[:, 1],
            "event_prob_neutral": probabilities[:, 2],
            "event_prob_positive": probabilities[:, 3],
            "event_prob_strong_positive": probabilities[:, 4],
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

    # Drop overlapping columns from base dataframe to prevent duplicates
    # (e.g. if integrate_classification_features is called multiple times)
    overlap_cols = [col for col in features_reset.columns if col in df_reset.columns]
    if overlap_cols:
        logger.debug(
            f"Dropping {len(overlap_cols)} overlapping classification columns "
            f"from base dataframe to avoid duplicates: {overlap_cols}"
        )
        df_reset = df_reset.drop(columns=overlap_cols)

    # Concatenate horizontally
    result = pd.concat([df_reset, features_reset], axis=1)

    logger.debug(f"Integration complete: {len(result)} rows, {len(result.columns)} total columns")

    return result


def integrate_classification_features(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Convenience wrapper to integrate classifier probabilities into ``df``.

    This helper mirrors the Phase 9.9 plan API by taking the raw
    ``y_proba`` array from an event classifier, converting it into
    standardized classification meta-features via
    :func:`extract_classification_features`, and then combining those
    features with the main regression dataframe using
    :func:`integrate_classification_features_into_dataframe`.

    The resulting dataframe has the same number of rows as ``df`` and
    includes the seven event meta-feature columns for the 5-class system:

    - ``event_prob_strong_negative``
    - ``event_prob_negative``
    - ``event_prob_neutral``
    - ``event_prob_positive``
    - ``event_prob_strong_positive``
    - ``event_class_predicted``
    - ``event_confidence``

    Parameters
    ----------
    df : pandas.DataFrame
        Base regression dataframe (ticker, sector, price_target, etc.).
    probabilities : numpy.ndarray
        Class probabilities of shape ``(n_samples, 5)`` for the 5-class
        event labeling system (Strong Negative, Negative, Neutral,
        Positive, Strong Positive).

    Returns
    -------
    pandas.DataFrame
        Enhanced dataframe with classification meta-features appended.

    Raises
    ------
    ValueError
        If ``probabilities`` does not have exactly 5 columns or its
        row count does not match ``len(df)``.
    """

    class_features = extract_classification_features(probabilities)
    return integrate_classification_features_into_dataframe(df, class_features)


def create_classification_interactions(
    df: pd.DataFrame, classification_cols: List[str], valuation_cols: List[str]
) -> pd.DataFrame:
    """
    Create interaction features between classification probabilities and valuation metrics.

    Delegates to the unified `build_prob_valuation_interactions` utility
    (Phase 9.5 P2).

    Args:
        df: Input DataFrame
        classification_cols: Classification feature columns
        valuation_cols: Valuation metric columns

    Returns:
        DataFrame with additional interaction features
    """
    from finance_ml.ml_workflow.regression.features import build_prob_valuation_interactions

    # Note: build_prob_valuation_interactions naming convention is {val}_x_{prob}
    # The original implementation here was {class}_x_{val}.
    # Code guidelines prefer valuation first.
    # We will use the new utility which enforces standard.
    return build_prob_valuation_interactions(df, valuation_cols, classification_cols)


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
        # Exclude target, price columns, and market_cap to prevent feature leakage
        # market_cap causes predictions on wrong scale (market_cap scale vs price scale)
        exclude_cols = [target_col, "last_price", "market_cap"]

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

    # ------------------------------------------------------------------
    # Priority 1 Fix: Remove market_cap leakage columns (Critical)
    # market_cap is derived from price (market_cap = price × shares_outstanding),
    # so including it or its derivatives causes feature leakage and predictions
    # on the wrong scale (~880K market_cap scale instead of ~736K price scale).
    # Remove all market_cap-related columns that may have been created during
    # earlier feature engineering (e.g., log_market_cap, market_cap_x_*, etc.)
    # ------------------------------------------------------------------
    leakage_cols = [col for col in X.columns if "market_cap" in col.lower()]

    if leakage_cols:
        logger.info(
            f"🔧 Removing {len(leakage_cols)} market_cap columns to prevent feature leakage"
        )
        logger.debug(
            f"   Leakage columns: {leakage_cols[:5]}{'...' if len(leakage_cols) > 5 else ''}"
        )
        X = X.drop(columns=leakage_cols, errors="ignore")
        logger.info(f"✓ Feature matrix shape after leakage removal: {X.shape}")
    else:
        logger.debug("✓ No market_cap leakage columns detected")

    # ------------------------------------------------------------------
    # Split data using the shared Phase 9.9 split policy helper where
    # possible.  This enforces the documented priority:
    #   1) time-aware (snapshot_date)
    #   2) grouped by ticker
    #   3) stratified by sector
    #   4) random as final fallback
    # while keeping the public signature and return types unchanged.
    # ------------------------------------------------------------------

    # Use create_train_test_split only when we have the original
    # dataframe available with potential policy columns; otherwise
    # fall back to sklearn.train_test_split on the numeric matrix.
    if len(df) == len(X):
        # Attach a stable row id so we can map back from the split
        # dataframes to the numeric feature matrix / target series
        df_for_split = df.copy()
        df_for_split["__row_id__"] = np.arange(len(df_for_split))

        date_col = "snapshot_date" if "snapshot_date" in df_for_split.columns else None
        group_col = "ticker" if "ticker" in df_for_split.columns else None
        stratify_col = "sector" if "sector" in df_for_split.columns else None

        train_df, test_df = create_train_test_split(
            df_for_split,
            date_col=date_col,
            group_col=group_col,
            stratify_col=stratify_col,
            test_size=test_size,
            random_state=random_state,
        )

        # Map back to numeric feature rows via the synthetic row id
        train_idx = train_df["__row_id__"].to_numpy()
        test_idx = test_df["__row_id__"].to_numpy()

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
    else:
        # Conservative fallback – should not normally happen, but keeps
        # behaviour identical to earlier versions if dataset shapes
        # diverge for any reason.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # === Phase 9.3: Feature engineering review and optimizations ===
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classification_features": classification_features,
        "all_features": list(X.columns),
        "pruned_features": [],
        "added_sector_interactions": 0,
    }

    # 1) Validate feature coverage (expect ~318 engineered features in rich setups)
    try:
        if validate_feature_coverage is not None:
            ok, report = validate_feature_coverage(X, expected=318, strict=False)
            feature_info["feature_coverage_ok"] = bool(ok)
            feature_info["feature_coverage_report"] = report
    except Exception:
        # Non-fatal
        pass

    # 2) Optional low-importance pruning using persisted feature_importance.csv
    #    Threshold can be overridden via env FEATURE_IMPORTANCE_THRESHOLD (default 0.01)
    try:
        threshold = float(os.getenv("FEATURE_IMPORTANCE_THRESHOLD", "0.01"))
    except Exception:
        threshold = 0.01

    try:
        fi_path = Path("outputs") / "regression" / "feature_importance.csv"
        if prune_low_importance_features is not None and fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            # Preserve classification probability features and confidence
            keep_cols = [
                c for c in X_train.columns if c.startswith("event_prob_") or c == "event_confidence"
            ]
            X_train, X_test, kept_imp = prune_low_importance_features(
                X_train, X_test, fi_df, threshold=threshold, keep_cols=keep_cols
            )
            feature_info["all_features"] = list(X_train.columns)
            pruned = [c for c in fi_df["feature"].tolist() if c not in X_train.columns]
            feature_info["pruned_features"] = pruned
    except Exception:
        # If importance file malformed or any issue, skip pruning silently to avoid breaking pipeline
        pass

    # 3) Add sector-specific interaction terms (Phase 9.3)
    #    Controlled by env FEATURE_SECTOR_INTERACTIONS (default: 1/True)
    def _add_sector_interactions(X_in: pd.DataFrame, idx_like) -> pd.DataFrame:
        # Craft a curated list of base columns; only include if present in X
        # NOTE: market_cap excluded to prevent feature leakage (causes predictions on market_cap scale)
        base_cols = [
            "p_e_ratio",
            "ev_ebitda_ratio",
            "gross_margin",
            "debt_to_equity",  # Fundamental risk metric (replaces market_cap)
            "beta_5y",
        ]
        existing = [c for c in base_cols if c in X_in.columns]
        if not existing:
            return X_in
        if "sector" not in df.columns:
            return X_in
        sectors = (
            df.loc[idx_like, "sector"]
            if isinstance(idx_like, (pd.Index, list, np.ndarray))
            else df.loc[X_in.index, "sector"]
        )
        dummies = pd.get_dummies(sectors.astype(str), prefix="sector", dummy_na=False)
        # Align indices
        dummies.index = X_in.index
        # Create interactions
        new_cols = {}
        for dcol in dummies.columns:
            d = dummies[dcol]
            for bcol in existing:
                inter_name = f"{dcol}__x__{bcol}"
                # Multiply safely; cast to float
                new_cols[inter_name] = d.values.astype(float) * X_in[bcol].values.astype(float)
        if new_cols:
            interactions_df = pd.DataFrame(new_cols, index=X_in.index)
            X_out = pd.concat([X_in, interactions_df], axis=1)
            return X_out
        return X_in

    enable_sector_ix = os.getenv("FEATURE_SECTOR_INTERACTIONS", "1").strip() not in {
        "0",
        "false",
        "False",
    }
    if enable_sector_ix:
        try:
            before_cols = set(X_train.columns)
            X_train = _add_sector_interactions(X_train, X_train.index)
            X_test = _add_sector_interactions(X_test, X_test.index)
            added = len(set(X_train.columns) - before_cols)
            feature_info["added_sector_interactions"] = added
            feature_info["all_features"] = list(X_train.columns)
        except Exception:
            # Non-fatal
            pass

    return X_train, X_test, y_train, y_test, feature_info


def add_sector_interactions_for_prediction(
    X: pd.DataFrame, df_with_sector: pd.DataFrame, base_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add sector interaction features to prediction data.

    Replicates the sector interaction logic from prepare_regression_data()
    for use on new/unseen data during prediction. This ensures feature parity
    between training and prediction pipelines.

    **Purpose**: The `prepare_regression_data()` function automatically generates
    sector-specific interaction features (e.g., `sector_Technology__x__p_e_ratio`)
    during training. When making predictions on new data (e.g., `all_stocks_phase95`),
    these interactions must be regenerated to match the model's expected feature set.

    **Feature Naming Convention**: `sector_{SectorName}__x__{base_feature}`
    - Example: `sector_Information Technology__x__ev_ebitda_ratio`
    - Total features: n_sectors × n_base_cols (e.g., 11 sectors × 5 features = 55)

    Args:
        X: Feature matrix (numeric features only, same index as df_with_sector)
        df_with_sector: Original DataFrame with 'sector' column aligned to X
        base_cols: Base columns for interactions. Defaults to:
            ['p_e_ratio', 'ev_ebitda_ratio', 'gross_margin', 'debt_to_equity', 'beta_5y']
            (market_cap excluded to prevent feature leakage)

    Returns:
        X with sector interaction features appended (column order: original + interactions)

    Raises:
        ValueError: If X and df_with_sector have mismatched indices

    Example:
        >>> # After training with prepare_regression_data() which added sector interactions
        >>> X_train, X_test, y_train, y_test, meta = prepare_regression_data(df_train)
        >>> print(X_train.shape)  # (5631, 42) including 22 sector interactions
        >>>
        >>> # For prediction on new data
        >>> X_pred = all_stocks_phase95[base_feature_cols].copy()
        >>> X_pred = add_sector_interactions_for_prediction(
        ...     X_pred,
        ...     df_with_sector=all_stocks_phase95
        ... )
        >>> print(X_pred.shape)  # (7055, 42) - now matches X_train features
        >>>
        >>> # Safe to predict
        >>> predictions = model.predict(X_pred)

    Note:
        - Only adds interactions for base_cols that exist in X
        - Returns original X unchanged if 'sector' column missing from df_with_sector
        - Controlled by FEATURE_SECTOR_INTERACTIONS environment variable (default: enabled)
        - Indices must match between X and df_with_sector for proper alignment

    See Also:
        - prepare_regression_data(): Training-time function that generates these features
        - docs/code_guidelines.md Section 16.5: Sector interaction feature policy
    """
    if base_cols is None:
        # NOTE: market_cap excluded to prevent feature leakage (causes predictions on market_cap scale)
        base_cols = ["p_e_ratio", "ev_ebitda_ratio", "gross_margin", "debt_to_equity", "beta_5y"]

    # Validate inputs
    if len(X) != len(df_with_sector):
        raise ValueError(
            f"Index mismatch: X has {len(X)} rows, df_with_sector has {len(df_with_sector)} rows"
        )

    # Check if sector column exists
    if "sector" not in df_with_sector.columns:
        logger.warning("No 'sector' column in df_with_sector; skipping sector interactions")
        return X

    # Filter base_cols to only those present in X
    existing = [c for c in base_cols if c in X.columns]
    if not existing:
        logger.warning(
            f"None of the base_cols {base_cols} found in X; skipping sector interactions"
        )
        return X

    logger.info(
        f"Generating sector interactions for {len(existing)} base features "
        f"across {df_with_sector['sector'].nunique()} sectors"
    )

    # Get sector values aligned to X's index
    sectors = df_with_sector.loc[X.index, "sector"]

    # Create one-hot encoded sector dummies
    dummies = pd.get_dummies(sectors.astype(str), prefix="sector", dummy_na=False)
    dummies.index = X.index

    # Generate interaction features: sector_dummy × base_feature
    new_cols = {}
    for dcol in dummies.columns:
        d = dummies[dcol]
        for bcol in existing:
            inter_name = f"{dcol}__x__{bcol}"
            # Element-wise multiplication: binary sector indicator × continuous feature
            new_cols[inter_name] = d.values.astype(float) * X[bcol].values.astype(float)

    if new_cols:
        interactions_df = pd.DataFrame(new_cols, index=X.index)
        X_out = pd.concat([X, interactions_df], axis=1)
        logger.info(
            f"✓ Added {len(new_cols)} sector interaction features "
            f"({X.shape[1]} → {X_out.shape[1]} columns)"
        )
        return X_out

    logger.warning("No sector interactions generated (empty dummies or existing features)")
    return X


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
    apply_imputation: bool = False,
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


# ============================================================================
# Phase 9.5 Task 7: Unified Test Data Alignment
# ============================================================================


def align_features_to_model(
    X_test: pd.DataFrame,
    model: Any,
    fill_value: float = 0.0,
    warn_missing: bool = True,
    warn_extra: bool = True,
) -> pd.DataFrame:
    """
    Align test features to match trained model's expected features.

    This function ensures X_test has exactly the features the model expects,
    in the correct order, preventing prediction-time errors. Implements
    code_guidelines.md Section 7.5 feature alignment policy.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test features to align
    model : sklearn estimator or compatible
        Trained model with feature_names_in_ attribute
    fill_value : float, default=0.0
        Value to fill missing features
    warn_missing : bool, default=True
        Log warning for missing features
    warn_extra : bool, default=True
        Log warning for extra features

    Returns
    -------
    pd.DataFrame
        Aligned test features matching model's feature order

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X_train = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> y_train = [5, 6]
    >>> model = LinearRegression().fit(X_train, y_train)
    >>> X_test = pd.DataFrame({'a': [1.5], 'c': [2.5]})  # Missing 'b', extra 'c'
    >>> X_aligned = align_features_to_model(X_test, model)
    >>> list(X_aligned.columns)
    ['a', 'b']
    >>> X_aligned['b'].iloc[0]
    0.0

    Notes
    -----
    - Missing features are filled with fill_value (default 0.0)
    - Extra features not expected by model are removed
    - Column order matches model.feature_names_in_
    - Supports sklearn, XGBoost, LightGBM, CatBoost models
    """
    # Get expected features from model
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):  # LightGBM
        expected_features = list(model.feature_name_)
    elif hasattr(model, "get_booster"):  # XGBoost
        expected_features = model.get_booster().feature_names
    else:
        logger.warning("Model does not expose feature names; returning X_test unchanged")
        return X_test

    # Identify missing and extra features
    test_features = set(X_test.columns)
    expected_features_set = set(expected_features)

    missing_features = expected_features_set - test_features
    extra_features = test_features - expected_features_set

    # Log warnings
    if missing_features and warn_missing:
        logger.warning(
            f"X_test missing {len(missing_features)} features expected by model. "
            f"Filling with {fill_value}. Missing: {sorted(missing_features)[:5]}..."
        )

    if extra_features and warn_extra:
        logger.warning(
            f"X_test has {len(extra_features)} extra features not in model. "
            f"Removing. Extra: {sorted(extra_features)[:5]}..."
        )

    # Create aligned dataframe
    X_aligned = X_test.copy()

    # Add missing features with fill_value
    for feature in missing_features:
        X_aligned[feature] = fill_value

    # Select only expected features in correct order
    X_aligned = X_aligned[expected_features]

    return X_aligned


def predict_with_model(
    model: Any, X_test: pd.DataFrame, auto_align: bool = True, **kwargs
) -> np.ndarray:
    """
    Predict with automatic feature alignment.

    Wrapper around model.predict() that automatically aligns test features
    to the model's expected feature set, eliminating prediction-time errors
    from feature mismatches.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    auto_align : bool, default=True
        Automatically align features to model
    **kwargs
        Additional arguments passed to model.predict()

    Returns
    -------
    np.ndarray
        Predictions

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> X_train = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
    >>> y_train = [10, 20, 30]
    >>> model = Ridge().fit(X_train, y_train)
    >>> X_test = pd.DataFrame({'x1': [1.5], 'x3': [7.0]})  # Mismatched features
    >>> preds = predict_with_model(model, X_test)  # Works without error
    >>> len(preds)
    1

    Notes
    -----
    - Set auto_align=False to disable automatic alignment
    - Compatible with sklearn, XGBoost, LightGBM, CatBoost models
    - Reduces notebook boilerplate by ~50 lines per prediction cell
    """
    if auto_align:
        X_test = align_features_to_model(X_test, model)

    return model.predict(X_test, **kwargs)
