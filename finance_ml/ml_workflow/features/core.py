"""
finance_ml.ml_workflow.features.core - Core feature engineering functions

This module provides basic feature engineering functions for financial data.
Part of Phase 9.3 refactor: Extracted from features.py for better modularity.

Functions:
- _safe_div: Safe division helper
- engineer_basic_ratios: P/E, P/B, debt ratios, ROE, ROA
- engineer_margin_features: Profit margins (gross, operating, net, EBITDA)
- engineer_volatility_features: Price volatility metrics
- engineer_revenue_cagr: Revenue growth rates
- build_features_and_target: Feature matrix preparation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def preprocess_for_lightgbm(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    return_encoders: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, LabelEncoder]]]:
    """
    Preprocess DataFrame for LightGBM compatibility with proper train/test split support.

    LightGBM requires all input features to be numeric (int, float, or bool).
    This function handles:
    1. Categorical columns (object dtype) - converted to numeric using LabelEncoder
    2. Datetime columns - extracted to numeric features (year, month, day, days_from_now)
    3. Ensures all remaining columns are numeric

    **IMPORTANT for Train/Test Usage:**
    - Training: Call with `return_encoders=True` to fit and save encoders
    - Testing: Call with `encoders=<training_encoders>` to apply same transformations
    - Always pass the same `reference_date` for consistent datetime features

    Args:
        df: Input DataFrame
        categorical_columns: List of categorical columns to encode. If None, auto-detects object dtype columns
        datetime_columns: List of datetime columns to convert. If None, auto-detects datetime64 dtype columns
        drop_columns: List of columns to drop before processing
        encoders: Pre-fitted LabelEncoders from training data (for test/inference).
                 If provided, uses transform() instead of fit_transform().
                 If None, fits new encoders (training mode).
        reference_date: Fixed reference date for datetime feature extraction.
                       If None, uses current timestamp (NOT recommended for train/test splits).
                       Store this from training and reuse for test data!
        return_encoders: If True, returns tuple (df, encoders_dict). If False, returns (df, None)

    Returns:
        Tuple of (preprocessed_df, encoders_dict or None)
        - preprocessed_df: DataFrame with only numeric types
        - encoders_dict: Dictionary with:
            - Column name -> LabelEncoder mappings (if return_encoders=True)
            - '_reference_date' -> reference date used for datetime features

    Examples:
        >>> # TRAINING MODE: Fit encoders and save reference date
        >>> X_train_processed, encoders = preprocess_for_lightgbm(
        ...     X_train,
        ...     categorical_columns=['sector', 'industry', 'region'],
        ...     datetime_columns=['next_earnings'],
        ...     return_encoders=True  # Save encoders for later
        ... )
        >>> # Extract reference date for test consistency
        >>> ref_date = encoders.get('_reference_date')
        >>>
        >>> # TEST/INFERENCE MODE: Use training encoders and reference date
        >>> X_test_processed, _ = preprocess_for_lightgbm(
        ...     X_test,
        ...     categorical_columns=['sector', 'industry', 'region'],
        ...     datetime_columns=['next_earnings'],
        ...     encoders=encoders,  # Use training encoders
        ...     reference_date=ref_date  # Use training reference date
        ... )
        >>>
        >>> # Interpret results using encoders
        >>> original_sector = encoders['sector'].inverse_transform([0, 1, 2])

    Notes:
        - Handles missing values by filling NaN with placeholder strings for categoricals
        - Unseen categories in test data are mapped to 'Unknown' class
        - Datetime NaN values are filled with 0 after feature extraction
        - All inf values are replaced with NaN and filled with 0
        - Stores reference_date in encoders dict with key '_reference_date'
    """
    result = df.copy()

    # Determine if we're in training mode (fitting) or inference mode (transforming)
    is_training = encoders is None

    # Initialize encoders dict for training or use provided encoders for inference
    if is_training:
        encoders_out = {} if return_encoders else None
        provided_encoders = {}
    else:
        encoders_out = encoders if return_encoders else None
        provided_encoders = encoders.copy()
        # Remove metadata keys from encoder dict
        provided_encoders.pop("_reference_date", None)

    # Set reference date for datetime features
    if reference_date is None:
        reference_date = pd.Timestamp(datetime.now())
        if not is_training:
            logger.warning(
                "No reference_date provided for test/inference data. "
                "Using current timestamp may cause train/test inconsistency!"
            )

    # Drop specified columns
    if drop_columns:
        cols_to_drop = [col for col in drop_columns if col in result.columns]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop[:5]}")

    # Auto-detect categorical columns if not specified
    if categorical_columns is None:
        categorical_columns = result.select_dtypes(include=["object", "category"]).columns.tolist()
        # Exclude columns that look like datetime strings
        date_keywords = ["date", "updated", "earnings", "time", "income"]
        categorical_columns = [
            col
            for col in categorical_columns
            if not any(keyword in col.lower() for keyword in date_keywords)
        ]

    # Auto-detect datetime columns if not specified
    if datetime_columns is None:
        datetime_columns = result.select_dtypes(include=["datetime64"]).columns.tolist()

        # Also check object columns that might be datetime strings
        # Look for columns with datetime-like keywords that weren't classified as categorical
        date_keywords = ["date", "updated", "earnings", "time", "income"]
        for col in result.select_dtypes(include=["object"]).columns:
            if col not in categorical_columns and any(
                keyword in col.lower() for keyword in date_keywords
            ):
                # Try to convert to datetime to confirm it's a date column
                try:
                    test_conversion = pd.to_datetime(result[col].dropna().head(10), errors="coerce")
                    if test_conversion.notna().any():
                        datetime_columns.append(col)
                        logger.debug(f"Auto-detected datetime string column: '{col}'")
                except (ValueError, TypeError):
                    pass

    # Handle categorical columns with LabelEncoder
    if categorical_columns:
        categorical_columns = [col for col in categorical_columns if col in result.columns]
        mode_str = "Fitting and encoding" if is_training else "Encoding"
        logger.info(f"{mode_str} {len(categorical_columns)} categorical columns")

        for col in categorical_columns:
            # Fill NaN with placeholder
            result[col] = result[col].fillna("Unknown").astype(str)

            if is_training:
                # TRAINING MODE: Fit new encoder
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col])

                # Store encoder if requested
                if return_encoders and encoders_out is not None:
                    encoders_out[col] = le

                logger.debug(f"Fitted encoder for '{col}' with {len(le.classes_)} unique values")
            else:
                # INFERENCE MODE: Use provided encoder
                if col not in provided_encoders:
                    raise ValueError(
                        f"Column '{col}' requires encoding but no encoder was provided. "
                        f"Available encoders: {list(provided_encoders.keys())}"
                    )

                le = provided_encoders[col]

                # Handle unseen categories by mapping them to 'Unknown'
                # First, ensure 'Unknown' is in the encoder's classes
                if "Unknown" not in le.classes_:
                    # Add 'Unknown' to the encoder classes
                    le.classes_ = np.append(le.classes_, "Unknown")

                # Map unseen categories to 'Unknown'
                unknown_idx = np.where(le.classes_ == "Unknown")[0][0]
                values = result[col].values

                # Transform known categories
                mask_known = np.isin(values, le.classes_)
                encoded = np.full(len(values), unknown_idx, dtype=int)
                if mask_known.any():
                    encoded[mask_known] = le.transform(values[mask_known])

                result[col] = encoded

                # Log if unseen categories were found
                unseen = set(values[~mask_known]) - {"Unknown"}
                if unseen:
                    logger.warning(
                        f"Column '{col}' has {len(unseen)} unseen categories in test data. "
                        f"Mapped to 'Unknown' class. Examples: {list(unseen)[:5]}"
                    )

                logger.debug(f"Transformed '{col}' using training encoder")

    # Handle datetime columns
    if datetime_columns:
        datetime_columns = [col for col in datetime_columns if col in result.columns]
        logger.info(f"Extracting features from {len(datetime_columns)} datetime columns")
        logger.debug(f"Using reference_date: {reference_date}")

        for col in datetime_columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors="coerce")

            # Extract numeric features
            result[f"{col}_year"] = result[col].dt.year
            result[f"{col}_month"] = result[col].dt.month
            result[f"{col}_day"] = result[col].dt.day

            # Days from CONSISTENT reference date (critical for train/test consistency)
            result[f"{col}_days_from_now"] = (result[col] - reference_date).dt.days

            # Fill NaN in extracted features
            for suffix in ["_year", "_month", "_day", "_days_from_now"]:
                new_col = f"{col}{suffix}"
                if new_col in result.columns:
                    result[new_col] = result[new_col].fillna(0)

            # Drop original datetime column
            result = result.drop(columns=[col])
            logger.debug(f"Extracted datetime features from '{col}'")

        # Store reference date in encoder dict for reproducibility (only when datetime columns exist)
        if return_encoders and encoders_out is not None:
            encoders_out["_reference_date"] = reference_date

    # Ensure all remaining columns are numeric
    logger.info("Converting all remaining columns to numeric")
    for col in result.columns:
        if result[col].dtype == "object":
            # Try to convert to numeric
            result[col] = pd.to_numeric(result[col], errors="coerce")
            logger.debug(f"Converted object column '{col}' to numeric")

    # Handle infinite values
    result = result.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with 0
    result = result.fillna(0)

    # Final validation
    non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Non-numeric columns remain: {non_numeric}")
        # Last resort: drop them
        result = result.drop(columns=non_numeric)

    logger.info(f"Preprocessing complete. Final shape: {result.shape}")
    logger.info(f"Data types: {result.dtypes.value_counts().to_dict()}")

    return result, encoders_out


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf with NaN.

    Args:
        numer: Numerator Series
        denom: Denominator Series

    Returns:
        Result Series with inf values replaced by NaN
    """
    result = numer.astype(float) / denom.astype(float)
    # Replace +/- inf with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def engineer_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add a minimal set of engineered ratio features if source columns exist.

    Ratios computed:
    - ev_to_ebitda = enterprise_value / ebitda (or ev / ebitda)
    - net_debt_to_ebitda = net_debt / ebitda
    - p_e = last_price / eps
    - p_b = last_price / book_value_per_share (or market_cap / total_equity)
    - debt_to_equity = total_debt / total_equity
    - roe = net_income / total_equity
    - roa = net_income / total_assets
    - market_cap_to_revenue = market_cap / revenue

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ratio features added (preserves original columns)
    """
    out = df.copy()
    cols = set(out.columns)

    # Support both naming conventions
    if {"enterprise_value", "ebitda"}.issubset(cols):
        out["ev_to_ebitda"] = _safe_div(out["enterprise_value"], out["ebitda"])
    elif {"ev", "ebitda"}.issubset(cols):  # Alternative naming
        out["ev_to_ebitda"] = _safe_div(out["ev"], out["ebitda"])

    if {"net_debt", "ebitda"}.issubset(cols):
        out["net_debt_to_ebitda"] = _safe_div(out["net_debt"], out["ebitda"])
    elif {"total_debt", "ebitda"}.issubset(cols):  # Alternative: use total_debt as proxy
        out["net_debt_to_ebitda"] = _safe_div(out["total_debt"], out["ebitda"])

    if {"last_price", "eps"}.issubset(cols):
        out["p_e"] = _safe_div(out["last_price"], out["eps"])

    if {"last_price", "book_value_per_share"}.issubset(cols):
        out["p_b"] = _safe_div(out["last_price"], out["book_value_per_share"])
    elif {"market_cap", "total_equity"}.issubset(cols):
        # Alternative P/B calculation
        out["p_b"] = _safe_div(out["market_cap"], out["total_equity"])

    # Debt to equity
    if {"total_debt", "total_equity"}.issubset(cols):
        out["debt_to_equity"] = _safe_div(out["total_debt"], out["total_equity"])

    # Return on Equity (ROE)
    if {"net_income", "total_equity"}.issubset(cols):
        out["roe"] = _safe_div(out["net_income"], out["total_equity"])

    # Return on Assets (ROA)
    if {"net_income", "total_assets"}.issubset(cols):
        out["roa"] = _safe_div(out["net_income"], out["total_assets"])

    # Market cap to revenue ratio
    if {"market_cap", "total_revenue"}.issubset(cols):
        out["market_cap_to_revenue"] = _safe_div(out["market_cap"], out["total_revenue"])
    elif {"market_cap", "revenue"}.issubset(cols):
        out["market_cap_to_revenue"] = _safe_div(out["market_cap"], out["revenue"])

    return out


def engineer_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add margin features if source columns exist.

    Supports both production columns (_ltm suffix) and simple test columns.

    Margins computed:
    - gross_margin = gross_profit / revenue
    - operating_margin = operating_income / revenue (or _ltm versions)
    - net_margin = net_income / revenue (or _ltm versions)
    - ebitda_margin = ebitda / revenue (or _ltm versions)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with margin features added (preserves original columns)
    """
    out = df.copy()
    cols = set(out.columns)

    # Support simple test column names
    if {"gross_profit", "revenue"}.issubset(cols):
        out["gross_margin"] = _safe_div(out["gross_profit"], out["revenue"])
    if {"operating_income", "revenue"}.issubset(cols):
        out["operating_margin"] = _safe_div(out["operating_income"], out["revenue"])
    if {"net_income", "revenue"}.issubset(cols):
        out["net_margin"] = _safe_div(out["net_income"], out["revenue"])
    if {"ebitda", "revenue"}.issubset(cols):
        out["ebitda_margin"] = _safe_div(out["ebitda"], out["revenue"])

    # Support alternative column names (total_revenue instead of revenue)
    if {"net_income", "total_revenue"}.issubset(cols) and "net_margin" not in out.columns:
        out["net_margin"] = _safe_div(out["net_income"], out["total_revenue"])
    if {"ebitda", "total_revenue"}.issubset(cols) and "ebitda_margin" not in out.columns:
        out["ebitda_margin"] = _safe_div(out["ebitda"], out["total_revenue"])

    # Support production column names with _ltm suffix
    if {"ebitda_ltm", "total_revenues_ltm"}.issubset(cols):
        if "ebitda_margin" not in out.columns:
            out["ebitda_margin"] = _safe_div(out["ebitda_ltm"], out["total_revenues_ltm"])
    if {"operating_income_ltm", "total_revenues_ltm"}.issubset(cols):
        if "operating_margin" not in out.columns:  # Don't overwrite if already created
            out["operating_margin"] = _safe_div(
                out["operating_income_ltm"], out["total_revenues_ltm"]
            )
    if {"net_income_ltm", "total_revenues_ltm"}.issubset(cols):
        if "net_margin" not in out.columns:  # Don't overwrite if already created
            out["net_margin"] = _safe_div(out["net_income_ltm"], out["total_revenues_ltm"])

    return out


def engineer_volatility_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate or aggregate volatility features.

    If 'last_price' column exists, calculates rolling standard deviation.
    If 'high_price' and 'low_price' exist, calculates price range and relative volatility.
    Also aggregates existing volatility columns into summary features.

    Args:
        df: Input DataFrame
        window: Rolling window size for volatility calculation (default: 30)

    Returns:
        DataFrame with volatility features added
    """
    out = df.copy()

    # If last_price exists, calculate rolling volatility
    if "last_price" in out.columns:
        col_name = f"price_volatility_{window}d"
        out[col_name] = out["last_price"].rolling(window=window, min_periods=window).std()

    # Calculate price range if high/low prices available
    if {"high_price", "low_price"}.issubset(out.columns):
        out["price_range"] = out["high_price"] - out["low_price"]

        # Relative volatility: range / last_price
        if "last_price" in out.columns:
            out["relative_volatility"] = _safe_div(out["price_range"], out["last_price"])

    # Find available volatility columns and create average
    vol_cols = [
        c
        for c in out.columns
        if "volatility" in c.lower()
        and c not in ["volatility_avg", f"price_volatility_{window}d", "relative_volatility"]
    ]

    if vol_cols:
        # Calculate average across available volatility columns
        out["volatility_avg"] = out[vol_cols].mean(axis=1)

    return out


def engineer_revenue_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate revenue CAGR (Compound Annual Growth Rate).

    Supports both production columns (_ltm suffix) and simple test columns.

    Calculates:
    - revenue_cagr_1y: 1-year CAGR
    - revenue_cagr_3y: 3-year CAGR (if 3-year historical data available)
    - revenue_cagr_5y: 5-year CAGR (if 5-year historical data available)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with revenue CAGR features added if source columns exist
    """
    out = df.copy()
    cols = set(out.columns)

    # Support simple test column names
    if {"revenue_current", "revenue_1y_ago"}.issubset(cols):
        # CAGR_1y: (current / 1y_ago)^(1/1) - 1
        out["revenue_cagr_1y"] = (out["revenue_current"] / out["revenue_1y_ago"]) - 1.0
        out["revenue_cagr_1y"] = out["revenue_cagr_1y"].replace([np.inf, -np.inf], np.nan)

    if {"revenue_current", "revenue_3y_ago"}.issubset(cols):
        # CAGR_3y: (current / 3y_ago)^(1/3) - 1
        out["revenue_cagr_3y"] = (out["revenue_current"] / out["revenue_3y_ago"]) ** (
            1.0 / 3.0
        ) - 1.0
        out["revenue_cagr_3y"] = out["revenue_cagr_3y"].replace([np.inf, -np.inf], np.nan)

    # Support alternate column names
    if {"revenue", "revenue_3y_ago"}.issubset(cols) and "revenue_cagr_3y" not in out.columns:
        out["revenue_cagr_3y"] = (out["revenue"] / out["revenue_3y_ago"]) ** (1.0 / 3.0) - 1.0
        out["revenue_cagr_3y"] = out["revenue_cagr_3y"].replace([np.inf, -np.inf], np.nan)

    if {"revenue", "revenue_5y_ago"}.issubset(cols):
        # CAGR_5y: (current / 5y_ago)^(1/5) - 1
        out["revenue_cagr_5y"] = (out["revenue"] / out["revenue_5y_ago"]) ** (1.0 / 5.0) - 1.0
        out["revenue_cagr_5y"] = out["revenue_cagr_5y"].replace([np.inf, -np.inf], np.nan)

    # Support production column names with _ltm suffix
    if {"total_revenues_ltm", "total_revenues_1fy"}.issubset(cols):
        if "revenue_cagr_1y" not in out.columns:  # Don't overwrite if already created
            # Simple growth rate: (current - previous) / previous
            out["revenue_cagr_1y"] = _safe_div(
                out["total_revenues_ltm"] - out["total_revenues_1fy"], out["total_revenues_1fy"]
            )

    return out


def build_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]:
    """Build feature matrix and target variable from DataFrame.

    Returns X (features), y (target), numeric_features, categorical_features.
    Tries to use 'price_target' or 'price_target_median' as y if present;
    otherwise returns y=None.

    Removes identifier columns (ticker, isin, name, description) from features.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of:
        - X: Feature DataFrame
        - y: Target Series (or None if no target column found)
        - numeric_features: List of numeric feature column names
        - categorical_features: List of categorical feature column names
    """
    # PREVENTIVE: Check for duplicate columns and remove them
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Found duplicate columns: {duplicates}")
        logger.warning("Keeping only the first occurrence of each column")
        # Keep only the first occurrence of each column
        df = df.loc[:, ~df.columns.duplicated()]

    y = None
    target_candidates = ["price_target", "price_target_median"]
    y_name = next((t for t in target_candidates if t in df.columns), None)
    if y_name:
        # DEFENSIVE: Ensure we get a Series, not a DataFrame
        y_series = df[y_name]
        if isinstance(y_series, pd.DataFrame):
            # If multiple columns with same name (shouldn't happen after dedup above), take first
            logger.warning(
                f"Column '{y_name}' returned DataFrame instead of Series, taking first column"
            )
            y = pd.to_numeric(y_series.iloc[:, 0], errors="coerce")
        else:
            y = pd.to_numeric(y_series, errors="coerce")

    X = df.copy()
    if y_name:
        X = X.drop(columns=[y_name])

    # Very simple heuristic for feature types
    categorical_features = [c for c in X.columns if X[c].dtype == "object"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    # Drop obvious identifiers from X if present
    drop_cols = [c for c in ["ticker", "isin", "name", "description"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        categorical_features = [c for c in categorical_features if c not in drop_cols]
        numeric_features = [c for c in numeric_features if c not in drop_cols]

    return X, y, numeric_features, categorical_features


__all__ = [
    "_safe_div",
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
]
