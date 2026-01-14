"""
Imputation module for finance_ml.ml_workflow.preprocessing.

This module provides a comprehensive 6-step imputation strategy for financial data:
1. Zero imputation for exceptional event columns (48 columns)
2. Sector-aware KNN imputation for core financial metrics (148 columns)
3. Price imputation for price target columns
4. Median imputation for all remaining numerical columns

The strategy ensures zero missing values while preserving financial data semantics.

Functions:
    - get_zero_imputation_columns: List of columns for zero imputation
    - get_knn_imputation_columns: List of columns for KNN imputation
    - impute_missing_values_knn_sector: Sector-aware KNN imputation
    - apply_zero_imputation: Apply zero imputation (Step 1)
    - apply_knn_imputation_enhanced: Apply KNN imputation (Step 2)
    - apply_price_imputation: Apply price imputation (Step 3)
    - apply_median_imputation: Apply median imputation (Step 4)
    - apply_enhanced_imputation_strategy_4step: Complete 6-step pipeline

Extracted from finance_ml.ml_workflow.advanced_preprocessing as part of Phase 9.1 refactor.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    list_non_recurring_cols,
    list_knn_imputable_cols,
    list_count_cols,
    list_price_cols,
    list_categorical_cols,
    list_date_cols,
)
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
)

# Initialize logger
logger = logging.getLogger(__name__)


def get_schema_aligned_columns_by_role(
    df: pd.DataFrame,
    roles: List[str],
    dtypes: Optional[List[str]] = None,
) -> List[str]:
    """Get DataFrame columns filtered by schema role and dtype.

    This helper function provides schema-aware column selection for imputation,
    ensuring columns are selected based on their semantic role rather than
    name patterns.

    Args:
        df: Input DataFrame
        roles: List of schema roles to include (e.g., ['feature', 'ratio', 'percentage'])
        dtypes: Optional list of schema dtypes to include (e.g., ['float', 'int'])

    Returns:
        List of column names from df that match the specified roles and dtypes

    Example:
        >>> # Get all numeric feature columns for KNN imputation
        >>> knn_cols = get_schema_aligned_columns_by_role(
        ...     df,
        ...     roles=['feature', 'market_value', 'ratio', 'percentage'],
        ...     dtypes=['float', 'int']
        ... )
    """
    from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name

    if dtypes is None:
        dtypes = ["float", "int", "bool"]

    matched_columns = []

    for col in df.columns:
        normalized = normalize_column_name(col)
        schema_entry = COLUMN_SCHEMA.get(normalized, {})

        col_role = schema_entry.get("role", "")
        col_dtype = schema_entry.get("dtype", "")

        if col_role in roles and col_dtype in dtypes:
            matched_columns.append(col)

    return matched_columns


def get_zero_imputation_columns() -> List[str]:
    """Return list of columns for zero imputation (Step 1 of 6-step strategy).

    These columns represent rare/exceptional non-recurring events where missing
    values typically mean the event did not occur. Zero is the economically
    correct imputation.

    Now uses schema-based selection via role='non_recurring' to ensure
    alignment with COLUMN_SCHEMA and prevent drift.

    Returns:
        List of column names for zero imputation (non-recurring items only)
    """
    # Primary: Schema-driven selection
    schema_columns = list_non_recurring_cols()

    # Fallback: Hardcoded list for backward compatibility if schema is incomplete
    hardcoded_columns = [
        # Impairment of Goodwill (all periods)
        "impairment_of_goodwill_fq",
        "impairment_of_goodwill_ltm",
        "impairment_of_goodwill_1fy",
        "impairment_of_goodwill_fy",
        "impairment_of_goodwill_5yavgfq",
        # Asset Writedown (all periods)
        "asset_writedown_fq",
        "asset_writedown_ltm",
        "asset_writedown_fy",
        "asset_writedown_1fy",
        "asset_writedown_5yavgfq",
        # Restructuring Charges (all periods)
        "restructuring_charges_fq",
        "restructuring_charges_ltm",
        "restructuring_charges_fy",
        "restructuring_charges_1fy",
        "restructuring_charges_5yavgfq",
        # Merger & Restructuring Charges (all periods)
        "merger_and_restructuring_charges_fq",
        "merger_and_restructuring_charges_ltm",
        "merger_and_restructuring_charges_fy",
        "merger_and_restructuring_charges_1fy",
        "merger_and_restructuring_charges_5yavgfq",
        # Other exceptional items
        "gain_loss_on_sale_of_assets_ltm",
        "other_unusual_items_total_ltm",
    ]

    # Merge schema-based and hardcoded, removing duplicates
    all_columns = list(set(schema_columns + hardcoded_columns))

    logger.debug(
        f"Zero imputation columns: {len(all_columns)} total "
        f"({len(schema_columns)} from schema, {len(hardcoded_columns)} hardcoded)"
    )

    return all_columns


def get_median_imputation_columns() -> List[str]:
    """Return columns that should use median imputation (recurring items).

    These are recurring operational items where median is appropriate:
    - Count columns (discrete integers like employee counts, shares outstanding)
    - R&D expenses, Capital expenditure, Interest expense/income
    - Balance sheet items (goodwill, intangible assets)
    - Trading metrics (volume, short interest)

    Now uses schema-based selection to ensure 100% coverage of count columns.

    IMPORTANT: Excludes zero-imputation columns to prevent overwriting zero values
    set in Step 1 (non-recurring exceptional items).

    Returns:
        List of column names for median imputation
    """
    # Get zero-imputation columns to exclude (prevent overwriting Step 1)
    zero_imputation_cols = set(get_zero_imputation_columns())

    # Primary: Schema-driven count columns
    count_columns = [col for col in list_count_cols() if col not in zero_imputation_cols]

    # Hardcoded recurring operational items (not in count role)
    hardcoded_columns = [
        # R&D Expenses - recurring operational expense
        "r_d_expenses_ltm",
        "r_d_expenses_fy",
        "r_d_expenses_fq",
        "r_d_expenses_5yavgfq",
        # Capital Expenditure - recurring investment
        "capital_expenditure_ltm",
        "capital_expenditure_fy",
        "capital_expenditure_fq",
        "capital_expenditure_1fy",
        "capital_expenditure_5yavgfq",
        # Interest expense/income - recurring financing
        "interest_expense_total_ltm",
        "interest_income_on_investments_ltm",
        # Cash Acquisitions - use median (0 is misleading for non-acquirers)
        "cash_acquisitions_fq",
        "cash_acquisitions_ltm",
        "cash_acquisitions_fy",
        "cash_acquisitions_1fy",
        "cash_acquisitions_5yavgfq",
        # Trading metrics - use median, not zero
        "volume_shrs",
        "short_int",
        # Balance sheet items - use median
        "goodwill_fq",
        "goodwill_ltm",
        "goodwill_fy",
        "goodwill_1fy",
        "goodwill_5yavgfq",
        "gross_intangible_assets_ltm",
        "gross_intangible_assets_fy",
        "gross_intangible_assets_5yavgfq",
    ]

    # Filter hardcoded to exclude zero-imputation columns
    filtered_hardcoded = [col for col in hardcoded_columns if col not in zero_imputation_cols]

    # Merge schema-based and filtered hardcoded
    all_columns = list(set(count_columns + filtered_hardcoded))

    logger.debug(
        f"Median imputation columns: {len(all_columns)} total "
        f"({len(count_columns)} count columns from schema, "
        f"{len(filtered_hardcoded)} hardcoded recurring items)"
    )

    return all_columns


def get_categorical_imputation_config() -> dict:
    """Return configuration for categorical column imputation strategies.

    Now uses schema-based selection via list_categorical_cols() to ensure
    alignment with COLUMN_SCHEMA and prevent drift.

    Returns:
        Dictionary mapping column names to imputation strategies
    """
    # Schema-driven categorical columns
    schema_categorical = list_categorical_cols()

    # Build config dynamically based on schema
    config = {}

    # Ordinal categories (have natural ordering)
    ordinal_mappings = {
        "style_class": ["Value", "Blend", "Growth"],
        "size_class": ["Small", "Mid", "Large"],
        "eps_surprise_magnitude": ["Small", "Medium", "Large"],
    }

    # Nominal categories (one-hot encoding recommended)
    onehot_columns = {"sector", "industry", "region", "country", "trading_country", "exchange"}

    # Status/flag categories (most frequent)
    most_frequent_columns = {
        "unit",
        "next_earnings_status",
        "next_earnings_when",
        "dividend_record_frequency",
        "dividend_record_currency",
        "fy_end",
        "next_fiscal_quarter",
    }

    # Identifier columns (constant fallback)
    constant_columns = {
        "ticker": "N/A",
        "isin": "N/A",
        "name": "Unknown",
        "description": "No description available",
    }

    # Build config
    for col in schema_categorical:
        if col in ordinal_mappings:
            config[col] = ("ordinal", ordinal_mappings[col])
        elif col in onehot_columns:
            config[col] = "onehot"
        elif col in most_frequent_columns:
            config[col] = "most_frequent"
        elif col in constant_columns:
            config[col] = ("constant", constant_columns[col])
        else:
            # Default to most_frequent for unknown categoricals
            config[col] = "most_frequent"

    # Add any constant columns not in schema
    for col, value in constant_columns.items():
        if col not in config:
            config[col] = ("constant", value)

    return config


def _get_schema_aligned_fallback(col: str) -> str:
    """Get schema-aligned fallback value for categorical column.

    Uses domain knowledge from create_equities_schema.sql to provide
    semantically appropriate fallback values.

    Args:
        col: Column name (normalized)

    Returns:
        Appropriate fallback value for the column
    """
    # Schema-aware fallback values based on create_equities_schema.sql
    fallback_map = {
        # CATEGORICAL: GICS classification
        "sector": "Other",
        "industry": "Other",
        # CATEGORICAL: Geographic classification
        "region": "Other",
        "country": "Unknown",
        "trading_country": "Unknown",
        # CATEGORICAL: Exchange
        "exchange": "OTC",  # Over-the-counter as fallback
        # CATEGORICAL: Investment style
        "style_class": "Blend",  # Neutral between Value/Growth
        "size_class": "Mid",  # Mid-cap as neutral
        # CATEGORICAL: Status fields
        "next_earnings_status": "Unknown",
        "next_earnings_report": "Unknown",  # Full Year/Interim
        "next_earnings_when": "Unknown",  # Before/After market
        # CATEGORICAL: Dividend record
        "dividend_record_frequency": "None",
        "dividend_record_currency": "USD",  # Default to USD
        # CATEGORICAL: Fiscal calendar
        "fy_end": "Dec",  # Most common fiscal year end
        "current_fiscal_quarter": "Q4",
        "next_fiscal_quarter": "Q1",
        # Identifier columns
        "ticker": "N/A",
        "isin": "N/A",
        "name": "Unknown",
        "description": "No description available",
        "unit": "N/A",
    }

    # Return schema-aligned value or generic fallback
    return fallback_map.get(col, "Unknown")


def get_dividend_imputation_config() -> dict:
    """Return imputation configuration for dividend-related columns.

    Business Logic:
        - Numeric dividend columns: Zero-fill (missing = no dividend)
        - Categorical dividend columns: N/A or 'None' fill
        - Date columns: Skip (handled in datetime imputation)

    Returns:
        Dictionary with 'zero_fill', 'na_fill', and 'skip' column sets
    """
    return {
        # Numeric columns: Zero means no dividend
        "zero_fill": {
            "dividend_record_amount",
            "dividend_streak",
            "common_dividends_paid_ltm",
            "common_dividends_paid_fy",
            "dividend_per_share_ltm",
            "div_yield_ind",
            "div_yield_ltm",
            "div_yield_1fyind",
            "div_yield_2fyind",
            "div_yield_3fyind",
            "div_yield_4fyind",
            "div_yield_5fyind",
        },
        # Categorical columns: N/A indicates no dividend program
        "na_fill": {
            "dividend_record_currency": "N/A",
            "dividend_record_frequency": "None",
        },
        # Date columns: Skip zero-fill (will be NaT for non-payers)
        "skip": {
            "dividend_record_announce_date",
            "dividend_record_payable_date",
            "dividend_record_record_date",
            "dividend_record_ex_date",
        },
    }


def get_analyst_rating_imputation_config() -> dict:
    """Return imputation configuration for analyst rating columns.

    Business Logic:
        - Count columns: Zero-fill (missing = no analyst coverage)
        - This prevents KNN from generating artificial coverage

    Returns:
        Dictionary with 'zero_fill' column set
    """
    return {
        "zero_fill": {
            "num_strong_sell_ratings",
            "num_strong_buys_ratings",
            "num_hold_ratings",
            "num_buys_ratings",
            "num_sell_ratings",
        },
    }


def get_financial_statement_zero_fill_config() -> dict:
    """Return zero-fill configuration for income statement and balance sheet items.

    Business Logic:
        Line items that may not exist for all companies should be zero-filled
        to prevent imputation from generating non-existent items.

    Returns:
        Dictionary with 'income_statement' and 'balance_sheet' column sets
    """
    return {
        # Income statement items (may not apply to all companies)
        "income_statement": {
            # R&D (tech/pharma have it, others may not)
            "randd_expenses_ltm",
            "r_d_expenses_ltm",
            "r_d_expenses_fy",
            "r_d_expenses_fq",
            "r_d_expenses_5yavgfq",
            # Marketing (discretionary reporting)
            "marketing_expenses",
            "marketing_expenses_fq",
            "marketing_expenses_fy",
            "marketing_expenses_1fy",
            "marketing_expenses_5yavgltm",
            # Interest income (requires investments)
            "interest_income_on_investments_ltm",
        },
        # Balance sheet items (may not exist for all companies)
        "balance_sheet": {
            # Goodwill & Intangibles (acquisition-driven)
            "goodwill_fq",
            "goodwill_ltm",
            "goodwill_fy",
            "goodwill_1fy",
            "goodwill_5yavgfq",
            "intangible_assets",
            "gross_intangible_assets_ltm",
            "gross_intangible_assets_fy",
            "gross_intangible_assets_5yavgfq",
            # Inventory (service companies may have zero)
            "inventory_ltm",
            "inventory_fq",
            "inventory_fy",
            "inventory_5yavgfq",
        },
    }


def apply_ordinal_encoding(
    df: pd.DataFrame,
    column: str,
    categories: List[str],
    handle_unknown: str = "use_encoded_value",
) -> pd.DataFrame:
    """Apply ordinal encoding to a categorical column.

    Preserves the natural ordering of categories (e.g., Small < Mid < Large).

    Args:
        df: Input DataFrame
        column: Column name to encode
        categories: Ordered list of categories (lowest to highest)
        handle_unknown: Strategy for unknown values ('use_encoded_value' or 'error')

    Returns:
        DataFrame with ordinal encoded column (original column replaced)

    Example:
        >>> df = apply_ordinal_encoding(
        ...     df, 'size_class', ['Small', 'Mid', 'Large']
        ... )
        >>> # Small=0, Mid=1, Large=2
    """
    from sklearn.preprocessing import OrdinalEncoder

    result = df.copy()

    if column not in result.columns:
        logger.warning(f"Column '{column}' not found for ordinal encoding")
        return result

    # Handle missing values first
    missing_mask = result[column].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        # Fill with middle category (most conservative)
        middle_idx = len(categories) // 2
        result.loc[missing_mask, column] = categories[middle_idx]
        logger.info(
            f"Filled {n_missing} missing values in '{column}' with '{categories[middle_idx]}'"
        )

    # Apply ordinal encoding
    encoder = OrdinalEncoder(
        categories=[categories],
        handle_unknown=handle_unknown,
        unknown_value=len(categories) // 2,  # Middle value for unknowns
    )

    # Reshape for sklearn
    values = result[[column]].values
    encoded = encoder.fit_transform(values)
    result[column] = encoded.flatten().astype(int)

    logger.info(
        f"Applied ordinal encoding to '{column}': {dict(zip(categories, range(len(categories))))}"
    )

    return result


def apply_onehot_encoding(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
    sparse_output: bool = False,
    min_frequency: Optional[float] = 0.01,
) -> pd.DataFrame:
    """Apply one-hot encoding to nominal categorical columns.

    Creates binary columns for each category. Rare categories below
    min_frequency are grouped into 'other' to prevent sparse features.

    Args:
        df: Input DataFrame
        columns: List of columns to one-hot encode
        drop_first: Drop first category to avoid multicollinearity (default: True)
        sparse_output: Return sparse matrix (default: False for dense)
        min_frequency: Minimum frequency threshold for category inclusion

    Returns:
        DataFrame with one-hot encoded columns (original columns removed)

    Example:
        >>> df = apply_onehot_encoding(df, ['sector', 'region'])
        >>> # Creates: sector_Technology, sector_Healthcare, region_US, etc.
    """
    from sklearn.preprocessing import OneHotEncoder

    result = df.copy()
    available_cols = [c for c in columns if c in result.columns]

    if not available_cols:
        logger.warning("No specified columns found for one-hot encoding")
        return result

    # Handle missing values
    for col in available_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna("Unknown")

    # Apply one-hot encoding
    encoder = OneHotEncoder(
        drop="first" if drop_first else None,
        sparse_output=sparse_output,
        handle_unknown="ignore",
        min_frequency=min_frequency,
    )

    encoded = encoder.fit_transform(result[available_cols])

    # Get feature names
    if hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out(available_cols)
    else:
        feature_names = [
            f"{col}_{cat}" for col, cats in zip(available_cols, encoder.categories_) for cat in cats
        ]

    # Convert to DataFrame
    if sparse_output:
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded, columns=feature_names, index=result.index
        )
    else:
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=result.index)

    # Drop original columns and add encoded columns
    result = result.drop(columns=available_cols)
    result = pd.concat([result, encoded_df], axis=1)

    logger.info(f"One-hot encoded {len(available_cols)} columns into {len(feature_names)} features")

    return result


def apply_categorical_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "most_frequent",
    fill_value: Optional[str] = None,
) -> pd.DataFrame:
    """Apply imputation to categorical/string columns (Step 5 of enhanced strategy).

    Args:
        df: Input DataFrame
        columns: List of categorical columns to impute. If None, auto-detects object dtype columns
        strategy: Imputation strategy - 'most_frequent', 'constant', or 'drop'
        fill_value: Value to use for 'constant' strategy

    Returns:
        DataFrame with imputed categorical columns

    Examples:
        >>> # Impute categorical columns with mode
        >>> df = apply_categorical_imputation(df, strategy='most_frequent')
        >>>
        >>> # Impute specific columns with constant
        >>> df = apply_categorical_imputation(
        ...     df,
        ...     columns=['flag', 'ticker'],
        ...     strategy='constant',
        ...     fill_value='Unknown'
        ... )
    """
    result = df.copy()

    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()
        # Exclude date columns that should be handled separately
        date_keywords = ["date", "updated", "earnings"]
        columns = [
            col for col in columns if not any(keyword in col.lower() for keyword in date_keywords)
        ]

    if len(columns) == 0:
        logger.info("No categorical columns found for imputation")
        return result

    # Filter to columns that exist and have missing values
    columns_to_impute = [
        col for col in columns if col in result.columns and result[col].isna().any()
    ]

    if len(columns_to_impute) == 0:
        logger.info("No missing values found in specified categorical columns")
        return result

    total_imputed = 0

    for col in columns_to_impute:
        n_missing = result[col].isna().sum()
        is_categorical = isinstance(result[col].dtype, pd.CategoricalDtype)

        if strategy == "most_frequent":
            # Use mode (most common value)
            mode_value = result[col].mode()
            if len(mode_value) > 0:
                fill_val = mode_value[0]

                # For categorical columns, ensure fill value is in categories
                if is_categorical and fill_val not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([fill_val])

                result[col] = result[col].fillna(fill_val)
                logger.debug(f"Imputed {n_missing} values in '{col}' with mode: {fill_val}")
            else:
                # Fallback to constant if no mode exists
                fallback_value = _get_schema_aligned_fallback(col)

                # For categorical columns, add fallback value to categories
                if is_categorical:
                    if fallback_value not in result[col].cat.categories:
                        result[col] = result[col].cat.add_categories([fallback_value])

                result[col] = result[col].fillna(fallback_value)
                logger.debug(
                    f"Imputed {n_missing} values in '{col}' with '{fallback_value}' (no mode found)"
                )

        elif strategy == "constant":
            fill = fill_value if fill_value is not None else _get_schema_aligned_fallback(col)

            # For categorical columns, ensure fill value is in categories
            if is_categorical:
                if fill not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([fill])

            result[col] = result[col].fillna(fill)
            logger.debug(f"Imputed {n_missing} values in '{col}' with constant: {fill}")

        elif strategy == "drop":
            # Drop rows with missing values (use cautiously)
            result = result.dropna(subset=[col])
            logger.debug(f"Dropped {n_missing} rows with missing values in '{col}'")

        total_imputed += n_missing

    logger.info(
        f"Applied {strategy} imputation to {total_imputed} categorical missing values across {len(columns_to_impute)} columns"
    )
    return result


def apply_datetime_imputation_and_formatting(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    strategy: str = "forward_fill",
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Apply imputation and formatting to datetime columns (Step 6 of enhanced strategy).

    This ensures proper datetime formatting for engineer_temporal_features() in Phase 9.3.

    Args:
        df: Input DataFrame
        date_columns: List of date columns. If None, auto-detects from schema (role='date')
        strategy: Imputation strategy - 'forward_fill', 'backward_fill', 'median', 'constant'
        reference_date: Optional reference date for 'constant' strategy

    Returns:
        DataFrame with properly formatted datetime columns and no NaN values

    Examples:
        >>> # Format and impute date columns
        >>> df = apply_datetime_imputation_and_formatting(
        ...     df,
        ...     date_columns=['last_updated', 'income_statement_report_date', 'next_earnings',],
        ...     strategy='forward_fill'
        ... )
    """
    result = df.copy()

    # Schema-driven date column detection
    if date_columns is None:
        # Primary: Use schema-based detection
        schema_date_cols = list_date_cols()
        date_columns = [col for col in schema_date_cols if col in result.columns]

        # If no schema matches in DataFrame, log and return
        if not date_columns:
            logger.info("No date columns found in DataFrame matching schema")
            return result

        logger.debug(f"Schema-detected date columns: {date_columns}")

    # Filter to columns that exist
    date_columns = [col for col in date_columns if col in result.columns]

    total_imputed = 0

    for col in date_columns:
        n_missing_initial = result[col].isna().sum()

        # Step 1: Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(result[col]):
            try:
                # Try parsing with various formats
                result[col] = pd.to_datetime(result[col], errors="coerce")
                logger.debug(f"Converted '{col}' to datetime format")
            except Exception as e:
                logger.warning(f"Could not convert '{col}' to datetime: {e}")
                continue

        # Step 2: Impute missing datetime values
        if result[col].isna().any():
            n_missing = result[col].isna().sum()

            if strategy == "forward_fill":
                result[col] = result[col].ffill()
                # If still missing (at start), use backward fill
                if result[col].isna().any():
                    result[col] = result[col].bfill()
                logger.debug(f"Forward-filled {n_missing} missing dates in '{col}'")

            elif strategy == "backward_fill":
                result[col] = result[col].bfill()
                # If still missing (at end), use forward fill
                if result[col].isna().any():
                    result[col] = result[col].ffill()
                logger.debug(f"Backward-filled {n_missing} missing dates in '{col}'")

            elif strategy == "median":
                # Use median timestamp
                median_ts = result[col].dropna().median()
                result[col] = result[col].fillna(median_ts)
                logger.debug(
                    f"Median-imputed {n_missing} missing dates in '{col}' with {median_ts}"
                )

            elif strategy == "constant":
                # Use reference date or current date
                fill_date = reference_date if reference_date else pd.Timestamp.now()
                result[col] = result[col].fillna(fill_date)
                logger.debug(
                    f"Constant-imputed {n_missing} missing dates in '{col}' with {fill_date}"
                )

            # If still missing after imputation, use current date as last resort
            if result[col].isna().any():
                n_still_missing = result[col].isna().sum()
                result[col] = result[col].fillna(pd.Timestamp.now())
                logger.warning(
                    f"Used current date to fill {n_still_missing} remaining missing dates in '{col}'"
                )

            total_imputed += n_missing

    logger.info(
        f"Applied datetime imputation and formatting to {total_imputed} missing values across {len(date_columns)} date columns"
    )
    return result


def get_knn_imputation_columns() -> List[str]:
    """Return list of columns for KNN imputation (Step 2 of 6-step strategy).

    These are core financial metrics where KNN can leverage sector relationships
    and correlations to provide better estimates than simple statistics.

    Now uses schema-based selection to ensure 100% coverage of numeric features.
    Includes columns with roles: feature, market, financial_statement,
    balance_sheet, cash_flow, ratio, percentage.

    IMPORTANT: Excludes zero-imputation columns to prevent overwriting zero values
    set in Step 1 (non-recurring exceptional items).

    Returns:
        List of column names for KNN imputation (dynamically generated from schema)
    """
    # Get zero-imputation columns to exclude (prevent overwriting Step 1)
    zero_imputation_cols = set(get_zero_imputation_columns())

    # Get count columns to exclude (use median instead)
    count_cols = set(list_count_cols())

    # Primary: Schema-driven selection
    schema_columns = list_knn_imputable_cols()

    # Filter out zero-imputation and count columns
    knn_columns = [
        col for col in schema_columns if col not in zero_imputation_cols and col not in count_cols
    ]

    logger.debug(
        f"KNN imputation columns: {len(knn_columns)} total "
        f"(from {len(schema_columns)} schema columns, "
        f"excluded {len(zero_imputation_cols)} zero-imputation, "
        f"{len(count_cols)} count columns)"
    )

    return knn_columns


def impute_missing_values_knn_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sector_column: str = "sector",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using sector-aware KNN imputation.

    This enhanced KNN imputation performs imputation separately within each sector,
    ensuring that missing values are filled using only neighbors from the same sector.
    This preserves sector-specific characteristics and improves imputation quality.

    Schema-aligned: Uses COLUMN_SCHEMA to ensure only numeric columns with appropriate
    roles (feature, market_value, ratio, percentage) are imputed via KNN.

    Args:
        df: Input DataFrame
        columns: Columns to impute (default: all numeric columns)
        sector_column: Name of the sector column for grouping (default: 'sector')
        n_neighbors: Number of neighbors to use for KNN (default: 5)

    Returns:
        DataFrame with imputed values

    Examples:
        >>> df_imputed = impute_missing_values_knn_sector(
        ...     df,
        ...     columns=['revenue', 'ebitda'],
        ...     sector_column='sector',
        ...     n_neighbors=5
        ... )
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove sector column from imputation if present
    columns = [col for col in columns if col != sector_column]

    # FIX: Filter columns to only those that exist in the DataFrame
    existing_columns = [col for col in columns if col in result.columns]

    if len(existing_columns) < len(columns):
        missing_cols = set(columns) - set(existing_columns)
        logger.debug(
            f"KNN imputation: Skipping {len(missing_cols)} columns not in DataFrame: "
            f"{list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}"
        )

    columns = existing_columns  # Use filtered list for all subsequent operations

    if not columns:
        logger.warning("No numeric columns to impute")
        return result

    # SCHEMA-ALIGNED FIX: Filter to only truly numeric columns based on schema
    # This prevents string/object columns from causing KNN failures
    numeric_columns = []
    for col in columns:
        # Check actual dtype first
        if not pd.api.types.is_numeric_dtype(result[col]):
            # Try to coerce to numeric
            result[col] = pd.to_numeric(result[col], errors="coerce")
            logger.debug(f"Pre-processing: Coerced column '{col}' to numeric")

        # Verify dtype after potential coercion
        if pd.api.types.is_numeric_dtype(result[col]):
            numeric_columns.append(col)
        else:
            logger.debug(f"Skipping non-numeric column '{col}' (dtype: {result[col].dtype})")

    columns = numeric_columns

    if not columns:
        logger.warning("No numeric columns available for KNN imputation after dtype filtering")
        return result

    # Check if sector column exists
    if sector_column not in df.columns:
        logger.warning(
            f"Sector column '{sector_column}' not found, falling back to global KNN imputation"
        )
        # Fall back to global KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        # FIX: Use explicit column list for imputation
        cols_for_knn = [c for c in columns if c in result.columns]
        imputed_values = imputer.fit_transform(result[cols_for_knn])
        result[cols_for_knn] = imputed_values
        logger.info(
            f"Applied global KNN imputation (k={n_neighbors}) to {len(cols_for_knn)} columns"
        )
        return result

    # Perform sector-aware KNN imputation
    sectors = df[sector_column].dropna().unique()
    imputed_count = 0
    skipped_sectors = []

    for sector in sectors:
        sector_mask = result[sector_column] == sector

        # FIX: Capture exact columns that exist at slice time
        cols_to_impute = [c for c in columns if c in result.columns]

        # FIX: Create an explicit copy with known columns to avoid view/copy ambiguity
        sector_data = result.loc[sector_mask, cols_to_impute].copy()

        # Verify column alignment
        if list(sector_data.columns) != cols_to_impute:
            logger.warning(
                f"Sector '{sector}': Column mismatch detected. "
                f"Expected {len(cols_to_impute)}, got {len(sector_data.columns)}"
            )
            cols_to_impute = list(sector_data.columns)  # Use actual columns

        # Check if sector has enough samples for KNN
        n_samples = sector_data.shape[0]
        if n_samples < 2:
            logger.warning(
                f"Sector '{sector}' has only {n_samples} sample(s), skipping KNN imputation"
            )
            skipped_sectors.append((sector, "insufficient_samples"))
            continue

        # Validate no object dtypes remain (should have been handled above)
        remaining_objects = sector_data.select_dtypes(include=["object"]).columns.tolist()
        if remaining_objects:
            logger.warning(
                f"Sector '{sector}': Skipping KNN due to non-numeric columns: {remaining_objects}"
            )
            skipped_sectors.append((sector, "non_numeric_columns"))
            continue

        # Adjust n_neighbors if sector has fewer samples
        k = min(n_neighbors, n_samples - 1)
        if k < 1:
            logger.warning(f"Sector '{sector}': k={k} is too small, skipping")
            skipped_sectors.append((sector, "k_too_small"))
            continue

        # Check if sector has any missing values
        if not sector_data.isna().any().any():
            continue

        # Apply KNN imputation to this sector
        imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
        try:
            # FIX: Store column order BEFORE numpy conversion
            impute_col_order = list(sector_data.columns)
            sector_index = sector_data.index.copy()

            # Perform imputation
            sector_imputed = imputer.fit_transform(sector_data.values)

            # FIX: Create DataFrame with explicit column/index alignment
            sector_imputed_df = pd.DataFrame(
                sector_imputed, index=sector_index, columns=impute_col_order
            )

            # FIX: Write back using explicit column list from imputed result
            result.loc[sector_mask, impute_col_order] = sector_imputed_df.values
            imputed_count += 1

        except ValueError as ve:
            # Catch shape mismatch errors specifically
            error_msg = str(ve)
            if "Shape of passed values" in error_msg:
                logger.warning(
                    f"KNN imputation failed for sector '{sector}': Shape mismatch. "
                    f"Sector data shape: {sector_data.shape}, columns: {len(impute_col_order)}. "
                    f"Error: {error_msg}. Skipping."
                )
            else:
                logger.warning(f"KNN imputation failed for sector '{sector}': {ve}. Skipping.")
            skipped_sectors.append((sector, str(ve)[:50]))
            continue
        except Exception as e:
            logger.warning(f"KNN imputation failed for sector '{sector}': {e}. Skipping.")
            skipped_sectors.append((sector, str(e)[:50]))
            continue

    # Handle rows with missing sector values using global imputation
    missing_sector_mask = result[sector_column].isna()
    if missing_sector_mask.any():
        missing_sector_data = result.loc[missing_sector_mask, columns].copy()

        if missing_sector_data.isna().any().any():
            k = min(n_neighbors, missing_sector_data.shape[0] - 1)
            if k > 0:
                imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
                try:
                    # FIX: Use same explicit alignment pattern
                    impute_col_order = list(missing_sector_data.columns)
                    missing_index = missing_sector_data.index.copy()

                    missing_imputed = imputer.fit_transform(missing_sector_data.values)

                    missing_imputed_df = pd.DataFrame(
                        missing_imputed,
                        index=missing_index,
                        columns=impute_col_order,
                    )
                    result.loc[missing_sector_mask, impute_col_order] = missing_imputed_df.values
                    logger.info(
                        f"Applied global KNN to {missing_sector_mask.sum()} rows with missing sector"
                    )
                except Exception as e:
                    logger.warning(f"KNN imputation failed for missing sectors: {e}")

    # Log summary including skipped sectors
    if skipped_sectors:
        logger.info(
            f"Sector-aware KNN: {imputed_count} sectors imputed, {len(skipped_sectors)} skipped. "
            f"Skipped reasons: {dict((s[1], sum(1 for x in skipped_sectors if x[1] == s[1])) for s in skipped_sectors)}"
        )
    else:
        logger.info(
            f"Applied sector-aware KNN imputation (k={n_neighbors}) to {imputed_count} sectors "
            f"across {len(columns)} columns"
        )

    return result


def apply_zero_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply zero imputation to specified columns.

    This imputation strategy is appropriate for columns representing rare/exceptional
    events where missing values typically indicate the event did not occur.

    Args:
        df: Input DataFrame
        columns: Columns to zero-impute (default: auto-detect from schema)

    Returns:
        DataFrame with zero-imputed values
    """
    result = df.copy()

    if columns is None:
        columns = get_zero_imputation_columns()

    # Normalize column names to match dataframe
    available_cols = [col for col in columns if col in result.columns]

    if not available_cols:
        logger.warning("No zero-imputation columns found in dataframe")
        return result

    # Apply zero imputation
    for col in available_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            result[col] = result[col].fillna(0)
            logger.debug(f"Zero-imputed {n_missing} values in column '{col}'")

    logger.info(f"Applied zero imputation to {len(available_cols)} columns")
    return result


def apply_knn_imputation_enhanced(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sector_column: str = "sector",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Apply enhanced KNN imputation with sector awareness.

    This is a wrapper around impute_missing_values_knn_sector that works with
    the predefined KNN imputation column list.

    Args:
        df: Input DataFrame
        columns: Columns for KNN imputation (default: auto-detect from schema)
        sector_column: Name of sector column for grouping
        n_neighbors: Number of neighbors for KNN

    Returns:
        DataFrame with KNN-imputed values
    """
    if columns is None:
        columns = get_knn_imputation_columns()

    # Normalize column names and filter to available columns
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        logger.warning("No KNN-imputation columns found in dataframe")
        return df.copy()

    logger.info(f"Applying KNN imputation to {len(available_cols)} columns")

    # Use existing sector-aware KNN imputation
    return impute_missing_values_knn_sector(
        df=df,
        columns=available_cols,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )


def apply_price_imputation(
    df: pd.DataFrame,
    price_column: str = "last_price",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply price imputation (Step 3 of 6-step strategy).

    Imputes price target columns using the current last_price as the best
    available estimate when analyst targets are missing.

    Now uses schema-based selection to ensure 100% coverage of price-related columns.

    Args:
        df: Input DataFrame
        price_column: Column to use for imputation (default: "last_price")
        columns: Price target columns to impute (default: all price/target columns from schema)

    Returns:
        DataFrame with price-imputed values

    Examples:
        >>> # Impute missing price targets from last_price
        >>> df_imputed = apply_price_imputation(df, price_column='last_price')
    """
    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

    result = df.copy()

    if columns is None:
        # Collect all price-related columns from schema (price, target, target_fallback roles)
        price_roles = {"price", "target", "target_fallback"}
        columns = []
        for col, meta in COLUMN_SCHEMA.items():
            role = meta.get("role", "")
            if role in price_roles and col != price_column:
                columns.append(col)

    # Check if price column exists
    if price_column not in result.columns:
        logger.warning(f"Price column '{price_column}' not found in dataframe")
        return result

    # Apply price imputation to available columns
    available_cols = [col for col in columns if col in result.columns]

    if not available_cols:
        logger.warning("No price target columns found in dataframe")
        return result

    for col in available_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            result[col] = result[col].fillna(result[price_column])
            logger.debug(
                f"Price-imputed {n_missing} values in column '{col}' from '{price_column}'"
            )

    logger.info(
        f"Applied price imputation to {len(available_cols)} price columns using '{price_column}'"
    )
    return result


def apply_median_imputation(
    df: pd.DataFrame,
    price_column: str = "last_price",
    priority_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply median imputation (Step 4 of 6-step strategy).

    Fallback imputation strategy that fills any remaining missing values
    in numerical columns with their median values. Can prioritize specific
    columns (e.g., recurring items) before general imputation.

    IMPORTANT: Excludes zero-imputation columns to prevent overwriting zero values
    set in Step 1 (non-recurring exceptional items).

    Args:
        df: Input DataFrame
        price_column: Reference price column to preserve (default: "last_price")
        priority_columns: Optional list of columns to impute first (e.g., recurring items)

    Returns:
        DataFrame with median-imputed values for all remaining missing numerical data

    Examples:
        >>> # Fill recurring items first, then remaining columns
        >>> recurring = get_median_imputation_columns()
        >>> df_complete = apply_median_imputation(df, priority_columns=recurring)
    """
    result = df.copy()

    # Get all numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude semantic price columns (preserve dollar units), but allow the reference price
    classification = classify_columns(result.columns.tolist())
    price_cols = set(classification.get("price", set()))
    if price_column in price_cols:
        price_cols.remove(price_column)
    if price_cols:
        before = len(numeric_cols)
        numeric_cols = [col for col in numeric_cols if col not in price_cols]
        excluded = before - len(numeric_cols)
        if excluded > 0:
            logger.info(
                f"Median imputation: excluded {excluded} semantic price columns to preserve units"
            )

    # CRITICAL FIX: Exclude zero-imputation columns to prevent overwriting Step 1 zeros
    zero_imputation_cols = set(get_zero_imputation_columns())
    if zero_imputation_cols:
        before = len(numeric_cols)
        numeric_cols = [col for col in numeric_cols if col not in zero_imputation_cols]
        excluded = before - len(numeric_cols)
        if excluded > 0:
            logger.info(
                f"Median imputation: excluded {excluded} zero-imputation columns to preserve Step 1 zeros"
            )

    # Handle priority columns first if specified
    if priority_columns:
        priority_cols = [
            col for col in priority_columns if col in numeric_cols and col in result.columns
        ]
        if priority_cols:
            logger.info(
                f"Median imputation: processing {len(priority_cols)} priority columns first"
            )
            for col in priority_cols:
                if result[col].isna().any():
                    n_missing = result[col].isna().sum()
                    median_val = result[col].median()
                    if not pd.isna(median_val):
                        result[col] = result[col].fillna(median_val)
                        logger.debug(
                            f"Priority median-imputed {n_missing} values in '{col}' with {median_val:.4f}"
                        )
            # Remove priority columns from general imputation list
            numeric_cols = [col for col in numeric_cols if col not in priority_cols]

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found in dataframe")
        return result

    total_imputed = 0

    for col in numeric_cols:
        # If column is an integer dtype (including pandas nullable Int64),
        # cast to float before median imputation to avoid TypeError
        if pd.api.types.is_integer_dtype(result[col].dtype) and not pd.api.types.is_bool_dtype(
            result[col].dtype
        ):
            result[col] = result[col].astype("float64")
            logger.debug(f"Cast column '{col}' from integer to float64 for median imputation")

        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            median_val = result[col].median()

            # If median is NaN (e.g., all values are NaN), skip this column
            if pd.isna(median_val):
                logger.debug(f"Skipping median imputation for column '{col}' because median is NaN")
                continue

            result[col] = result[col].fillna(median_val)
            total_imputed += n_missing
            logger.debug(
                f"Median-imputed {n_missing} values in column '{col}' with {float(median_val):.4f}"
            )

    logger.info(f"Applied median imputation to {total_imputed} total missing values")
    return result


def validate_imputation_completeness(
    df: pd.DataFrame, critical_date_columns: Optional[List[str]] = None
) -> dict:
    """Validate that imputation is complete and datetime columns are properly formatted.

    Args:
        df: DataFrame to validate
        critical_date_columns: List of date columns required for temporal features

    Returns:
        Dictionary with validation results including:
        - 'is_complete': bool - True if no missing values
        - 'missing_count': int - Total missing values
        - 'missing_by_type': dict - Missing values by data type
        - 'datetime_formatted': dict - Status of datetime columns
        - 'ready_for_temporal_features': bool - Ready for engineer_temporal_features()
    """
    if critical_date_columns is None:
        critical_date_columns = [
            "last_updated",
            "income_statement_report_date",
            "next_earnings",
            "dividend_record_announce_date",
            "dividend_record_ex_date",
            "dividend_record_payable_date",
            "dividend_record_record_date",
        ]

    missing_total = df.isna().sum().sum()
    missing_numeric = df.select_dtypes(include=[np.number]).isna().sum().sum()
    missing_categorical = df.select_dtypes(include=["object", "category"]).isna().sum().sum()

    # Check datetime formatting
    datetime_status = {}
    for col in critical_date_columns:
        if col in df.columns:
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
            has_missing = df[col].isna().any()
            datetime_status[col] = {
                "is_datetime": is_datetime,
                "has_missing": has_missing,
                "ready": is_datetime and not has_missing,
            }
        else:
            datetime_status[col] = {
                "is_datetime": False,
                "has_missing": True,
                "ready": False,
            }

    # Ready for temporal features if all EXISTING critical date columns are properly formatted
    # (missing columns are acceptable - they just won't have temporal features)
    existing_date_cols = [col for col in critical_date_columns if col in df.columns]
    if existing_date_cols:
        ready_for_temporal = all(datetime_status[col]["ready"] for col in existing_date_cols)
    else:
        # If none of the critical date columns exist, consider ready (no temporal features to create)
        ready_for_temporal = missing_total == 0

    result = {
        "is_complete": missing_total == 0,
        "missing_count": missing_total,
        "missing_by_type": {
            "numeric": missing_numeric,
            "categorical": missing_categorical,
            "other": missing_total - missing_numeric - missing_categorical,
        },
        "datetime_formatted": datetime_status,
        "ready_for_temporal_features": ready_for_temporal,
    }

    # Log results
    if result["is_complete"] and ready_for_temporal:
        logger.info("✓ Imputation validation PASSED - Ready for Phase 9.3 feature engineering")
    else:
        logger.warning(f"✗ Imputation validation FAILED - {missing_total} missing values remain")
        if not ready_for_temporal:
            logger.warning("✗ Date columns not ready for temporal feature engineering")

    return result


def apply_business_rule_imputation(
    df: pd.DataFrame,
    apply_dividend_rules: bool = True,
    apply_analyst_rules: bool = True,
    apply_financial_statement_rules: bool = True,
) -> pd.DataFrame:
    """Apply business-rule-based imputation for domain-specific columns.

    This function should be called BEFORE the main 6-step imputation
    to prevent statistical imputation from distorting these columns.

    Args:
        df: Input DataFrame
        apply_dividend_rules: Zero-fill dividend columns
        apply_analyst_rules: Zero-fill analyst rating columns
        apply_financial_statement_rules: Zero-fill financial statement items

    Returns:
        DataFrame with business-rule imputation applied
    """
    result = df.copy()
    total_filled = 0

    # Dividend columns
    if apply_dividend_rules:
        div_config = get_dividend_imputation_config()

        # Zero-fill numeric dividend columns
        for col in div_config["zero_fill"]:
            if col in result.columns and result[col].isna().any():
                count = int(result[col].isna().sum())
                result[col] = result[col].fillna(0)
                total_filled += count
                logger.debug(f"Dividend zero-fill: {count} values in '{col}'")

        # N/A-fill categorical dividend columns
        for col, fill_val in div_config["na_fill"].items():
            if col in result.columns and result[col].isna().any():
                count = int(result[col].isna().sum())
                if isinstance(result[col].dtype, pd.CategoricalDtype):
                    if fill_val not in result[col].cat.categories:
                        result[col] = result[col].cat.add_categories([fill_val])
                result[col] = result[col].fillna(fill_val)
                total_filled += count
                logger.debug(f"Dividend N/A-fill: {count} values in '{col}'")

    # Analyst rating columns
    if apply_analyst_rules:
        analyst_config = get_analyst_rating_imputation_config()
        for col in analyst_config["zero_fill"]:
            if col in result.columns and result[col].isna().any():
                count = int(result[col].isna().sum())
                result[col] = result[col].fillna(0)
                total_filled += count
                logger.debug(f"Analyst rating zero-fill: {count} values in '{col}'")

    # Financial statement items
    if apply_financial_statement_rules:
        fs_config = get_financial_statement_zero_fill_config()
        for category in ["income_statement", "balance_sheet"]:
            for col in fs_config[category]:
                if col in result.columns and result[col].isna().any():
                    count = int(result[col].isna().sum())
                    result[col] = result[col].fillna(0)
                    total_filled += count
                    logger.debug(f"{category} zero-fill: {count} values in '{col}'")

    logger.info(f"Business-rule imputation: {total_filled} total values filled")
    return result


def validate_imputation_schema_alignment(df: pd.DataFrame) -> dict:
    """Validate that imputation column lists align with schema and DataFrame.

    This diagnostic function checks for:
    1. Columns in imputation lists but not in schema
    2. Columns in imputation lists but not in DataFrame
    3. Schema columns with appropriate roles not in any imputation list
    4. Overlap between imputation strategies (should be zero)

    Args:
        df: Input DataFrame to validate against

    Returns:
        Dictionary with validation results:
        - 'is_aligned': bool - True if fully aligned
        - 'zero_imputation_gaps': List of gap descriptions
        - 'knn_imputation_gaps': List of gap descriptions
        - 'median_imputation_gaps': List of gap descriptions
        - 'overlap_issues': List of columns in multiple strategies
        - 'uncovered_columns': Schema columns not in any imputation list
    """
    results = {
        "is_aligned": True,
        "zero_imputation_gaps": [],
        "knn_imputation_gaps": [],
        "median_imputation_gaps": [],
        "overlap_issues": [],
        "uncovered_columns": [],
    }

    # Get imputation column lists
    zero_cols = set(get_zero_imputation_columns())
    knn_cols = set(get_knn_imputation_columns())
    median_cols = set(get_median_imputation_columns())

    # Check for overlaps (should be none)
    zero_knn_overlap = zero_cols & knn_cols
    zero_median_overlap = zero_cols & median_cols
    knn_median_overlap = knn_cols & median_cols

    if zero_knn_overlap:
        results["overlap_issues"].append(f"Zero/KNN overlap: {list(zero_knn_overlap)[:5]}")
        results["is_aligned"] = False
    if zero_median_overlap:
        results["overlap_issues"].append(f"Zero/Median overlap: {list(zero_median_overlap)[:5]}")
        results["is_aligned"] = False
    if knn_median_overlap:
        # Note: Some overlap is expected (both can handle same columns)
        logger.debug(f"KNN/Median overlap (expected): {len(knn_median_overlap)} columns")

    # Check schema coverage for numeric columns
    numeric_roles = {
        "feature",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
        "count",
        "non_recurring",
    }

    all_imputation_cols = zero_cols | knn_cols | median_cols

    for col, meta in COLUMN_SCHEMA.items():
        role = meta.get("role", "")
        dtype = meta.get("dtype", "")

        # Only check numeric columns with imputable roles
        if role in numeric_roles and dtype in ["float", "int"]:
            if col not in all_imputation_cols:
                results["uncovered_columns"].append(f"{col} (role={role})")
                results["is_aligned"] = False

    # Check DataFrame coverage
    df_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in zero_cols:
        if col in df.columns and col not in df_numeric:
            results["zero_imputation_gaps"].append(f"{col} not numeric in DataFrame")

    for col in knn_cols:
        if col in df.columns and col not in df_numeric:
            results["knn_imputation_gaps"].append(f"{col} not numeric in DataFrame")

    # Log summary
    if results["is_aligned"]:
        logger.info("✓ Imputation schema alignment validation passed")
    else:
        logger.warning(
            f"✗ Imputation schema alignment issues: "
            f"{len(results['overlap_issues'])} overlaps, "
            f"{len(results['uncovered_columns'])} uncovered columns"
        )

    return results


def apply_enhanced_imputation_strategy_6step(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
    price_column: str = "last_price",
    handle_categoricals: bool = True,
    handle_dates: bool = True,
    categorical_strategy: str = "most_frequent",
    date_strategy: str = "forward_fill",
    apply_categorical_encoding: bool = False,
    validate_schema_alignment: bool = True,  # NEW: Validate alignment before imputation
) -> pd.DataFrame:
    """Apply comprehensive 6-step imputation strategy for financial data.

    SCHEMA-ALIGNED APPROACH: This strategy now uses COLUMN_SCHEMA roles
    to automatically select columns for each imputation step, ensuring
    alignment with the canonical schema and preventing drift.

    Steps:
        1. Zero imputation for non-recurring items (role='non_recurring')
        2. Sector-aware KNN for core metrics (role in knn_roles)
        3. Price imputation for price targets (role in price_roles)
        4. Median imputation for count columns and remaining numerics
        5. Categorical imputation (schema categorical columns)
        6. Datetime imputation (schema date columns)

    Args:
        df: Input DataFrame with financial data
        sector_column: Column name for sector grouping in KNN
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Reference price column for price target imputation
        handle_categoricals: Apply Step 5 categorical imputation
        handle_dates: Apply Step 6 datetime imputation
        categorical_strategy: Strategy for categorical imputation
        date_strategy: Strategy for datetime imputation
        apply_categorical_encoding: Apply ordinal/one-hot encoding after imputation
        validate_schema_alignment: Validate column alignment before imputation

    Returns:
        DataFrame with complete imputation applied (zero missing values)
    """
    result = df.copy()
    missing_initial = result.isna().sum().sum()
    logger.info(f"Starting 6-step imputation. Initial missing: {missing_initial}")

    # Optional: Validate schema alignment
    if validate_schema_alignment:
        alignment = validate_imputation_schema_alignment(result)
        if not alignment["is_aligned"]:
            logger.warning(
                f"Schema alignment issues detected: {len(alignment['uncovered_columns'])} uncovered columns. "
                "Proceeding with imputation but results may be incomplete."
            )

    # Step 1: Zero imputation for NON-RECURRING items (schema-driven)
    logger.info("Step 1: Zero imputation for non-recurring exceptional items")
    zero_columns = get_zero_imputation_columns()  # Now schema-driven
    result = apply_zero_imputation(result, columns=zero_columns)
    missing_after_step1 = result.isna().sum().sum()
    logger.info(f"After Step 1: {missing_after_step1} missing values remain")

    # Step 2: Sector-aware KNN imputation for core metrics (schema-driven)
    logger.info("Step 2: Sector-aware KNN imputation for core metrics")
    knn_columns = get_knn_imputation_columns()  # Now schema-driven
    result = impute_missing_values_knn_sector(
        result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
        columns=knn_columns,
    )
    missing_after_step2 = result.isna().sum().sum()
    logger.info(f"After Step 2: {missing_after_step2} missing values remain")

    # Step 3: Price imputation for price targets (schema-driven)
    logger.info("Step 3: Price imputation for price target columns")
    price_cols = list_price_cols()  # Now schema-driven
    result = apply_price_imputation(result, price_column=price_column, columns=price_cols)
    missing_after_step3 = result.isna().sum().sum()
    logger.info(f"After Step 3: {missing_after_step3} missing values remain")

    # Step 4: Median imputation for count columns and remaining numerics (schema-driven)
    logger.info("Step 4: Median imputation (count columns + remaining numerics)")
    median_priority_columns = get_median_imputation_columns()  # Now schema-driven
    result = apply_median_imputation(
        result, price_column=price_column, priority_columns=median_priority_columns
    )
    missing_after_step4 = result.isna().sum().sum()
    logger.info(f"After Step 4: {missing_after_step4} missing values remain")

    # Step 5: Categorical imputation (schema-driven)
    if handle_categoricals:
        logger.info("Step 5: Categorical imputation")
        config = get_categorical_imputation_config()  # Now schema-driven

        for col, strategy in config.items():
            if col not in result.columns:
                continue

            if isinstance(strategy, tuple):
                strat_type, strat_param = strategy
                if strat_type == "ordinal" and apply_categorical_encoding:
                    result = apply_ordinal_encoding(result, col, strat_param)
                elif strat_type == "constant":
                    result[col] = result[col].fillna(strat_param)
            elif strategy == "most_frequent":
                mode_val = result[col].mode(dropna=True)
                if not mode_val.empty:
                    result[col] = result[col].fillna(mode_val.iloc[0])
            elif strategy == "onehot" and apply_categorical_encoding:
                result = apply_onehot_encoding(result, [col])

        missing_after_step5 = result.isna().sum().sum()
        logger.info(f"After Step 5: {missing_after_step5} missing values remain")

    # Step 6: Datetime imputation (schema-driven)
    if handle_dates:
        logger.info("Step 6: Datetime imputation")
        result = apply_datetime_imputation_and_formatting(
            result, strategy=date_strategy
        )  # Now auto-detects from schema
        missing_after_step6 = result.isna().sum().sum()
        logger.info(f"After Step 6: {missing_after_step6} missing values remain")

    # Step 7 (Verification): Check for any remaining missing values and apply emergency fallback
    if missing_after_step6 > 0:
        logger.warning(
            f"Step 7: {missing_after_step6} missing values remaining. Applying emergency fallback."
        )
        # Emergency fallback for anything missed by specific steps
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if pd.isna(median_val):
                    median_val = 0
                result[col] = result[col].fillna(median_val)

    final_missing = result.isna().sum().sum()
    logger.info(f"Imputation complete. Final missing values: {final_missing}")

    return result


def fillna_by_dtype(
    df: pd.DataFrame,
    numeric_fill: float = 0,
    categorical_strategy: str = "mode",
    string_fill: str = "Unknown",
    datetime_strategy: str = "forward_fill",
) -> pd.DataFrame:
    """Fill missing values by respecting column dtypes.

    This function implements a type-aware filling strategy to avoid errors when
    applying numeric fill values to categorical columns. It's the recommended
    approach for handling mixed-type DataFrames.

    Strategy:
        - Numeric columns: Fill with numeric_fill (default: 0)
        - Categorical columns: Fill with mode (most frequent) or add new category
        - String/object columns: Fill with string_fill (default: "Unknown")
        - Datetime columns: Forward fill or fill with current timestamp

    Args:
        df: Input DataFrame with potentially missing values
        numeric_fill: Value to use for numeric columns (default: 0)
        categorical_strategy: Strategy for categorical columns:
            - "mode": Use most frequent value (default)
            - "unknown": Add "Unknown" category and fill
            - "add_value": Add numeric_fill as a new category
        string_fill: String to use for object/string columns (default: "Unknown")
        datetime_strategy: Strategy for datetime columns:
            - "forward_fill": Forward fill (default)
            - "now": Fill with current timestamp

    Returns:
        DataFrame with missing values filled according to dtype

    Examples:
        >>> # Safe filling that respects column types
        >>> df_filled = fillna_by_dtype(df, numeric_fill=0, categorical_strategy="mode")

        >>> # Fill categoricals by adding 0 as a new category
        >>> df_filled = fillna_by_dtype(df, categorical_strategy="add_value")

    Note:
        This function prevents the common error:
        "TypeError: Cannot setitem on a Categorical with a new category (0)"
        which occurs when using df.fillna(0) on DataFrames with categorical columns.
    """
    result = df.copy()

    # 1. Fill numeric columns
    numeric_cols = result.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        result[numeric_cols] = result[numeric_cols].fillna(numeric_fill)
        logger.debug(f"Filled {len(numeric_cols)} numeric columns with {numeric_fill}")

    # 2. Fill categorical columns
    categorical_cols = result.select_dtypes(include=["category"]).columns
    for col in categorical_cols:
        if result[col].isna().any():
            if categorical_strategy == "mode":
                # Use mode (most frequent value)
                mode_val = result[col].mode(dropna=True)
                if not mode_val.empty:
                    fill_val = mode_val.iloc[0]
                    result[col] = result[col].fillna(fill_val)
                    logger.debug(f"Filled categorical column '{col}' with mode: {fill_val}")
                else:
                    # All NaN or empty - convert to object and fill with "Unknown"
                    result[col] = result[col].astype("object").fillna("Unknown")
                    logger.debug(f"Filled empty categorical column '{col}' with 'Unknown'")
            elif categorical_strategy == "unknown":
                # Add "Unknown" as a category
                if "Unknown" not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories(["Unknown"])
                result[col] = result[col].fillna("Unknown")
                logger.debug(f"Filled categorical column '{col}' with 'Unknown' category")
            elif categorical_strategy == "add_value":
                # Add numeric_fill as a category
                if numeric_fill not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([numeric_fill])
                result[col] = result[col].fillna(numeric_fill)
                logger.debug(f"Filled categorical column '{col}' with new category: {numeric_fill}")

    # 3. Fill string/object columns
    object_cols = result.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        result[object_cols] = result[object_cols].fillna(string_fill)
        logger.debug(f"Filled {len(object_cols)} object columns with '{string_fill}'")

    # 4. Fill datetime columns
    datetime_cols = result.select_dtypes(include=["datetime64"]).columns
    for col in datetime_cols:
        if result[col].isna().any():
            if datetime_strategy == "forward_fill":
                result[col] = result[col].fillna(method="ffill")
                # If still NaN (first values), backfill
                result[col] = result[col].fillna(method="bfill")
                logger.debug(f"Forward-filled datetime column '{col}'")
            elif datetime_strategy == "now":
                result[col] = result[col].fillna(pd.Timestamp.now())
                logger.debug(f"Filled datetime column '{col}' with current timestamp")

    missing_after = result.isna().sum().sum()
    if missing_after > 0:
        logger.warning(
            f"After type-aware filling, {missing_after} missing values remain. "
            f"Consider running apply_enhanced_imputation_strategy_6step() first."
        )
    else:
        logger.info("All missing values filled successfully using type-aware strategy")

    return result


def apply_enhanced_imputation_strategy_4step(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
    price_column: str = "last_price",
) -> pd.DataFrame:
    """Backward compatibility wrapper for 6-step imputation.

    DEPRECATED: Use apply_enhanced_imputation_strategy_6step() instead.
    This wrapper calls the 6-step function with categorical and date handling enabled.

    Step 1: Zero imputation for exceptional event columns (48 columns)
    Step 2: Sector-aware KNN imputation for core financial metrics (148 columns)
    Step 3: Price imputation for semantic price columns (preserve original units)
    Step 4: Median imputation for all remaining numerical columns (price-protected)

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation

    Returns:
        DataFrame with complete imputation applied (zero missing values)

    Examples:
        >>> # Apply complete 6-step imputation pipeline
        >>> df_complete = apply_enhanced_imputation_strategy_4step(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5,
        ...     price_column='last_price'
        ... )
        >>> # Verify no missing values remain
        >>> assert df_complete.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    """
    logger.warning(
        "apply_enhanced_imputation_strategy_4step is deprecated. "
        "Use apply_enhanced_imputation_strategy_6step for full imputation coverage."
    )
    return apply_enhanced_imputation_strategy_6step(
        df,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
        price_column=price_column,
        handle_categoricals=True,
        handle_dates=True,
    )
