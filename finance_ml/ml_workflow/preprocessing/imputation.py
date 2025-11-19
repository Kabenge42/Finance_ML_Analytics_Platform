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

# Initialize logger
logger = logging.getLogger(__name__)


def get_zero_imputation_columns() -> List[str]:
    """Return list of columns for zero imputation (Step 1 of 6-step strategy).

    These columns represent rare/exceptional events (impairments, restructuring,
    acquisitions, etc.) where missing values typically mean the event did not occur.
    Zero is the economically correct imputation.

    Returns:
        List of 48 column names for zero imputation
    """
    return [
        # Impairment of Goodwill (5 columns)
        "impairment_of_goodwill_fq",
        "impairment_of_goodwill_ltm",
        "impairment_of_goodwill_1fy",
        "impairment_of_goodwill_fy",
        "impairment_of_goodwill_5yavgfq",
        # Asset writedown (5 columns)
        "asset_writedown_fq",
        "asset_writedown_ltm",
        "asset_writedown_fy",
        "asset_writedown_1fy",
        "asset_writedown_5yavgfq",
        # Merger & restructuring charges (5 columns)
        "merger_restructuring_charges_fq",
        "merger_restructuring_charges_fy",
        "merger_restructuring_charges_ltm",
        "merger_restructuring_charges_5yavgfq",
        "interest_expense_total_ltm",
        # Restructuring charges (5 columns)
        "restructuring_charges_ltm",
        "restructuring_charges_fq",
        "restructuring_charges_1fy",
        "restructuring_charges_fy",
        "restructuring_charges_5yavgfq",
        # Cash acquisitions (5 columns)
        "cash_acquisitions_fq",
        "cash_acquisitions_ltm",
        "cash_acquisitions_fy",
        "cash_acquisitions_1fy",
        "cash_acquisitions_5yavgfq",
        # Capital expenditure (5 columns)
        "capital_expenditure_ltm",
        "capital_expenditure_1fy",
        "capital_expenditure_fy",
        "capital_expenditure_fq",
        "capital_expenditure_5yavgfq",
        # R&D and Other (6 columns)
        "r_d_expenses_ltm",
        "other_unusual_items_total_ltm",
        "interest_income_on_investments_ltm",
        "volume_shrs",
        "short_int",
        "gain_loss_on_sale_of_assets_ltm",
        # Additional exceptional events (4 columns) - to reach 48 total
        "merger_restructuring_charges_1fy",
        "r_d_expenses_fy",
        "r_d_expenses_fq",
        "r_d_expenses_5yavgfq",
        # Goodwill (5 columns)
        "goodwill_fq",
        "goodwill_ltm",
        "goodwill_fy",
        "goodwill_1fy",
        "goodwill_5yavgfq",
        # Gross intangible assets (3 columns)
        "gross_intangible_assets_ltm",
        "gross_intangible_assets_fy",
        "gross_intangible_assets_5yavgfq",
    ]


def get_categorical_imputation_config() -> dict:
    """Return configuration for categorical column imputation strategies.

    Returns:
        Dictionary mapping column names to imputation strategies:
        - 'most_frequent': Use mode (most common value)
        - 'constant': Use a specific constant value
        - 'forward_fill': Use forward fill (for ordered data)

    Examples:
        >>> config = get_categorical_imputation_config()
        >>> config['style_class']  # Returns 'most_frequent'
    """
    return {
        # Classification categories - use most frequent (mode)
        "style_class": "most_frequent",
        "size_class": "most_frequent",
        "next_earnings_status": "most_frequent",
        "sector": "most_frequent",
        "industry": "most_frequent",
        "region": "most_frequent",
        "country": "most_frequent",
        "trading_country": "most_frequent",
        # Flags - use constant 'Unknown' or most frequent
        "flag": ("constant", "Unknown"),
        # Identifiers - use constant 'MISSING'
        "ticker": ("constant", "N/A"),
        "isin": ("constant", "N/A"),
        # Text descriptions - use constant
        "description": ("constant", "No description available"),
        "name": ("constant", "Unknown"),
    }


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

        if strategy == "most_frequent":
            # Use mode (most common value)
            mode_value = result[col].mode()
            if len(mode_value) > 0:
                result[col] = result[col].fillna(mode_value[0])
                logger.debug(f"Imputed {n_missing} values in '{col}' with mode: {mode_value[0]}")
            else:
                # Fallback to constant if no mode exists
                result[col] = result[col].fillna("Unknown")
                logger.debug(
                    f"Imputed {n_missing} values in '{col}' with 'Unknown' (no mode found)"
                )

        elif strategy == "constant":
            fill = fill_value if fill_value is not None else "Unknown"
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
        date_columns: List of date columns. If None, auto-detects from common patterns
        strategy: Imputation strategy - 'forward_fill', 'backward_fill', 'median', 'constant'
        reference_date: Optional reference date for 'constant' strategy

    Returns:
        DataFrame with properly formatted datetime columns and no NaN values

    Examples:
        >>> # Format and impute date columns
        >>> df = apply_datetime_imputation_and_formatting(
        ...     df,
        ...     date_columns=['last_updated', 'income_statement_report_date', 'next_earnings'],
        ...     strategy='forward_fill'
        ... )
    """
    result = df.copy()

    # Auto-detect date columns if not specified
    if date_columns is None:
        date_columns = []
        # Common date column patterns
        date_patterns = ["date", "updated", "earnings", "report"]
        for col in result.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                date_columns.append(col)

    if len(date_columns) == 0:
        logger.info("No date columns found for imputation")
        return result

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

    Returns:
        List of 148 column names for KNN imputation
    """
    return [
        # Market metrics (3 columns)
        "market_cap",
        "enterprise_value",
        "market_cap_country_r",
        # Analyst ratings (6 columns)
        "analyst_rating",
        "strong_sell_ratings",
        "strong_buys_ratings",
        "hold_ratings",
        "buys_ratings",
        "sell_ratings",
        # Returns (4 columns)
        "total_return_ytd",
        "total_return_5y",
        "total_return_10y",
        "tot_return_cagr_3y",
        # Valuation ratios (8 columns)
        "p_e_ntm",
        "p_e_ltm",
        "p_e_1fyltm",
        "p_e_5yavgltm",
        "p_b_ltm",
        "p_b_1fy",
        "p_b_5yavg",
        "p_tbv_ltm",
        # Altman Z-Score (3 columns)
        "altman_z_score_fy",
        "altman_z_score_fq",
        "altman_z_score_ltm",
        # Profitability ratios (11 columns)
        "roe_5yavg",
        "roe_ltm",
        "roe_1fy",
        "roe_fy",
        "roa_5yavg",
        "roa_ltm",
        "roa_1fy",
        "roa_fy",
        "roic_5yavg",
        "roic_ltm",
        "roic_1fy",
        # Margin ratios (12 columns)
        "operating_margin_ltm",
        "operating_margin_1fy",
        "operating_margin_fy",
        "operating_margin_5yavg",
        "pretax_margin_ltm",
        "pretax_margin_1fy",
        "pretax_margin_fy",
        "pretax_margin_5yavg",
        "net_profit_margin_ltm",
        "net_profit_margin_1fy",
        "net_profit_margin_fy",
        "net_profit_margin_5yavg",
        # Leverage ratios (8 columns)
        "net_debt_ebitda_ltm",
        "net_debt_ebitda_1fy",
        "net_debt_ebitda_5yavg",
        "total_debt_total_equity_ltm",
        "total_debt_total_equity_1fy",
        "total_debt_total_equity_fy",
        "total_debt_total_equity_5yavg",
        "long_term_debt_total_capital_ltm",
        # Liquidity ratios (6 columns)
        "current_ratio_ltm",
        "current_ratio_1fy",
        "current_ratio_fy",
        "current_ratio_5yavg",
        "quick_ratio_ltm",
        "quick_ratio_1fy",
        # Growth metrics (8 columns)
        "revenue_cagr_3y",
        "revenue_cagr_5y",
        "revenue_cagr_10y",
        "ebitda_cagr_3y",
        "ebitda_cagr_5y",
        "net_income_cagr_3y",
        "net_income_cagr_5y",
        "eps_basic_excl_extra_items_cagr_5y",
        # Dividend metrics (8 columns)
        "dividend_yield_ltm",
        "dividend_yield_1fy",
        "dividend_yield_5yavg",
        "dividend_per_share_ltm",
        "dividend_per_share_1fy",
        "dividend_per_share_fy",
        "dividend_payout_ratio_ltm",
        "dividend_payout_ratio_1fy",
        # EPS metrics (11 columns)
        "eps_normalized_ltm",
        "eps_normalized_1fy",
        "eps_normalized_fy",
        "eps_normalized_5yavg",
        "eps_diluted_ltm",
        "eps_diluted_1fy",
        "eps_diluted_fy",
        "eps_basic_excl_extra_items_ltm",
        "eps_basic_excl_extra_items_1fy",
        "eps_basic_excl_extra_items_fy",
        "eps_basic_incl_extra_items_ltm",
        # Cash flow metrics (12 columns)
        "free_cash_flow_ltm",
        "free_cash_flow_1fy",
        "free_cash_flow_fy",
        "free_cash_flow_5yavg",
        "free_cash_flow_per_share_ltm",
        "free_cash_flow_per_share_1fy",
        "operating_cash_flow_ltm",
        "operating_cash_flow_1fy",
        "operating_cash_flow_fy",
        "operating_cash_flow_5yavg",
        "cash_from_financing_activities_ltm",
        "cash_from_investing_activities_ltm",
        # Balance sheet metrics (10 columns)
        "total_assets_ltm",
        "total_assets_1fy",
        "total_assets_fy",
        "total_assets_5yavg",
        "tangible_book_value_per_share_ltm",
        "tangible_book_value_per_share_1fy",
        "book_value_per_share_ltm",
        "book_value_per_share_1fy",
        "book_value_per_share_fy",
        "book_value_per_share_5yavg",
        # Income statement metrics (10 columns)
        "revenue_ltm",
        "revenue_1fy",
        "revenue_fy",
        "revenue_5yavg",
        "ebitda_ltm",
        "ebitda_1fy",
        "ebitda_fy",
        "ebitda_5yavg",
        "net_income_ltm",
        "net_income_1fy",
        # Working capital metrics (8 columns)
        "accounts_receivable_ltm",
        "accounts_receivable_fy",
        "accounts_receivable_5yavg",
        "inventory_ltm",
        "inventory_fy",
        "working_capital_ltm",
        "working_capital_fy",
        "working_capital_5yavgfy",
        # Other metrics (4 columns)
        "buyback_yield_ltm",
        "avg_employees_ltm",
        "avg_employees_fy",
        "avg_employees_5yavgfy",
    ]


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

    if not columns:
        logger.warning("No numeric columns to impute")
        return result

    # FIX: Coerce all columns to numeric BEFORE sector loop
    # This ensures string-contaminated columns are cleaned globally
    non_numeric_global = []
    for col in columns:
        if result[col].dtype == "object":
            non_numeric_global.append(col)
            logger.debug(f"Pre-processing: Converting column '{col}' from object to numeric")
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if non_numeric_global:
        logger.info(
            f"Pre-processed {len(non_numeric_global)} object columns to numeric before imputation"
        )

    # Check if sector column exists
    if sector_column not in df.columns:
        logger.warning(
            f"Sector column '{sector_column}' not found, falling back to global KNN imputation"
        )
        # Fall back to global KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        result[columns] = imputer.fit_transform(df[columns])
        logger.info(f"Applied global KNN imputation (k={n_neighbors}) to {len(columns)} columns")
        return result

    # Perform sector-aware KNN imputation
    sectors = df[sector_column].dropna().unique()
    imputed_count = 0

    for sector in sectors:
        sector_mask = df[sector_column] == sector
        sector_data = df.loc[sector_mask, columns].copy()

        # Check if sector has enough samples for KNN
        n_samples = sector_data.shape[0]
        if n_samples < 2:
            logger.warning(
                f"Sector '{sector}' has only {n_samples} sample(s), skipping KNN imputation"
            )
            continue

        # FIX: Coerce all columns to numeric, converting strings to NaN
        # This prevents "could not convert string to float" errors in KNN imputation
        non_numeric_cols = []
        for col in columns:
            if sector_data[col].dtype == "object":
                non_numeric_cols.append(col)
                logger.debug(f"Sector '{sector}': Converting column '{col}' from object to numeric")
                sector_data[col] = pd.to_numeric(sector_data[col], errors="coerce")

        if non_numeric_cols:
            logger.info(
                f"Sector '{sector}': Coerced {len(non_numeric_cols)} object columns to numeric"
            )

        # Validate no object dtypes remain
        remaining_objects = sector_data.select_dtypes(include=["object"]).columns.tolist()
        if remaining_objects:
            logger.warning(
                f"Sector '{sector}': Skipping KNN due to non-numeric columns: {remaining_objects}"
            )
            continue

        # Adjust n_neighbors if sector has fewer samples
        k = min(n_neighbors, n_samples - 1)

        # Check if sector has any missing values
        if not sector_data.isna().any().any():
            continue

        # Apply KNN imputation to this sector
        imputer = KNNImputer(n_neighbors=k)
        try:
            sector_imputed = imputer.fit_transform(sector_data)
            # Convert back to DataFrame to preserve column alignment
            sector_imputed_df = pd.DataFrame(
                sector_imputed, index=sector_data.index, columns=sector_data.columns
            )
            result.loc[sector_mask, columns] = sector_imputed_df
            imputed_count += 1
        except Exception as e:
            logger.warning(f"KNN imputation failed for sector '{sector}': {e}. Skipping.")
            continue

    # Handle rows with missing sector values using global imputation
    missing_sector_mask = df[sector_column].isna()
    if missing_sector_mask.any():
        missing_sector_data = df.loc[missing_sector_mask, columns].copy()

        # FIX: Coerce all columns to numeric for global imputation
        non_numeric_cols_global = []
        for col in columns:
            if missing_sector_data[col].dtype == "object":
                non_numeric_cols_global.append(col)
                logger.debug(f"Global imputation: Converting column '{col}' from object to numeric")
                missing_sector_data[col] = pd.to_numeric(missing_sector_data[col], errors="coerce")

        if non_numeric_cols_global:
            logger.info(
                f"Global imputation: Coerced {len(non_numeric_cols_global)} object columns to numeric"
            )

        if missing_sector_data.isna().any().any():
            k = min(n_neighbors, missing_sector_data.shape[0] - 1)
            if k > 0:
                imputer = KNNImputer(n_neighbors=k)
                try:
                    missing_imputed = imputer.fit_transform(missing_sector_data)
                    # Convert back to DataFrame to preserve column alignment
                    missing_imputed_df = pd.DataFrame(
                        missing_imputed,
                        index=missing_sector_data.index,
                        columns=missing_sector_data.columns,
                    )
                    result.loc[missing_sector_mask, columns] = missing_imputed_df
                    logger.info(
                        f"Applied global KNN to {missing_sector_mask.sum()} rows with missing sector"
                    )
                except Exception as e:
                    logger.warning(f"KNN imputation failed for missing sectors: {e}")

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

    Args:
        df: Input DataFrame
        price_column: Column to use for imputation (default: "last_price")
        columns: Price target columns to impute (default: all 5 price target columns)

    Returns:
        DataFrame with price-imputed values

    Examples:
        >>> # Impute missing price targets from last_price
        >>> df_imputed = apply_price_imputation(df, price_column='last_price')
    """
    result = df.copy()

    if columns is None:
        columns = [
            "price_target",
            "price_target_low",
            "price_target_median",
            "price_target_high",
            "price_target_ytd_ago",
            "price_5d_ago",
            "price_1w_ago",
            "price_1m_ago",
            "price_3m_ago",
            "price_6m_ago",
            "price_1y_ago",
            "price_3y_ago",
            "price_5y_ago",
            "price_qtd_ago",
        ]

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

    logger.info(f"Applied price imputation to {len(available_cols)} columns using '{price_column}'")
    return result


def apply_median_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median imputation (Step 4 of 6-step strategy).

    Fallback imputation strategy that fills any remaining missing values
    in numerical columns with their median values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with median-imputed values for all remaining missing numerical data

    Examples:
        >>> # Fill remaining missing values with column medians
        >>> df_complete = apply_median_imputation(df)
    """
    result = df.copy()

    # Get all numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found in dataframe")
        return result

    total_imputed = 0

    for col in numeric_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)
            total_imputed += n_missing
            logger.debug(
                f"Median-imputed {n_missing} values in column '{col}' with {median_val:.4f}"
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
        critical_date_columns = ["last_updated", "income_statement_report_date", "next_earnings"]

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
            datetime_status[col] = {"is_datetime": False, "has_missing": True, "ready": False}

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


def apply_enhanced_imputation_strategy_6step(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
    price_column: str = "last_price",
    handle_categoricals: bool = True,
    handle_dates: bool = True,
    categorical_strategy: str = "most_frequent",
    date_strategy: str = "forward_fill",
) -> pd.DataFrame:
    """Apply complete 6-step imputation strategy from Phase 9.1 (ENHANCED).

    Step 1: Zero imputation for exceptional event columns (48 columns)
    Step 2: Sector-aware KNN imputation for core financial metrics (148 columns)
    Step 3: Price imputation for price target columns (5 columns)
    Step 4: Median imputation for all remaining numerical columns
    Step 5: Categorical imputation for string/object columns (NEW)
    Step 6: Datetime imputation and formatting for date columns (NEW)

    This ensures ZERO missing values in the output dataframe across ALL data types.

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation
        handle_categoricals: Whether to apply categorical imputation (Step 5)
        handle_dates: Whether to apply datetime imputation (Step 6)
        categorical_strategy: Strategy for categorical imputation ('most_frequent' or 'constant')
        date_strategy: Strategy for date imputation ('forward_fill', 'median', 'constant')

    Returns:
        DataFrame with complete 6-step imputation applied (zero missing values)

    Examples:
        >>> # Apply complete 6-step imputation pipeline
        >>> df_complete = apply_enhanced_imputation_strategy_6step(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5,
        ...     price_column='last_price',
        ...     handle_categoricals=True,
        ...     handle_dates=True
        ... )
        >>> # Verify no missing values remain in ANY column
        >>> assert df_complete.isna().sum().sum() == 0
    """
    logger.info("Starting Phase 9.1 enhanced 6-step imputation strategy")

    # Track missing values at each step
    missing_initial = df.isna().sum().sum()
    missing_numeric_initial = df.select_dtypes(include=[np.number]).isna().sum().sum()
    missing_categorical_initial = (
        df.select_dtypes(include=["object", "category"]).isna().sum().sum()
    )

    logger.info(
        f"Initial missing values: {missing_initial} total ({missing_numeric_initial} numeric, {missing_categorical_initial} categorical)"
    )

    # Step 1: Zero imputation for exceptional events
    logger.info("Step 1: Applying zero imputation for exceptional event columns (48 cols)")
    result = apply_zero_imputation(df)
    missing_after_step1 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 1: {missing_after_step1} numeric missing values remain")

    # Step 2: KNN imputation for core financial metrics
    logger.info("Step 2: Applying sector-aware KNN imputation for financial metrics (148 cols)")
    result = apply_knn_imputation_enhanced(
        result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )
    missing_after_step2 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 2: {missing_after_step2} numeric missing values remain")

    # Step 3: Price imputation for price targets
    logger.info("Step 3: Applying price imputation for price target columns (5 cols)")
    result = apply_price_imputation(result, price_column=price_column)
    missing_after_step3 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 3: {missing_after_step3} numeric missing values remain")

    # Step 4: Median imputation for remaining numerical columns
    logger.info("Step 4: Applying median imputation for remaining numerical columns")
    result = apply_median_imputation(result)
    missing_after_step4 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 4: {missing_after_step4} numeric missing values remain")

    # Step 5: Categorical imputation (NEW)
    if handle_categoricals:
        logger.info(f"Step 5: Applying {categorical_strategy} imputation for categorical columns")

        # Get configuration for specific columns
        cat_config = get_categorical_imputation_config()

        # Group columns by strategy
        most_frequent_cols = [
            col
            for col, strat in cat_config.items()
            if strat == "most_frequent" and col in result.columns
        ]
        constant_cols = [
            (col, strat[1])
            for col, strat in cat_config.items()
            if isinstance(strat, tuple) and strat[0] == "constant" and col in result.columns
        ]

        # Apply most_frequent strategy
        if most_frequent_cols:
            result = apply_categorical_imputation(
                result, columns=most_frequent_cols, strategy="most_frequent"
            )

        # Apply constant strategy for specific columns
        for col, fill_value in constant_cols:
            result = apply_categorical_imputation(
                result, columns=[col], strategy="constant", fill_value=fill_value
            )

        # Catch any remaining categorical columns
        remaining_cat_cols = result.select_dtypes(include=["object", "category"]).columns
        remaining_with_na = [col for col in remaining_cat_cols if result[col].isna().any()]
        if remaining_with_na:
            result = apply_categorical_imputation(
                result, columns=remaining_with_na, strategy=categorical_strategy
            )

        missing_after_step5 = (
            result.select_dtypes(include=["object", "category"]).isna().sum().sum()
        )
        logger.info(f"After Step 5: {missing_after_step5} categorical missing values remain")
    else:
        missing_after_step5 = (
            result.select_dtypes(include=["object", "category"]).isna().sum().sum()
        )

    # Step 6: Datetime imputation and formatting (NEW)
    if handle_dates:
        logger.info(
            f"Step 6: Applying {date_strategy} imputation and formatting for datetime columns"
        )

        # Specify critical date columns for temporal features
        critical_date_cols = ["last_updated", "income_statement_report_date", "next_earnings"]

        result = apply_datetime_imputation_and_formatting(
            result, date_columns=critical_date_cols, strategy=date_strategy
        )

        missing_after_step6 = result.isna().sum().sum()
        logger.info(f"After Step 6: {missing_after_step6} total missing values remain")
    else:
        missing_after_step6 = result.isna().sum().sum()

    # Final verification and reporting
    missing_final = result.isna().sum().sum()
    total_reduction = missing_initial - missing_final

    logger.info(
        f"6-step imputation complete: Reduced missing values from {missing_initial} "
        f"to {missing_final} (reduction: {total_reduction})"
    )

    if missing_final > 0:
        # Identify which columns still have missing values
        cols_with_missing = result.columns[result.isna().any()].tolist()
        logger.warning(
            f"WARNING: {missing_final} NaN values still present\n"
            f"  Affected columns ({len(cols_with_missing)}): {cols_with_missing[:11]}..."
        )

        # Emergency fallback: only fill if user requested that data type to be handled
        if handle_categoricals or handle_dates:
            logger.warning("  Applying emergency fallback imputation...")

            for col in cols_with_missing:
                if pd.api.types.is_numeric_dtype(result[col]):
                    # Always fill numeric (Steps 1-4 should have handled these)
                    result[col] = result[col].fillna(result[col].median())
                elif handle_categoricals and not pd.api.types.is_datetime64_any_dtype(result[col]):
                    # Fill categorical only if handle_categoricals=True
                    result[col] = result[col].fillna("Unknown")
                elif handle_dates and pd.api.types.is_datetime64_any_dtype(result[col]):
                    # Fill datetime only if handle_dates=True
                    result[col] = result[col].fillna(pd.Timestamp.now())

            missing_final = result.isna().sum().sum()
            logger.info(f"After emergency fallback: {missing_final} missing values remain")

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
    Step 3: Price imputation for price target columns (5 columns)
    Step 4: Median imputation for all remaining numerical columns

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
