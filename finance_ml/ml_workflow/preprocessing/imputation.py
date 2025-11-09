"""
Imputation module for finance_ml.ml_workflow.preprocessing.

This module provides a comprehensive 4-step imputation strategy for financial data:
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
    - apply_enhanced_imputation_strategy_4step: Complete 4-step pipeline

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
    """Return list of columns for zero imputation (Step 1 of 4-step strategy).

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


def get_knn_imputation_columns() -> List[str]:
    """Return list of columns for KNN imputation (Step 2 of 4-step strategy).

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
    """Apply price imputation (Step 3 of 4-step strategy).

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
    """Apply median imputation (Step 4 of 4-step strategy).

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


def apply_enhanced_imputation_strategy_4step(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
    price_column: str = "last_price",
) -> pd.DataFrame:
    """Apply complete 4-step imputation strategy from Phase 9.1.

    Step 1: Zero imputation for exceptional event columns (48 columns)
    Step 2: Sector-aware KNN imputation for core financial metrics (148 columns)
    Step 3: Price imputation for price target columns (5 columns)
    Step 4: Median imputation for all remaining numerical columns

    This ensures zero missing values in the output dataframe.

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation

    Returns:
        DataFrame with complete 4-step imputation applied (zero missing values)

    Examples:
        >>> # Apply complete 4-step imputation pipeline
        >>> df_complete = apply_enhanced_imputation_strategy_4step(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5,
        ...     price_column='last_price'
        ... )
        >>> # Verify no missing values remain
        >>> assert df_complete.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    """
    logger.info("Starting Phase 9.1 enhanced 4-step imputation strategy")

    # Track missing values at each step
    missing_initial = df.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"Initial missing values: {missing_initial}")

    # Step 1: Zero imputation for exceptional events
    logger.info("Step 1: Applying zero imputation for exceptional event columns (48 cols)")
    result = apply_zero_imputation(df)
    missing_after_step1 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 1: {missing_after_step1} missing values remain")

    # Step 2: KNN imputation for core financial metrics
    logger.info("Step 2: Applying sector-aware KNN imputation for financial metrics (148 cols)")
    result = apply_knn_imputation_enhanced(
        result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )
    missing_after_step2 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 2: {missing_after_step2} missing values remain")

    # Step 3: Price imputation for price targets
    logger.info("Step 3: Applying price imputation for price target columns (5 cols)")
    result = apply_price_imputation(result, price_column=price_column)
    missing_after_step3 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 3: {missing_after_step3} missing values remain")

    # Step 4: Median imputation for remaining columns
    logger.info("Step 4: Applying median imputation for remaining columns")
    result = apply_median_imputation(result)
    missing_final = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 4: {missing_final} missing values remain")

    # Log summary
    total_reduction = missing_initial - missing_final
    logger.info(
        f"4-step imputation complete: Reduced missing values from {missing_initial} "
        f"to {missing_final} (reduction: {total_reduction})"
    )

    if missing_final > 0:
        logger.warning(
            f"Warning: {missing_final} missing values still remain after 4-step imputation"
        )

    return result
