"""Imputation stage for ETL."""

import logging
from typing import Set

import pandas as pd

from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    get_zero_imputation_columns,
)

logger = logging.getLogger(__name__)


def get_pre_imputation_zero_fill_columns() -> Set[str]:
    """Return columns that should be zero-filled BEFORE imputation.
    
    These columns have business meaning when missing that would be
    distorted by KNN or median imputation.
    
    Returns:
        Set of column names for pre-imputation zero fill
    """
    # Start with the non-recurring exceptional items
    zero_cols = set(get_zero_imputation_columns())
    
    # Add dividend numeric columns (missing = no dividend)
    zero_cols.update({
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
    })
    
    # Add analyst rating counts (missing = no coverage)
    zero_cols.update({
        "num_strong_sell_ratings",
        "num_strong_buys_ratings",
        "num_hold_ratings",
        "num_buys_ratings",
        "num_sell_ratings",
    })
    
    return zero_cols


def get_pre_imputation_na_fill_columns() -> dict:
    """Return categorical columns with their N/A fill values.
    
    Returns:
        Dict mapping column names to fill values
    """
    return {
        "dividend_record_currency": "N/A",
        "dividend_record_frequency": "None",
    }


def apply_pre_imputation_business_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Apply business-rule fills before main imputation strategy.
    
    This ensures dividend, analyst rating, and other business-specific
    columns are zero-filled to prevent distortion from statistical imputation.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with pre-imputation business fills applied
    """
    result = df.copy()
    
    # Zero-fill numeric columns with business meaning
    zero_cols = get_pre_imputation_zero_fill_columns()
    zero_filled = 0
    for col in zero_cols:
        if col in result.columns and result[col].isna().any():
            count = result[col].isna().sum()
            result[col] = result[col].fillna(0)
            zero_filled += count
            logger.debug(f"Pre-imputation: Zero-filled {count} values in '{col}'")
    
    # N/A-fill categorical columns
    na_fill_map = get_pre_imputation_na_fill_columns()
    na_filled = 0
    for col, fill_val in na_fill_map.items():
        if col in result.columns and result[col].isna().any():
            count = result[col].isna().sum()
            # Handle categorical dtype
            if isinstance(result[col].dtype, pd.CategoricalDtype):
                if fill_val not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([fill_val])
            result[col] = result[col].fillna(fill_val)
            na_filled += count
            logger.debug(f"Pre-imputation: N/A-filled {count} values in '{col}'")
    
    if zero_filled > 0 or na_filled > 0:
        logger.info(
            f"Pre-imputation business fills: {zero_filled} zero-fills, "
            f"{na_filled} categorical fills"
        )
    
    return result


def run_imputation_stage(
    df: pd.DataFrame,
    strategy: str = "6step",
    sector_column: str = "sector",
    reference_price_column: str = "last_price",
    apply_pre_imputation_fills: bool = True,
) -> pd.DataFrame:
    """Stage 5: Apply imputation strategy with business-rule pre-fills.
    
    This enhanced imputation stage applies business-rule zero/NA fills
    for dividend and analyst rating columns BEFORE the 6-step strategy
    to prevent distorting imputation.
    
    Args:
        df: DataFrame to process
        strategy: Imputation strategy name
        sector_column: Column name for sector
        reference_price_column: Column name for reference price
        apply_pre_imputation_fills: Apply business-rule fills first (default: True)
        
    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Stage 5: Applying imputation strategy: {strategy}")
    
    result = df.copy()
    
    # Step 1: Apply business-rule pre-fills
    if apply_pre_imputation_fills:
        result = apply_pre_imputation_business_fills(result)
    
    # Step 2: Apply main imputation strategy
    if strategy == "6step":
        result = apply_enhanced_imputation_strategy_6step(
            result,
            sector_column=sector_column,
            price_column=reference_price_column,
        )
    else:
        logger.warning(
            f"Strategy {strategy} not fully implemented, using 6step fallback"
        )
        result = apply_enhanced_imputation_strategy_6step(
            result,
            sector_column=sector_column,
            price_column=reference_price_column,
        )
    
    return result
