"""Data Sanitization stage for ETL."""

import logging
from typing import Set

import pandas as pd

from finance_ml.core.schema import list_non_recurring_cols
from finance_ml.ml_workflow.preprocessing.data import sanitize_dataframe_with_logging

logger = logging.getLogger(__name__)


def get_dividend_zero_fill_columns() -> Set[str]:
    """Return dividend-related columns that should be zero-filled when missing.
    
    Business Logic:
        Missing dividend data typically indicates the company does not pay dividends.
        Zero-filling prevents KNN/median imputation from generating distorting values
        that would incorrectly imply dividend payments.
    
    Returns:
        Set of dividend column names for zero-fill treatment
    """
    return {
        # Dividend Record Metadata
        "dividend_record_amount",
        "dividend_streak",
        
        # Dividend Payments
        "common_dividends_paid_ltm",
        "common_dividends_paid_fy",
        "dividend_per_share_ltm",
        
        # Dividend Yields (all periods)
        "div_yield_ind",
        "div_yield_ltm",
        "div_yield_1fyind",
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
    }


def get_dividend_na_fill_columns() -> Set[str]:
    """Return dividend categorical columns that should be filled with 'N/A' or 'None'.
    
    Business Logic:
        Categorical dividend fields should indicate absence of data rather than
        being imputed with mode values from dividend-paying companies.
    
    Returns:
        Set of categorical dividend column names for N/A fill treatment
    """
    return {
        "dividend_record_currency",
        "dividend_record_frequency",
    }


def get_dividend_date_columns() -> Set[str]:
    """Return dividend date columns that should NOT be imputed.
    
    Business Logic:
        Date columns for non-dividend-paying companies should remain NaT
        rather than being forward-filled with dates from dividend payers.
        These will be handled separately in datetime imputation.
    
    Returns:
        Set of dividend date column names to skip in sanitization
    """
    return {
        "dividend_record_announce_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
        "dividend_record_ex_date",
    }


def get_analyst_rating_zero_fill_columns() -> Set[str]:
    """Return analyst rating count columns that should be zero-filled when missing.
    
    Business Logic:
        Missing analyst ratings typically indicate no analyst coverage.
        Zero-filling prevents imputation from generating artificial coverage
        that would distort consensus analysis.
    
    Returns:
        Set of analyst rating column names for zero-fill treatment
    """
    return {
        "num_strong_sell_ratings",
        "num_strong_buys_ratings",
        "num_hold_ratings",
        "num_buys_ratings",
        "num_sell_ratings",
        "num_analyst_ratings",
        "price_target_count",
        "price_target_count_1w_ago",
        "price_target_count_1m_ago",
        "price_target_count_3m_ago",
        "price_target_count_6m_ago",
        "price_target_count_mtd_ago",
        "price_target_count_qtd_ago",
        "price_target_count_ytd_ago",
        "price_target_count_1y_ago",
    }


def get_income_statement_zero_fill_columns() -> Set[str]:
    """Return income statement and cash flow columns that should be zero-filled when missing.

    Business Logic:
        These are rare/exceptional non-recurring items or specific cash flow
        line items (like acquisitions) where missing values typically indicate
        the event did not occur. Zero is the economically correct imputation.

    IMPORTANT: Excludes recurring operational items (R&D, Interest) which
    now use median imputation to prevent financial picture distortion.

    Returns:
        Set of zero-fill candidate column names
    """
    cols = set(list_non_recurring_cols())
    # Add acquisition-related cash flow items (typically 0 if missing)
    cols.update(
        {
            "cash_acquisitions_ltm",
            "cash_acquisitions_fy",
            "cash_acquisitions_fq",
            "cash_acquisitions_1fy",
            "cash_acquisitions_1fqfq",
            "cash_acquisitions_2fqfq",
            "cash_acquisitions_3fqfq",
            "cash_acquisitions_4fqfq",
            "cash_acquisitions_5yavgfq",
        }
    )
    return cols


def get_balance_sheet_zero_fill_columns() -> Set[str]:
    """Return balance sheet columns that should be zero-filled when missing.

    Business Logic:
        Aligns with imputation strategy: Balance sheet items like Goodwill
        and Intangibles are now handled via median imputation in the main
        pipeline to avoid sector bias, rather than zero-filling here.
    """
    return set()


def apply_business_rule_zero_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Apply business-rule-based zero fills before general sanitization.
    
    This function applies domain-specific zero-filling for columns where
    missing values have business meaning (e.g., no dividends, no analyst coverage).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with business-rule zero fills applied
    """
    result = df.copy()
    
    # Collect all zero-fill columns
    zero_fill_columns = (
        get_dividend_zero_fill_columns() |
        get_analyst_rating_zero_fill_columns() |
        get_income_statement_zero_fill_columns() |
        get_balance_sheet_zero_fill_columns()
    )
    
    # Apply zero fills
    zero_filled_count = 0
    for col in zero_fill_columns:
        # Fills missing numeric values with zero if present
        if col in result.columns:
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                result[col] = result[col].fillna(0)
                zero_filled_count += missing_count
                logger.debug(f"Zero-filled {missing_count} missing values in '{col}'")
    
    # Apply N/A fills for categorical dividend columns
    na_fill_columns = get_dividend_na_fill_columns()
    na_filled_count = 0
    for col in na_fill_columns:
        if col in result.columns:
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                # Handle categorical dtype mapping per v1.19 guidelines
                fill_val = "N/A" if "currency" in col else "None"
                
                if isinstance(result[col].dtype, pd.CategoricalDtype):
                    if fill_val not in result[col].cat.categories:
                        result[col] = result[col].cat.add_categories([fill_val])
                result[col] = result[col].fillna(fill_val)
                na_filled_count += missing_count
                logger.debug(f"Categorical fill: {missing_count} values in '{col}' set to {fill_val}")

    logger.info(
        f"Stage 4 Sanitization: {zero_filled_count} zero-fills, "
        f"{na_filled_count} N/A-fills (aligned with v1.19 strategy)"
    )
    
    return result


def run_sanitization_stage(
    df: pd.DataFrame,
    apply_business_rules: bool = True,
) -> pd.DataFrame:
    """Stage 4: Sanitize data with business-rule-aware zero fills.
    
    This enhanced sanitization stage applies domain-specific zero-filling
    for dividend, analyst rating, and financial statement columns BEFORE
    general sanitization to prevent distorting imputation.
    
    Args:
        df: DataFrame to process
        apply_business_rules: Whether to apply business-rule zero fills (default: True)
        
    Returns:
        Sanitized DataFrame
    """
    logger.info("Stage 4: Sanitizing data")
    
    result = df.copy()
    
    # Step 1: Apply business-rule zero fills first
    if apply_business_rules:
        result = apply_business_rule_zero_fills(result)
    
    # Step 2: Apply general sanitization (inf, nan, extreme values)
    result = sanitize_dataframe_with_logging(result, apply_business_rules=False)
    
    return result
