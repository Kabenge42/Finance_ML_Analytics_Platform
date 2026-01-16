"""Imputation stage for ETL.

This module provides role-based imputation strategies that leverage the semantic
roles defined in finance_ml.core.schema to apply appropriate imputation methods
for each column type.

Role-Based Imputation Strategy:
    - non_recurring: Zero fill (missing = event didn't occur)
    - count: Conditional zero/median (analyst counts → zero; others → median)
    - financial_statement/balance_sheet/cash_flow: Sector median
    - ratio/percentage: Bounded KNN with outlier clipping
    - market/feature: Sector-aware KNN
    - target: Fallback chain (price_target → median → last_price)
    - auxiliary: Global median
"""

import logging
from typing import Any, Dict, List, Set

import pandas as pd

from finance_ml.core.schema import COLUMN_SCHEMA, list_count_cols, normalize_column_name
from finance_ml.etl.stages.sanitization import (
    get_dividend_zero_fill_columns,
    get_analyst_rating_zero_fill_columns,
    get_income_statement_zero_fill_columns,
    get_balance_sheet_zero_fill_columns,
)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    get_zero_imputation_columns,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Role-Based Imputation Strategy Configuration
# =============================================================================

# Role-based imputation strategy configuration.
# Each role maps to a method and priority. Lower priority = processed first.
# Strategy priority: role-specific → dtype-specific → fallback.
ROLE_IMPUTATION_CONFIG: Dict[str, Dict[str, Any]] = {
    "non_recurring": {
        "method": "zero",
        "priority": 1,
        "rationale": "Missing = event did not occur (impairments, writedowns)",
    },
    "count": {
        "method": "conditional_zero_or_median",
        "priority": 2,
        "rationale": "Analyst counts → zero (no coverage); others → sector median",
    },
    "financial_statement": {
        "method": "sector_median",
        "priority": 3,
        "rationale": "Zero is valid but missing ≠ zero; use sector median",
    },
    "balance_sheet": {
        "method": "sector_median",
        "priority": 3,
        "rationale": "Balance sheet items can legitimately be zero",
    },
    "cash_flow": {
        "method": "sector_median",
        "priority": 3,
        "rationale": "Cash flows can be negative or zero legitimately",
    },
    "ratio": {
        "method": "knn_bounded",
        "priority": 4,
        "rationale": "Ratios have natural bounds; outlier-clip before imputation",
    },
    "percentage": {
        "method": "knn_bounded",
        "priority": 4,
        "rationale": "Percentages bounded but can exceed 100 for growth rates",
    },
    "market": {
        "method": "knn_sector",
        "priority": 4,
        "rationale": "Market data correlates strongly within sector/size",
    },
    "feature": {
        "method": "knn_sector",
        "priority": 5,
        "rationale": "Engineered features use sector-aware KNN",
    },
    "target": {
        "method": "fallback_chain",
        "priority": 0,
        "rationale": "Use median target, then last price as proxy",
    },
    "target_fallback": {
        "method": "knn_sector",
        "priority": 4,
        "rationale": "Secondary targets use sector-aware KNN",
    },
    "auxiliary": {
        "method": "global_median",
        "priority": 6,
        "rationale": "Legacy/optional columns use simple global median",
    },
}


# =============================================================================
# Role-Based Imputation Helper Functions
# =============================================================================


def get_columns_by_role(df: pd.DataFrame, role: str) -> List[str]:
    """Get DataFrame columns matching a specific schema role.

    Args:
        df: DataFrame to analyze
        role: Role name from schema (e.g., 'non_recurring', 'count')

    Returns:
        List of column names with the specified role
    """
    matched = []
    for col in df.columns:
        normalized = normalize_column_name(col)
        meta = COLUMN_SCHEMA.get(normalized, {})
        if meta.get("role") == role:
            matched.append(col)
    return matched


def _impute_count_columns(
    df: pd.DataFrame,
    columns: List[str],
    sector_column: str,
    log: dict,
) -> pd.DataFrame:
    """Impute count columns with conditional zero or sector median.

    Analyst rating counts get zero (no coverage), others get sector median.

    Args:
        df: DataFrame to impute
        columns: List of count columns with missing values
        sector_column: Column to use for sector grouping
        log: Dictionary to record imputation details

    Returns:
        DataFrame with count columns imputed
    """
    result = df.copy()
    zero_patterns = ["num_", "rating", "dividend_streak"]

    for col in columns:
        count = result[col].isna().sum()
        col_lower = col.lower()

        # Check if this is a zero-fill pattern
        is_zero_fill = any(p in col_lower for p in zero_patterns)

        if is_zero_fill:
            result[col] = result[col].fillna(0)
            log[col] = {
                "method": "zero",
                "count": count,
                "reason": "analyst_count_pattern",
            }
        else:
            # Sector median for other counts
            if sector_column in result.columns:
                sector_medians = result.groupby(sector_column)[col].transform("median")
                result[col] = result[col].fillna(sector_medians)
            remaining = result[col].isna().sum()
            if remaining > 0:
                global_med = result[col].median()
                result[col] = result[col].fillna(
                    global_med if pd.notna(global_med) else 0
                )
            log[col] = {"method": "sector_median", "count": count}

    return result


def _impute_sector_median(
    df: pd.DataFrame,
    columns: List[str],
    sector_column: str,
    log: dict,
) -> pd.DataFrame:
    """Impute using sector median with global fallback.

    Args:
        df: DataFrame to impute
        columns: List of columns with missing values
        sector_column: Column to use for sector grouping
        log: Dictionary to record imputation details

    Returns:
        DataFrame with sector median imputation applied
    """
    result = df.copy()

    for col in columns:
        count = result[col].isna().sum()

        # Step 1: Sector median
        if sector_column in result.columns:
            sector_medians = result.groupby(sector_column)[col].transform("median")
            result[col] = result[col].fillna(sector_medians)

        # Step 2: Global median for remaining
        remaining = result[col].isna().sum()
        if remaining > 0:
            global_med = result[col].median()
            result[col] = result[col].fillna(global_med if pd.notna(global_med) else 0)

        log[col] = {"method": "sector_median", "count": count}

    return result


def _impute_knn_bounded(
    df: pd.DataFrame,
    columns: List[str],
    sector_column: str,
    log: dict,
    clip_lower: float = 1,
    clip_upper: float = 99,
) -> pd.DataFrame:
    """Impute ratios/percentages with outlier clipping and sector median.

    Clips outliers to specified percentiles before applying sector median
    imputation. This prevents extreme values from distorting imputation.

    Args:
        df: DataFrame to impute
        columns: List of ratio/percentage columns with missing values
        sector_column: Column to use for sector grouping
        log: Dictionary to record imputation details
        clip_lower: Lower percentile for clipping (default: 1)
        clip_upper: Upper percentile for clipping (default: 99)

    Returns:
        DataFrame with bounded imputation applied
    """
    result = df.copy()

    for col in columns:
        count = result[col].isna().sum()

        # Clip outliers before imputation
        lower = result[col].quantile(clip_lower / 100)
        upper = result[col].quantile(clip_upper / 100)
        if pd.notna(lower) and pd.notna(upper):
            result[col] = result[col].clip(lower, upper)

        # Sector median (simplified - could use sklearn KNNImputer)
        if sector_column in result.columns:
            sector_medians = result.groupby(sector_column)[col].transform("median")
            result[col] = result[col].fillna(sector_medians)

        remaining = result[col].isna().sum()
        if remaining > 0:
            global_med = result[col].median()
            result[col] = result[col].fillna(global_med if pd.notna(global_med) else 0)

        log[col] = {"method": "knn_bounded", "count": count, "bounds": (lower, upper)}

    return result


def _impute_knn_sector(
    df: pd.DataFrame,
    columns: List[str],
    sector_column: str,
    log: dict,
) -> pd.DataFrame:
    """Impute market/feature columns using sector-aware strategy.

    Uses sector median as a proxy for KNN imputation, with global
    median fallback for sectors with insufficient data.

    Args:
        df: DataFrame to impute
        columns: List of market/feature columns with missing values
        sector_column: Column to use for sector grouping
        log: Dictionary to record imputation details

    Returns:
        DataFrame with sector-aware imputation applied
    """
    result = df.copy()

    for col in columns:
        count = result[col].isna().sum()

        # Sector median as proxy for KNN
        if sector_column in result.columns:
            sector_medians = result.groupby(sector_column)[col].transform("median")
            result[col] = result[col].fillna(sector_medians)

        remaining = result[col].isna().sum()
        if remaining > 0:
            global_med = result[col].median()
            result[col] = result[col].fillna(global_med if pd.notna(global_med) else 0)

        log[col] = {"method": "knn_sector", "count": count}

    return result


def _impute_target_fallback(
    df: pd.DataFrame,
    columns: List[str],
    log: dict,
) -> pd.DataFrame:
    """Impute target columns using fallback chain.

    Uses price_target_median first, then last_price as fallback
    for missing target values.

    Args:
        df: DataFrame to impute
        columns: List of target columns with missing values
        log: Dictionary to record imputation details

    Returns:
        DataFrame with target fallback imputation applied
    """
    result = df.copy()
    fallback_chain = ["price_target_median", "last_price"]

    for col in columns:
        count = result[col].isna().sum()

        for fallback in fallback_chain:
            if fallback in result.columns:
                result[col] = result[col].fillna(result[fallback])

        log[col] = {
            "method": "fallback_chain",
            "count": count,
            "fallbacks": fallback_chain,
        }

    return result


def apply_role_based_imputation(
    df: pd.DataFrame,
    sector_column: str = "sector",
) -> pd.DataFrame:
    """Apply imputation strategies based on column roles from schema.

    This method processes columns in priority order:
    1. Target columns → Fallback chain (price_target → median → last_price)
    2. Non-recurring items → Zero fill
    3. Count columns → Conditional zero/median
    4. Financial statements → Sector median
    5. Ratios/percentages → Bounded KNN
    6. Market/features → Sector KNN
    7. Auxiliary → Global median

    Args:
        df: DataFrame to impute
        sector_column: Column to use for sector grouping

    Returns:
        DataFrame with role-appropriate imputation applied
    """
    result = df.copy()
    imputation_log: Dict[str, Dict[str, Any]] = {}

    # Sort roles by priority
    sorted_roles = sorted(
        ROLE_IMPUTATION_CONFIG.items(), key=lambda x: x[1]["priority"]
    )

    for role, config in sorted_roles:
        role_cols = get_columns_by_role(result, role)
        if not role_cols:
            continue

        method = config["method"]
        cols_with_missing = [c for c in role_cols if result[c].isna().any()]

        if not cols_with_missing:
            continue

        logger.info(
            f"Imputing {len(cols_with_missing)} '{role}' columns with '{method}'"
        )

        if method == "zero":
            for col in cols_with_missing:
                count = result[col].isna().sum()
                result[col] = result[col].fillna(0)
                imputation_log[col] = {"method": "zero", "count": count}

        elif method == "conditional_zero_or_median":
            result = _impute_count_columns(
                result, cols_with_missing, sector_column, imputation_log
            )

        elif method == "sector_median":
            result = _impute_sector_median(
                result, cols_with_missing, sector_column, imputation_log
            )

        elif method == "knn_bounded":
            result = _impute_knn_bounded(
                result, cols_with_missing, sector_column, imputation_log
            )

        elif method == "knn_sector":
            result = _impute_knn_sector(
                result, cols_with_missing, sector_column, imputation_log
            )

        elif method == "global_median":
            for col in cols_with_missing:
                count = result[col].isna().sum()
                median_val = result[col].median()
                result[col] = result[col].fillna(
                    median_val if pd.notna(median_val) else 0
                )
                imputation_log[col] = {
                    "method": "global_median",
                    "count": count,
                    "value": median_val,
                }

        elif method == "fallback_chain":
            result = _impute_target_fallback(result, cols_with_missing, imputation_log)

    # Log summary
    total_imputed = sum(v.get("count", 0) for v in imputation_log.values())
    logger.info(
        f"Role-based imputation complete: {total_imputed} values across "
        f"{len(imputation_log)} columns"
    )

    return result


# =============================================================================
# Existing Helper Functions
# =============================================================================


def get_schema_aligned_columns_by_role(
    df: pd.DataFrame,
    roles: List[str],
    dtypes: List[str] = None,
) -> List[str]:
    """Get columns from DataFrame that match specified schema roles and dtypes.

    Args:
        df: DataFrame to analyze
        roles: List of role names to include (e.g., ['feature', 'market', 'ratio'])
        dtypes: Optional list of dtype strings to filter by (e.g., ['float', 'Float64'])

    Returns:
        List of column names matching the criteria
    """
    matched_cols = []

    for col in df.columns:
        normalized = normalize_column_name(col)
        meta = COLUMN_SCHEMA.get(normalized, {})
        role = meta.get("role")

        if role in roles:
            if dtypes is None:
                matched_cols.append(col)
            else:
                col_dtype = meta.get("dtype", "")
                if any(dt in col_dtype for dt in dtypes):
                    matched_cols.append(col)

    return matched_cols


def get_pre_imputation_zero_fill_columns() -> Set[str]:
    """Return columns that should be zero-filled BEFORE imputation.

    These columns have business meaning when missing that would be
    distorted by KNN or median imputation.

    Returns:
        Set of column names for pre-imputation zero fill
    """
    # Start with the non-recurring exceptional items
    zero_cols = set(get_zero_imputation_columns())

    # Add all business-rule zero-fill columns from sanitization
    zero_cols.update(get_dividend_zero_fill_columns())
    zero_cols.update(get_analyst_rating_zero_fill_columns())
    zero_cols.update(get_income_statement_zero_fill_columns())
    zero_cols.update(get_balance_sheet_zero_fill_columns())

    # Add count-role columns from schema for consistent handling
    count_columns = set(list_count_cols())
    # Filter to analyst rating counts only (zero means no coverage)
    analyst_count_cols = {c for c in count_columns if "num_" in c and "rating" in c}
    zero_cols.update(analyst_count_cols)

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
    use_schema_aligned_selection: bool = True,
    use_role_based_strategy: bool = True,
) -> pd.DataFrame:
    """Stage 5: Apply imputation strategy with business-rule pre-fills.

    This enhanced imputation stage applies business-rule zero/NA fills
    for dividend and analyst rating columns BEFORE the main strategy,
    then optionally applies role-based imputation for semantically
    appropriate handling of each column type.

    Role-Based Imputation Priority:
        1. Target columns → Fallback chain (price_target → median → last_price)
        2. Non-recurring items → Zero fill (missing = event didn't occur)
        3. Count columns → Conditional zero/median
        4. Financial statements → Sector median
        5. Ratios/percentages → Bounded KNN with outlier clipping
        6. Market/features → Sector-aware KNN
        7. Auxiliary → Global median

    Args:
        df: DataFrame to process
        strategy: Imputation strategy name ('6step' or 'role_based')
        sector_column: Column name for sector grouping
        reference_price_column: Column name for reference price
        apply_pre_imputation_fills: Apply business-rule fills first (default: True)
        use_schema_aligned_selection: Use schema roles for column selection (default: True)
        use_role_based_strategy: Apply role-based imputation before main strategy (default: True)

    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Stage 5: Applying imputation strategy: {strategy}")

    result = df.copy()

    # Step 1: Apply business-rule pre-fills
    if apply_pre_imputation_fills:
        result = apply_pre_imputation_business_fills(result)

    # Step 2: Apply role-based imputation if enabled
    if use_role_based_strategy:
        result = apply_role_based_imputation(result, sector_column=sector_column)
        logger.info("Role-based imputation complete")

    # Get schema-aligned columns for KNN imputation (informational logging)
    if use_schema_aligned_selection:
        knn_cols = get_schema_aligned_columns_by_role(
            result,
            roles=["feature", "market", "ratio", "percentage", "financial_statement"],
            dtypes=["float", "Float64"],
        )
        logger.info(f"Schema-aligned KNN imputation: {len(knn_cols)} columns selected")

    # Step 3: Apply main imputation strategy for remaining gaps
    if strategy == "6step":
        result = apply_enhanced_imputation_strategy_6step(
            result,
            sector_column=sector_column,
            price_column=reference_price_column,
        )
    elif strategy == "role_based":
        # Role-based strategy already applied above, skip 6-step
        logger.info("Using role-based strategy only (6-step skipped)")
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
