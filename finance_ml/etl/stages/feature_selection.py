"""Feature selection stage for ETL."""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from finance_ml.ml_workflow.features.selection import select_features_auto

logger = logging.getLogger(__name__)

# Default identifier columns to preserve (ml_workflow_guidelines.md Section 8.2)
DEFAULT_IDENTIFIER_COLUMNS = [
    "ticker",
    "name",
    "isin",
    "sector",
    "region",
    "country",
    "trading_country",
    "exchange",
    "unit",
    "industry",
    "next_earnings",
    "income_statement_report_date",
    "fy_end_date",
    "next_fy_end_date",
    "next_fiscal_quarter",
]


def run_feature_selection_stage(
    df: pd.DataFrame,
    method: str = "both",
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    preserve_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Stage 10: Automated feature selection with identifier preservation.
    
    This stage applies automated feature selection while preserving critical
    identifier columns (ticker, isin, sector, region, country, industry) as
    specified in ml_workflow_guidelines.md Section 8.2.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and target column
    method : str, default="both"
        Selection method: 'mutual_info', 'correlation', or 'both'
    importance_threshold : float, default=0.01
        Minimum importance score to retain feature
    correlation_threshold : float, default=0.95
        Maximum correlation threshold for deduplication
    preserve_columns : list, optional
        Columns to always preserve. If None, uses DEFAULT_IDENTIFIER_COLUMNS.
        
    Returns
    -------
    Tuple[pd.DataFrame, int, int, int]
        (result_df, features_before, features_after, features_removed)
    """
    logger.info("Stage 10: Applying automated feature selection")
    features_before = len(df.columns)
    
    # Use default identifier columns if preserve_columns not specified
    if preserve_columns is None:
        preserve_columns = DEFAULT_IDENTIFIER_COLUMNS.copy()
    
    # Identify which preserve columns actually exist in the DataFrame
    available_preserve_cols = [col for col in preserve_columns if col in df.columns]
    missing_preserve_cols = [col for col in preserve_columns if col not in df.columns]
    
    if missing_preserve_cols:
        logger.warning(f"Preserve columns not found in DataFrame: {missing_preserve_cols}")
    if available_preserve_cols:
        logger.info(f"Preserving identifier columns: {available_preserve_cols}")
    
    # Need target column for feature selection
    target_col = None
    for col in ["price_target", "price_target_median"]:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        logger.warning("Feature selection skipped: no target column found")
        return df, features_before, features_before, 0
        
    try:
        y = df[target_col]
        
        # Exclude target and identifier columns from feature selection input
        # but preserve identifier columns in the output
        exclude_from_X = [target_col] + available_preserve_cols
        X = df.drop(columns=exclude_from_X, errors='ignore')
        
        X_selected = select_features_auto(
            X,
            y,
            importance_threshold=importance_threshold,
            correlation_threshold=correlation_threshold,
            method=method,
            preserve_columns=None,  # Don't use internal preserve, we handle it here
        )
        
        # Build result with preserved columns + selected features + target
        result = pd.DataFrame(index=df.index)
        
        # Add preserved identifier columns first
        for col in available_preserve_cols:
            result[col] = df[col]
        
        # Add selected features
        for col in X_selected.columns:
            result[col] = X_selected[col]
        
        # Add target column
        result[target_col] = y
        
        # Calculate feature counts (excluding identifiers and target)
        features_after = len(X_selected.columns)
        features_removed = len(X.columns) - features_after
        
        logger.info(
            f"Feature selection complete: {len(X.columns)} -> {features_after} features "
            f"({features_removed} removed), {len(available_preserve_cols)} identifiers preserved"
        )
        
        return result, features_before, len(result.columns), features_removed
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        return df, features_before, features_before, 0
