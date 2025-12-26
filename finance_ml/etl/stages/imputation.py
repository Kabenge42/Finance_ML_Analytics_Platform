"""Imputation stage for ETL."""

import logging

import pandas as pd

from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy_6step

logger = logging.getLogger(__name__)

def run_imputation_stage(
    df: pd.DataFrame, 
    strategy: str = "6step",
    sector_column: str = "sector",
    reference_price_column: str = "last_price"
) -> pd.DataFrame:
    """Stage 5: Apply imputation strategy.
    
    Args:
        df: DataFrame to process
        strategy: Imputation strategy name
        sector_column: Column name for sector
        reference_price_column: Column name for reference price
        
    Returns:
        DataFrame with imputed values
    """
    logger.info(f"Stage 5: Applying imputation strategy: {strategy}")
    if strategy == "6step":
        # Signature: apply_enhanced_imputation_strategy_6step(df, sector_column, n_neighbors, price_column, ...)
        return apply_enhanced_imputation_strategy_6step(
            df, 
            sector_column=sector_column,
            price_column=reference_price_column
        )
    # Fallback to simple imputation if strategy not matched (could be extended)
    logger.warning(f"Strategy {strategy} not fully implemented in modular stage, using 6step fallback")
    return apply_enhanced_imputation_strategy_6step(df, sector_column=sector_column, price_column=reference_price_column)
