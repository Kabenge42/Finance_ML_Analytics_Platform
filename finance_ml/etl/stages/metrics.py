"""Financial Metrics stages for ETL."""

import logging
from typing import Tuple, Dict

import pandas as pd

from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
    compute_valuation_metrics,
    compute_profitability_metrics,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_target_vs_price_metrics,
    handle_sector_specific_metrics,
    compute_sector_specific_ratios,
    impute_computed_metrics,
    CONDITIONAL_METRICS,
)

logger = logging.getLogger(__name__)

def run_financial_metrics_stage(
    df: pd.DataFrame,
    compute_valuation: bool = True,
    compute_profitability: bool = True,
    compute_growth: bool = True,
    compute_leverage: bool = True,
    compute_target_vs_price: bool = True,
    compute_sector_specific: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Stage 8: Compute financial metrics."""
    result = df.copy()
    stats = {}
    
    initial_cols = set(result.columns)
    
    if compute_valuation:
        logger.info("Stage 8a: Computing valuation metrics")
        result = compute_valuation_metrics(result)
        new_cols = set(result.columns) - initial_cols
        stats["valuation_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        
    if compute_profitability:
        logger.info("Stage 8b: Computing profitability metrics")
        result = compute_profitability_metrics(result)
        new_cols = set(result.columns) - initial_cols
        stats["profitability_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        
    if compute_growth:
        logger.info("Stage 8c: Computing growth metrics")
        result = compute_growth_metrics(result)
        new_cols = set(result.columns) - initial_cols
        stats["growth_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        
    if compute_leverage:
        logger.info("Stage 8d: Computing leverage metrics")
        result = compute_leverage_metrics(result)
        new_cols = set(result.columns) - initial_cols
        stats["leverage_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        
    if compute_target_vs_price:
        logger.info("Stage 8e: Computing target vs price metrics")
        result = compute_target_vs_price_metrics(result)
        new_cols = set(result.columns) - initial_cols
        stats["target_vs_price_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        
    if compute_sector_specific:
        logger.info("Stage 8f: Handling sector-specific metrics")
        result = handle_sector_specific_metrics(result)
        result = compute_sector_specific_ratios(result)
        new_cols = set(result.columns) - initial_cols
        stats["sector_specific_metrics_added"] = len(new_cols)
        
    return result, stats

def run_post_metrics_imputation_stage(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Stage 8g: Post-metrics imputation."""
    result = df.copy()
    missing_before = int(result.isna().sum().sum())
    
    if missing_before > 0:
        logger.info(f"Stage 8g: Post-metrics imputation (missing={missing_before})")
        try:
            result, _ = impute_computed_metrics(result, method="sector_median", sector_column="sector")
            
            for col in [c for c in CONDITIONAL_METRICS if c in result.columns]:
                result[f"{col}_applicable"] = result[col].notna()
                result[col] = result[col].fillna(0)
        except Exception as e:
            logger.warning(f"Post-metrics imputation step failed: {e}")
            
    missing_after = int(result.isna().sum().sum())
    return result, missing_after
