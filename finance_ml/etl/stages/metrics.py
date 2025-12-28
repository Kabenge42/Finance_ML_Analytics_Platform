"""Financial Metrics stages for ETL."""

import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

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

__all__ = [
    "run_financial_metrics_stage",
    "run_post_metrics_imputation_stage",
    "compute_valuation_metrics",
    "compute_profitability_metrics",
    "compute_growth_metrics",
    "compute_leverage_metrics",
    "compute_target_vs_price_metrics",
    "generate_metrics_dashboard",  # Added to exports
]

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

def generate_metrics_dashboard(
    df: pd.DataFrame,
    sector_column: str = "sector",
    region_column: Optional[str] = "region",
) -> Dict[str, Any]:
    """
    Generate a comprehensive metrics dashboard JSON structure.

    Dashboard includes:
    - Timestamp
    - Total stock count
    - Overall metrics (valuation, profitability, growth, leverage)
    - By-sector breakdown with statistics
    - By-region breakdown (if region_column provided)

    Args:
        df: DataFrame with computed financial metrics
        sector_column: Column name for sector grouping (default: 'sector')
        region_column: Column name for region grouping (default: 'region')

    Returns:
        Dictionary suitable for JSON serialization with structure:
        {
            'timestamp': ISO timestamp,
            'total_stocks': int,
            'by_sector': {category: {metric: stats}},
            'by_region': {category: {metric: stats}},
            'by_group': {sector_name: {category: {metric: stats}}}
        }
    """
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks": len(df),
        "by_sector": {},
        "by_region": {},
        "by_group": {},
    }

    def compute_stats(series: pd.Series) -> Dict[str, Any]:
        """Compute summary statistics for a numeric series."""
        numeric = pd.to_numeric(series, errors="coerce")
        return {
            "mean": float(numeric.mean()) if not numeric.isna().all() else None,
            "median": float(numeric.median()) if not numeric.isna().all() else None,
            "std": float(numeric.std()) if not numeric.isna().all() else None,
            "min": float(numeric.min()) if not numeric.isna().all() else None,
            "max": float(numeric.max()) if not numeric.isna().all() else None,
            "count": int(numeric.notna().sum()),
        }

    # =========================================================================
    # Overall Metrics (across all stocks)
    # =========================================================================
    
    # Valuation metrics
    valuation_metrics = {}
    for metric_col, metric_name in [
        ("p_e_ratio", "p_e"),
        ("p_b_ratio", "p_b"),
        ("p_s_ratio", "p_s"),
        ("ev_ebitda_ratio", "ev_ebitda"),
        ("ev_sales_ratio", "ev_sales"),
    ]:
        if metric_col in df.columns:
            valuation_metrics[metric_name] = compute_stats(df[metric_col])

    dashboard["by_sector"]["valuation"] = valuation_metrics

    # Profitability metrics
    profitability_metrics = {}
    for metric_col, metric_name in [
        ("gross_margin_pct", "gross_margin"),
        ("operating_margin_pct", "operating_margin"),
        ("net_margin_pct", "net_margin"),
        ("roe", "roe"),
        ("roa", "roa"),
    ]:
        if metric_col in df.columns:
            profitability_metrics[metric_name] = compute_stats(df[metric_col])

    dashboard["by_sector"]["profitability"] = profitability_metrics

    # Growth metrics
    growth_metrics = {}
    for metric_col, metric_name in [
        ("revenue_growth", "revenue_growth"),
        ("ebitda_growth", "ebitda_growth"),
        ("earnings_growth", "earnings_growth"),
    ]:
        if metric_col in df.columns:
            growth_metrics[metric_name] = compute_stats(df[metric_col])

    dashboard["by_sector"]["growth"] = growth_metrics

    # Leverage metrics
    leverage_metrics = {}
    for metric_col, metric_name in [
        ("debt_to_equity", "debt_to_equity"),
        ("debt_to_assets", "debt_to_assets"),
    ]:
        if metric_col in df.columns:
            leverage_metrics[metric_name] = compute_stats(df[metric_col])

    dashboard["by_sector"]["leverage"] = leverage_metrics

    # =========================================================================
    # Per-Sector Breakdown
    # =========================================================================
    if sector_column in df.columns:
        by_group = {}
        for sector in df[sector_column].dropna().unique():
            sector_df = df[df[sector_column] == sector]
            sector_stats = {
                "count": len(sector_df),
                "valuation": {},
                "profitability": {},
                "growth": {},
                "leverage": {},
            }

            # Valuation by sector
            for metric_col, metric_name in [
                ("p_e_ratio", "p_e"),
                ("p_s_ratio", "p_s"),
                ("ev_ebitda_ratio", "ev_ebitda"),
            ]:
                if metric_col in sector_df.columns:
                    sector_stats["valuation"][metric_name] = compute_stats(sector_df[metric_col])

            # Profitability by sector
            for metric_col, metric_name in [
                ("roe", "roe"),
                ("roa", "roa"),
                ("gross_margin_pct", "gross_margin"),
                ("operating_margin_pct", "operating_margin"),
            ]:
                if metric_col in sector_df.columns:
                    sector_stats["profitability"][metric_name] = compute_stats(sector_df[metric_col])

            # Growth by sector
            for metric_col, metric_name in [
                ("revenue_growth", "revenue_growth"),
                ("ebitda_growth", "ebitda_growth"),
                ("earnings_growth", "earnings_growth"),
            ]:
                if metric_col in sector_df.columns:
                    sector_stats["growth"][metric_name] = compute_stats(sector_df[metric_col])

            # Leverage by sector
            for metric_col, metric_name in [
                ("debt_to_equity", "debt_to_equity"),
                ("debt_to_assets", "debt_to_assets"),
            ]:
                if metric_col in sector_df.columns:
                    sector_stats["leverage"][metric_name] = compute_stats(sector_df[metric_col])

            by_group[str(sector)] = sector_stats

        dashboard["by_group"] = by_group

    # =========================================================================
    # Per-Region Breakdown (if region column provided)
    # =========================================================================
    if region_column and region_column in df.columns:
        by_region_group = {}
        for region in df[region_column].dropna().unique():
            region_df = df[df[region_column] == region]
            region_stats = {
                "count": len(region_df),
                "valuation": {},
                "profitability": {},
                "growth": {},
                "leverage": {},
            }

            # Valuation by region
            for metric_col, metric_name in [
                ("p_e_ratio", "p_e"),
                ("p_s_ratio", "p_s"),
                ("ev_ebitda_ratio", "ev_ebitda"),
            ]:
                if metric_col in region_df.columns:
                    region_stats["valuation"][metric_name] = compute_stats(region_df[metric_col])

            # Profitability by region
            for metric_col, metric_name in [
                ("roe", "roe"),
                ("roa", "roa"),
                ("gross_margin_pct", "gross_margin"),
            ]:
                if metric_col in region_df.columns:
                    region_stats["profitability"][metric_name] = compute_stats(region_df[metric_col])

            # Growth by region
            for metric_col, metric_name in [
                ("revenue_growth", "revenue_growth"),
                ("ebitda_growth", "ebitda_growth"),
                ("earnings_growth", "earnings_growth"),
            ]:
                if metric_col in region_df.columns:
                    region_stats["growth"][metric_name] = compute_stats(region_df[metric_col])

            # Leverage by region
            for metric_col, metric_name in [
                ("debt_to_equity", "debt_to_equity"),
                ("debt_to_assets", "debt_to_assets"),
            ]:
                if metric_col in region_df.columns:
                    region_stats["leverage"][metric_name] = compute_stats(region_df[metric_col])

            by_region_group[str(region)] = region_stats

        dashboard["by_region"] = by_region_group

    return dashboard
