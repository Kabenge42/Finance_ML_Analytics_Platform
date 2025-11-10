"""
Phase 9.2: EDA Module

Quick summaries, distributions, correlations, and sector slices for exploratory data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def eda_summary(
    df: pd.DataFrame,
    sector_column: str = "sector",
    include_correlations: bool = False,
) -> Dict[str, Any]:
    """
    Generate comprehensive EDA summary statistics for a dataframe.

    Args:
        df: Input dataframe
        sector_column: Name of sector column for sector-specific analysis
        include_correlations: Whether to include correlation matrix

    Returns:
        Dictionary containing:
        - shape: (rows, columns)
        - columns: List of column names
        - dtypes: Data types of columns
        - missing_values: Missing value counts and percentages
        - numeric_summary: Summary statistics for numeric columns
        - categorical_summary: Value counts for categorical columns
        - correlations: Correlation matrix (if include_correlations=True)
    """
    summary = {}

    # Basic information
    summary["shape"] = df.shape
    summary["columns"] = list(df.columns)
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missing values analysis
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100) if len(df) > 0 else pd.Series()
    summary["missing_values"] = {
        "counts": missing_counts.to_dict(),
        "percentages": missing_pct.to_dict() if len(df) > 0 else {},
    }

    # Numeric column summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        summary["numeric_summary"] = numeric_stats.to_dict()
    else:
        summary["numeric_summary"] = {}

    # Categorical column summary
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    categorical_summary = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        categorical_summary[col] = {
            "unique_count": df[col].nunique(),
            "top_values": value_counts.head(10).to_dict(),
        }
    summary["categorical_summary"] = categorical_summary

    # Sector-specific analysis if sector column exists
    if sector_column in df.columns:
        sector_counts = df[sector_column].value_counts()
        summary["sector_distribution"] = sector_counts.to_dict()

    # Correlations if requested
    if include_correlations and len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        summary["correlations"] = corr_matrix.to_dict()

    return summary


def sector_distribution_summary(
    df: pd.DataFrame,
    sector_column: str = "sector",
    metrics: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate sector-wise distribution summaries for specified metrics.

    Args:
        df: Input dataframe
        sector_column: Name of sector column
        metrics: List of metric columns to analyze (defaults to all numeric)

    Returns:
        Dictionary mapping metric names to sector-wise summary dataframes
    """
    if sector_column not in df.columns:
        logger.warning(f"Sector column '{sector_column}' not found in dataframe")
        return {}

    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()

    summaries = {}
    for metric in metrics:
        if metric in df.columns:
            sector_summary = df.groupby(sector_column)[metric].describe()
            summaries[metric] = sector_summary

    return summaries


def correlation_analysis(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.

    Args:
        df: Input dataframe
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations required

    Returns:
        Correlation matrix as dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        logger.warning("Need at least 2 numeric columns for correlation analysis")
        return pd.DataFrame()

    return df[numeric_cols].corr(method=method, min_periods=min_periods)


def distribution_summary(
    df: pd.DataFrame,
    column: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Generate distribution summary for a specific column.

    Args:
        df: Input dataframe
        column: Column name to analyze
        bins: Number of bins for histogram

    Returns:
        Dictionary with distribution statistics and histogram data
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    series = df[column].dropna()

    summary = {
        "count": len(series),
        "mean": series.mean() if pd.api.types.is_numeric_dtype(series) else None,
        "median": series.median() if pd.api.types.is_numeric_dtype(series) else None,
        "std": series.std() if pd.api.types.is_numeric_dtype(series) else None,
        "min": series.min() if pd.api.types.is_numeric_dtype(series) else None,
        "max": series.max() if pd.api.types.is_numeric_dtype(series) else None,
    }

    # Histogram data for numeric columns
    if pd.api.types.is_numeric_dtype(series):
        hist_counts, bin_edges = np.histogram(series, bins=bins)
        summary["histogram"] = {
            "counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

    return summary
