"""
Phase 9.2: EDA Module

Quick summaries, distributions, correlations, and sector slices for exploratory data analysis.

Phase 9.3 Integration: Enhanced with Phase 9.3 feature family tracking and reporting.
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.eda.phase93_categories import (
    categorize_dataframe_columns,
    get_phase93_coverage_stats,
    get_category_description,
)

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


# ============================================================================
# Phase 9.3 Integration: Feature Family Tracking
# ============================================================================


def eda_summary_with_phase93(
    df: pd.DataFrame,
    sector_column: str = "sector",
    include_correlations: bool = False,
) -> Dict[str, Any]:
    """
    Generate EDA summary with explicit Phase 9.3 feature category coverage.

    This is an enhanced version of eda_summary() that includes Phase 9.3
    feature family tracking.

    Args:
        df: Input dataframe
        sector_column: Name of sector column for sector-specific analysis
        include_correlations: Whether to include correlation matrix

    Returns:
        Dictionary with standard EDA summary plus phase93_category_coverage
    """
    # Get standard EDA summary
    summary = eda_summary(df, sector_column, include_correlations)

    # Add Phase 9.3 category coverage
    summary["phase93_category_coverage"] = get_phase93_coverage_stats(df)

    return summary


def generate_phase93_coverage_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive Phase 9.3 feature coverage report.

    Args:
        df: DataFrame with potential Phase 9.3 features

    Returns:
        Dictionary with coverage statistics and breakdown
    """
    categorized = categorize_dataframe_columns(df)
    coverage_stats = get_phase93_coverage_stats(df)

    total_present = sum(coverage_stats.values())

    # Calculate coverage percentage (out of all possible Phase 9.3 features)
    from finance_ml.ml_workflow.eda.phase93_categories import list_all_phase93_features

    total_possible = len(list_all_phase93_features())
    coverage_pct = (total_present / total_possible * 100) if total_possible > 0 else 0.0

    report = {
        "total_phase93_features": total_present,
        "total_possible_features": total_possible,
        "coverage_percentage": coverage_pct,
        "category_breakdown": coverage_stats,
        "categorized_columns": categorized,
    }

    # Add category descriptions
    descriptions = {}
    for category in coverage_stats.keys():
        descriptions[category] = get_category_description(category)
    report["category_descriptions"] = descriptions

    return report


def analyze_phase93_by_sector(
    df: pd.DataFrame,
    sector_column: str = "sector",
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze Phase 9.3 feature distributions by sector.

    Args:
        df: DataFrame with Phase 9.3 features
        sector_column: Name of sector column

    Returns:
        Dictionary mapping sector names to their Phase 9.3 summaries
    """
    if sector_column not in df.columns:
        logger.warning(f"Sector column '{sector_column}' not found")
        return {}

    categorized = categorize_dataframe_columns(df)
    sector_analysis = {}

    for sector in df[sector_column].unique():
        if pd.isna(sector):
            continue

        sector_df = df[df[sector_column] == sector]
        sector_summary = {}

        # For each Phase 9.3 category, compute sector-specific statistics
        for category, features in categorized.items():
            present_features = [f for f in features if f in sector_df.columns]
            if present_features:
                # Compute summary stats for numeric features in this category
                numeric_features = (
                    sector_df[present_features].select_dtypes(include=[np.number]).columns.tolist()
                )
                if numeric_features:
                    sector_summary[category] = {
                        "feature_count": len(numeric_features),
                        "mean_values": sector_df[numeric_features].mean().to_dict(),
                        "std_values": sector_df[numeric_features].std().to_dict(),
                    }

        sector_analysis[sector] = sector_summary

    return sector_analysis


# ============================================================================
# Aliases for code_guidelines.md v1.10 API compliance
# ============================================================================


# Alias: compute_descriptive_stats -> eda_summary
def compute_descriptive_stats(
    df: pd.DataFrame,
    sector_column: str = "sector",
    include_correlations: bool = False,
) -> Dict[str, Any]:
    """
    Compute descriptive statistics for the DataFrame.

    Alias for eda_summary() to match code_guidelines.md v1.10 API.

    Args:
        df: Input dataframe
        sector_column: Name of sector column for sector-specific analysis
        include_correlations: Whether to include correlation matrix

    Returns:
        Dictionary with descriptive statistics
    """
    return eda_summary(df, sector_column, include_correlations)


# Alias: plot_distributions -> distribution_summary (returns data for plotting)
def plot_distributions(
    df: pd.DataFrame,
    column: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Generate distribution summary data for plotting.

    Alias for distribution_summary() to match code_guidelines.md v1.10 API.

    Args:
        df: Input dataframe
        column: Column to analyze
        bins: Number of histogram bins

    Returns:
        Dictionary with distribution data suitable for plotting
    """
    return distribution_summary(df, column, bins)


# Alias: compute_correlation_matrix -> correlation_analysis
def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.

    Alias for correlation_analysis() to match code_guidelines.md v1.10 API.

    Args:
        df: Input dataframe
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum observations required per pair

    Returns:
        Correlation matrix as DataFrame
    """
    return correlation_analysis(df, method, min_periods)
