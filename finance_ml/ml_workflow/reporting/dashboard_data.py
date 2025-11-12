"""
Dashboard data preparation functions.

This module provides helper functions to prepare data for interactive dashboards,
including financial metrics calculation, data quality alerts, and Plotly chart data.

Critical for existing Streamlit and Dash dashboard applications.

Phase 9.8 - Reporting Refactor
"""

import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_financial_metrics_dashboard(df: pd.DataFrame, group_by: Optional[str] = None) -> Dict:
    """
    Calculate comprehensive financial metrics dashboard.

    Computes statistics for four categories of financial metrics:
    - Valuation: P/E, P/B, EV/EBITDA
    - Profitability: Margins (gross, operating, net), ROE, ROA
    - Growth: Revenue growth, earnings growth
    - Leverage: Debt-to-equity, debt ratios

    Args:
        df: DataFrame with financial data
        group_by: Optional column name to group by (e.g., 'sector', 'region')

    Returns:
        Dictionary with metrics organized by category, each containing
        mean, median, std, min, max for available metrics

    Example:
        >>> df = pd.DataFrame({
        ...     'p_e': [15, 20, 25],
        ...     'roe': [0.15, 0.20, 0.18],
        ...     'sector': ['Tech', 'Tech', 'Finance']
        ... })
        >>> metrics = calculate_financial_metrics_dashboard(df, group_by='sector')
        >>> 'valuation' in metrics
        True
    """
    dashboard = {
        "valuation": {},
        "profitability": {},
        "growth": {},
        "leverage": {},
    }

    # Define metric mappings
    valuation_metrics = ["p_e", "p_b", "ev_ebitda"]
    profitability_metrics = ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]
    growth_metrics = ["revenue_growth", "earnings_growth", "ebitda_growth"]
    leverage_metrics = ["debt_to_equity", "debt_to_assets", "net_debt_to_ebitda"]

    def calculate_stats(series: pd.Series) -> Dict:
        """Calculate statistics for a series, handling NaN values."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        return {
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "std": float(clean_series.std()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "count": int(len(clean_series)),
        }

    # Calculate valuation metrics
    for metric in valuation_metrics:
        if metric in df.columns:
            dashboard["valuation"][metric] = calculate_stats(df[metric])

    # Calculate profitability metrics
    for metric in profitability_metrics:
        if metric in df.columns:
            dashboard["profitability"][metric] = calculate_stats(df[metric])

    # Calculate growth metrics
    for metric in growth_metrics:
        if metric in df.columns:
            dashboard["growth"][metric] = calculate_stats(df[metric])

    # Calculate leverage metrics
    for metric in leverage_metrics:
        if metric in df.columns:
            dashboard["leverage"][metric] = calculate_stats(df[metric])

    # If group_by is specified, add grouped statistics
    if group_by and group_by in df.columns:
        dashboard["by_group"] = {}
        for group_val in df[group_by].dropna().unique():
            group_df = df[df[group_by] == group_val]
            dashboard["by_group"][str(group_val)] = {
                "valuation": {},
                "profitability": {},
                "growth": {},
                "leverage": {},
            }

            # Valuation by group
            for metric in valuation_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["valuation"][metric] = calculate_stats(
                        group_df[metric]
                    )

            # Profitability by group
            for metric in profitability_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["profitability"][metric] = (
                        calculate_stats(group_df[metric])
                    )

            # Growth by group
            for metric in growth_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["growth"][metric] = calculate_stats(
                        group_df[metric]
                    )

            # Leverage by group
            for metric in leverage_metrics:
                if metric in group_df.columns:
                    dashboard["by_group"][str(group_val)]["leverage"][metric] = calculate_stats(
                        group_df[metric]
                    )

    return dashboard


def generate_data_quality_alerts(df: pd.DataFrame, outlier_threshold: float = 3.0) -> List[Dict]:
    """
    Generate data quality alerts for financial data.

    Detects:
    - Missing values (NaN, null)
    - Statistical outliers (using Z-score method)
    - Negative values in metrics that should be positive
    - Extreme values that may indicate data errors

    Args:
        df: DataFrame with financial data
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        List of alert dictionaries with keys:
        - severity: 'low', 'medium', 'high', 'critical'
        - message: Human-readable alert message
        - column: Column name with the issue
        - count: Number of rows affected (optional)

    Example:
        >>> df = pd.DataFrame({
        ...     'market_cap': [1e9, 2e9, np.nan, -1e6],
        ...     'last_price': [100, 150, 200, 0]
        ... })
        >>> alerts = generate_data_quality_alerts(df)
        >>> len(alerts) > 0
        True
    """
    alerts = []

    # Financial columns that should not be negative
    positive_only_columns = [
        "market_cap",
        "revenue",
        "total_assets",
        "total_equity",
        "ebitda",
        "last_price",
        "price_target",
    ]

    # Check for missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct_missing = (count / len(df)) * 100
            if pct_missing > 50:
                severity = "critical"
            elif pct_missing > 20:
                severity = "high"
            elif pct_missing > 5:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(
                {
                    "severity": severity,
                    "message": f"Column '{col}' has {count} missing values ({pct_missing:.1f}%)",
                    "column": col,
                    "count": int(count),
                }
            )

    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        clean_data = df[col].dropna()
        if len(clean_data) > 10:  # Need sufficient data for outlier detection
            mean = clean_data.mean()
            std = clean_data.std()

            if std > 0:  # Avoid division by zero
                z_scores = np.abs((clean_data - mean) / std)
                outlier_count = (z_scores > outlier_threshold).sum()

                if outlier_count > 0:
                    pct_outliers = (outlier_count / len(clean_data)) * 100

                    if pct_outliers > 10:
                        severity = "high"
                    elif pct_outliers > 5:
                        severity = "medium"
                    else:
                        severity = "low"

                    alerts.append(
                        {
                            "severity": severity,
                            "message": f"Column '{col}' has {outlier_count} outliers ({pct_outliers:.1f}%) beyond {outlier_threshold} standard deviations",
                            "column": col,
                            "count": int(outlier_count),
                        }
                    )

    # Check for negative values in columns that should be positive
    for col in positive_only_columns:
        if col in df.columns:
            clean_data = df[col].dropna()
            negative_count = (clean_data < 0).sum()

            if negative_count > 0:
                alerts.append(
                    {
                        "severity": "high",
                        "message": f"Column '{col}' has {negative_count} negative values (should be positive)",
                        "column": col,
                        "count": int(negative_count),
                    }
                )

    # Check for zero or near-zero values in key financial metrics
    critical_columns = ["market_cap", "revenue", "last_price"]
    for col in critical_columns:
        if col in df.columns:
            clean_data = df[col].dropna()
            zero_count = (clean_data == 0).sum()
            near_zero_count = ((clean_data > 0) & (clean_data < 0.01)).sum()

            if zero_count > 0:
                alerts.append(
                    {
                        "severity": "medium",
                        "message": f"Column '{col}' has {zero_count} zero values",
                        "column": col,
                        "count": int(zero_count),
                    }
                )

            if near_zero_count > 0:
                alerts.append(
                    {
                        "severity": "low",
                        "message": f"Column '{col}' has {near_zero_count} near-zero values (< 0.01)",
                        "column": col,
                        "count": int(near_zero_count),
                    }
                )

    return alerts


def prepare_plotly_dashboard_data(
    df: pd.DataFrame, include_timeseries: bool = False, color_scheme: str = "plotly"
) -> Dict:
    """
    Prepare structured data for interactive Plotly dashboards.

    Generates data structures optimized for various Plotly chart types:
    scatter plots, histograms, box plots, heatmaps, sunburst charts, and treemaps.

    Args:
        df: DataFrame with financial data
        include_timeseries: Whether to include time-series data (requires 'date' column)
        color_scheme: Color scheme for visualizations (default: 'plotly')

    Returns:
        Dictionary with data structured for different Plotly chart types

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'MSFT'],
        ...     'sector': ['Tech', 'Tech'],
        ...     'market_cap': [2.5e12, 2.3e12],
        ...     'mispricing_score': [0.15, -0.10],
        ...     'last_price': [150, 300]
        ... })
        >>> data = prepare_plotly_dashboard_data(df)
        >>> 'scatter_data' in data
        True
    """
    result = {
        "scatter_data": {},
        "histogram_data": {},
        "box_data": {},
        "heatmap_data": {},
        "sunburst_data": {},
        "treemap_data": {},
        "color_scales": {"default": color_scheme},
    }

    # 1. Scatter plot data (mispricing vs market cap)
    if all(col in df.columns for col in ["last_price", "market_cap", "mispricing_score"]):
        result["scatter_data"] = {
            "x": df["market_cap"].tolist(),
            "y": df["mispricing_score"].tolist(),
            "text": df["ticker"].tolist() if "ticker" in df.columns else None,
            "color": df["sector"].tolist() if "sector" in df.columns else None,
            "size": df["last_price"].tolist(),
        }

    # 2. Histogram data (mispricing by sector)
    if "mispricing_score" in df.columns and "sector" in df.columns:
        hist_by_sector = []
        for sector in df["sector"].dropna().unique():
            sector_data = df[df["sector"] == sector]["mispricing_score"].dropna()
            if len(sector_data) > 0:
                hist_by_sector.append(
                    {"sector": sector, "values": sector_data.tolist(), "name": sector}
                )
        result["histogram_data"]["mispricing_by_sector"] = hist_by_sector

    # 3. Box plot data (sector and region comparisons)
    box_comparisons = {}

    if "sector" in df.columns and "p_e" in df.columns:
        sector_box = []
        for sector in df["sector"].dropna().unique():
            sector_data = df[df["sector"] == sector]["p_e"].dropna()
            if len(sector_data) > 0:
                sector_box.append({"name": sector, "y": sector_data.tolist()})
        box_comparisons["sector_comparisons"] = sector_box

    if "region" in df.columns and "roe" in df.columns:
        region_box = []
        for region in df["region"].dropna().unique():
            region_data = df[df["region"] == region]["roe"].dropna()
            if len(region_data) > 0:
                region_box.append({"name": region, "y": region_data.tolist()})
        box_comparisons["region_comparisons"] = region_box

    result["box_data"] = box_comparisons

    # 4. Heatmap data (correlation matrix)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Select key financial metrics if available
        key_metrics = ["p_e", "p_b", "roe", "revenue_growth", "mispricing_score", "market_cap"]
        available_metrics = [col for col in key_metrics if col in numeric_cols]

        if len(available_metrics) >= 2:
            corr_matrix = df[available_metrics].corr()
            result["heatmap_data"]["correlation_matrix"] = {
                "z": corr_matrix.values.tolist(),
                "x": corr_matrix.columns.tolist(),
                "y": corr_matrix.index.tolist(),
            }

    # 5. Sunburst chart data (hierarchical: region > sector > ticker)
    if all(col in df.columns for col in ["region", "sector", "ticker"]):
        labels = ["All"]
        parents = [""]
        values = [len(df)]

        # Region level
        for region in df["region"].dropna().unique():
            labels.append(str(region))
            parents.append("All")
            region_count = len(df[df["region"] == region])
            values.append(region_count)

            # Sector level within region
            region_df = df[df["region"] == region]
            for sector in region_df["sector"].dropna().unique():
                sector_label = f"{region}_{sector}"
                labels.append(str(sector))
                parents.append(str(region))
                sector_count = len(region_df[region_df["sector"] == sector])
                values.append(sector_count)

        result["sunburst_data"] = {"labels": labels, "parents": parents, "values": values}

    # 6. Treemap data (sector/region breakdown with market cap)
    if all(col in df.columns for col in ["sector", "region", "market_cap"]):
        labels = []
        parents = []
        values = []

        # Add root
        labels.append("All")
        parents.append("")
        values.append(df["market_cap"].sum())

        # Add sectors
        for sector in df["sector"].dropna().unique():
            sector_df = df[df["sector"] == sector]
            labels.append(str(sector))
            parents.append("All")
            values.append(sector_df["market_cap"].sum())

            # Add regions within sectors
            for region in sector_df["region"].dropna().unique():
                region_df = sector_df[sector_df["region"] == region]
                labels.append(f"{sector}_{region}")
                parents.append(str(sector))
                values.append(region_df["market_cap"].sum())

        result["treemap_data"] = {"labels": labels, "parents": parents, "values": values}

    # 7. Time-series data (optional)
    if include_timeseries and "date" in df.columns:
        ts_data = {}

        # Try common price columns
        price_col = None
        for col_name in ["price", "last_price", "close", "adj_close"]:
            if col_name in df.columns:
                price_col = col_name
                break

        if price_col:
            # Aggregate by date
            daily_avg = df.groupby("date")[price_col].mean().reset_index()
            ts_data = {"dates": daily_avg["date"].tolist(), "values": daily_avg[price_col].tolist()}
        elif len(df.select_dtypes(include=[np.number]).columns) > 0:
            # Fallback: use first numeric column
            numeric_col = df.select_dtypes(include=[np.number]).columns[0]
            daily_avg = df.groupby("date")[numeric_col].mean().reset_index()
            ts_data = {
                "dates": daily_avg["date"].tolist(),
                "values": daily_avg[numeric_col].tolist(),
            }

        result["timeseries_data"] = ts_data

    return result
