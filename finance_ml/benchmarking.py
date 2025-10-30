"""
Finance ML Benchmarking Module

Sector and region-specific benchmarking functions for financial analysis.

Phase 9.2: Exploratory Data Analysis - Benchmarking Implementation
"""

import logging
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats


def compare_sector_distributions(
    df: pd.DataFrame, metrics: List[str], sector_column: str = "sector"
) -> pd.DataFrame:
    """Compare distribution of valuation metrics across sectors.

    Calculates comprehensive statistics for specified metrics within each sector.

    Args:
        df: DataFrame with sector and metric columns
        metrics: List of metric column names to analyze (e.g., ['p_e', 'p_b', 'ev_ebitda'])
        sector_column: Name of the sector column (default: 'sector')

    Returns:
        DataFrame with columns: sector, metric, mean, median, std, min, max, count

    Example:
        >>> df = pd.DataFrame({
        ...     'sector': ['Tech', 'Tech', 'Finance', 'Finance'],
        ...     'p_e': [25, 30, 12, 10],
        ...     'p_b': [5, 6, 1.2, 1.0]
        ... })
        >>> result = compare_sector_distributions(df, metrics=['p_e', 'p_b'])
    """
    if sector_column not in df.columns:
        logging.warning(f"Sector column '{sector_column}' not found in DataFrame")
        return pd.DataFrame()

    results = []

    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric '{metric}' not found in DataFrame, skipping")
            continue

        # Group by sector and calculate statistics
        for sector in df[sector_column].dropna().unique():
            sector_data = df[df[sector_column] == sector][metric].dropna()

            if len(sector_data) == 0:
                continue

            results.append(
                {
                    "sector": sector,
                    "metric": metric,
                    "mean": float(sector_data.mean()),
                    "median": float(sector_data.median()),
                    "std": float(sector_data.std()),
                    "min": float(sector_data.min()),
                    "max": float(sector_data.max()),
                    "count": int(len(sector_data)),
                    "q25": float(sector_data.quantile(0.25)),
                    "q75": float(sector_data.quantile(0.75)),
                }
            )

    return pd.DataFrame(results)


def compare_regional_valuations(
    df: pd.DataFrame,
    metrics: List[str],
    region_column: str = "region",
    include_tests: bool = False,
    test_method: str = "anova",
) -> Union[pd.DataFrame, Dict]:
    """Compare valuation metrics across regions with optional statistical tests.

    Args:
        df: DataFrame with region and metric columns
        metrics: List of metric column names to analyze
        region_column: Name of the region column (default: 'region')
        include_tests: If True, return dict with distributions and statistical tests
        test_method: Statistical test method ('anova' or 'kruskal')

    Returns:
        If include_tests=False: DataFrame with region, metric, and statistics
        If include_tests=True: Dict with 'distributions' (DataFrame) and 'statistical_tests' (Dict)

    Example:
        >>> result = compare_regional_valuations(df, metrics=['p_e'], include_tests=True)
        >>> print(result['statistical_tests']['p_e']['p_value'])
    """
    if region_column not in df.columns:
        logging.warning(f"Region column '{region_column}' not found in DataFrame")
        return (
            pd.DataFrame()
            if not include_tests
            else {"distributions": pd.DataFrame(), "statistical_tests": {}}
        )

    results = []

    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric '{metric}' not found in DataFrame, skipping")
            continue

        # Group by region and calculate statistics
        for region in df[region_column].dropna().unique():
            region_data = df[df[region_column] == region][metric].dropna()

            if len(region_data) == 0:
                continue

            results.append(
                {
                    "region": region,
                    "metric": metric,
                    "mean": float(region_data.mean()),
                    "median": float(region_data.median()),
                    "std": float(region_data.std()),
                    "min": float(region_data.min()),
                    "max": float(region_data.max()),
                    "count": int(len(region_data)),
                }
            )

    distributions_df = pd.DataFrame(results)

    if not include_tests:
        return distributions_df

    # Perform statistical tests for regional differences
    statistical_tests = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Get data for each region
        region_groups = []
        regions = df[region_column].dropna().unique()

        for region in regions:
            region_data = df[df[region_column] == region][metric].dropna()
            if len(region_data) >= 3:  # Need at least 3 samples
                region_groups.append(region_data.values)

        if len(region_groups) < 2:
            logging.warning(f"Not enough regions with data for metric '{metric}'")
            continue

        # Perform statistical test
        try:
            if test_method == "anova":
                statistic, p_value = stats.f_oneway(*region_groups)
                method = "ANOVA"
            else:  # kruskal
                statistic, p_value = stats.kruskal(*region_groups)
                method = "Kruskal-Wallis"

            statistical_tests[metric] = {
                "method": method,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n_regions": len(region_groups),
            }
        except Exception as e:
            logging.warning(f"Statistical test failed for metric '{metric}': {e}")

    return {"distributions": distributions_df, "statistical_tests": statistical_tests}


def find_peer_group(
    df: pd.DataFrame,
    ticker: str,
    n_peers: int = 5,
    sector_column: str = "sector",
    criteria: str = "market_cap",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Find peer companies within the same sector.

    Identifies similar companies based on sector and a similarity criterion
    (e.g., market cap, P/E ratio).

    Args:
        df: DataFrame with stock data
        ticker: Target stock ticker symbol
        n_peers: Number of peer stocks to return
        sector_column: Name of the sector column
        criteria: Column to use for similarity (default: 'market_cap')
        ticker_column: Name of the ticker column

    Returns:
        DataFrame with peer stocks (excluding target stock)

    Example:
        >>> peers = find_peer_group(df, ticker='AAPL', n_peers=3, criteria='market_cap')
    """
    if ticker_column not in df.columns:
        logging.error(f"Ticker column '{ticker_column}' not found")
        return pd.DataFrame()

    # Find target stock
    target_row = df[df[ticker_column] == ticker]
    if target_row.empty:
        logging.error(f"Ticker '{ticker}' not found in DataFrame")
        return pd.DataFrame()

    target_sector = target_row[sector_column].iloc[0]
    target_value = target_row[criteria].iloc[0] if criteria in df.columns else None

    if pd.isna(target_sector):
        logging.warning(f"Target stock '{ticker}' has no sector, returning empty peers")
        return pd.DataFrame()

    # Filter to same sector, exclude target
    peers_df = df[(df[sector_column] == target_sector) & (df[ticker_column] != ticker)].copy()

    if peers_df.empty:
        logging.warning(f"No peers found in sector '{target_sector}'")
        return pd.DataFrame()

    # Sort by similarity to target based on criteria
    if criteria in peers_df.columns and target_value is not None:
        peers_df["_similarity"] = abs(peers_df[criteria] - target_value)
        peers_df = peers_df.sort_values("_similarity")
        peers_df = peers_df.drop("_similarity", axis=1)

    # Return top n_peers
    return peers_df.head(n_peers)


def compare_to_peers(
    df: pd.DataFrame,
    ticker: str,
    metrics: List[str],
    n_peers: int = 5,
    criteria: str = "market_cap",
    ticker_column: str = "ticker",
) -> Dict:
    """Compare a stock's metrics to its peer group.

    Args:
        df: DataFrame with stock data
        ticker: Target stock ticker symbol
        metrics: List of metrics to compare
        n_peers: Number of peers to include in comparison
        criteria: Similarity criterion for peer selection
        ticker_column: Name of the ticker column

    Returns:
        Dictionary with target values, peer statistics, and deviations

    Example:
        >>> comparison = compare_to_peers(df, 'AAPL', metrics=['p_e', 'p_b'], n_peers=3)
        >>> print(comparison['p_e']['deviation_from_mean'])
    """
    # Find peers
    peers = find_peer_group(
        df, ticker, n_peers=n_peers, criteria=criteria, ticker_column=ticker_column
    )

    if peers.empty:
        logging.warning(f"No peers found for '{ticker}'")
        return {}

    # Get target stock data
    target = df[df[ticker_column] == ticker]
    if target.empty:
        return {}

    result = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        target_value = target[metric].iloc[0]
        peer_values = peers[metric].dropna()

        if len(peer_values) == 0:
            continue

        peers_mean = float(peer_values.mean())
        peers_median = float(peer_values.median())
        peers_std = float(peer_values.std())

        # Calculate deviation
        deviation_from_mean = float(target_value - peers_mean)
        deviation_pct = (
            float((target_value - peers_mean) / peers_mean * 100) if peers_mean != 0 else 0
        )
        z_score = float((target_value - peers_mean) / peers_std) if peers_std != 0 else 0

        result[metric] = {
            "target": float(target_value),
            "peers_mean": peers_mean,
            "peers_median": peers_median,
            "peers_std": peers_std,
            "deviation_from_mean": deviation_from_mean,
            "deviation_pct": deviation_pct,
            "z_score": z_score,
            "n_peers": int(len(peer_values)),
        }

    return result


def analyze_metric_trend(
    df: pd.DataFrame,
    ticker: str,
    metric: str,
    date_column: str = "date",
    ticker_column: str = "ticker",
) -> Optional[Dict]:
    """Analyze time-series trend for a specific metric.

    Performs linear regression to detect trend direction and magnitude.

    Args:
        df: DataFrame with time-series data
        ticker: Target stock ticker symbol
        metric: Metric column name to analyze
        date_column: Name of the date column
        ticker_column: Name of the ticker column

    Returns:
        Dictionary with trend_direction, slope, r_squared, or None if insufficient data

    Example:
        >>> trend = analyze_metric_trend(df, 'AAPL', 'p_e', date_column='date')
        >>> print(trend['trend_direction'])  # 'increasing', 'decreasing', or 'stable'
    """
    if date_column not in df.columns:
        logging.warning(f"Date column '{date_column}' not found")
        return None

    if ticker_column not in df.columns or metric not in df.columns:
        return None

    # Filter to target stock
    stock_df = df[df[ticker_column] == ticker].copy()

    if stock_df.empty or len(stock_df) < 3:
        logging.warning(f"Insufficient data for trend analysis of '{ticker}'")
        return None

    # Sort by date
    stock_df = stock_df.sort_values(date_column)

    # Get metric values
    metric_values = stock_df[metric].dropna()

    if len(metric_values) < 3:
        return None

    # Create time index (0, 1, 2, ...)
    time_index = np.arange(len(metric_values))

    # Perform linear regression
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, metric_values)

        # Determine trend direction
        if abs(slope) < std_err * 1.96:  # Not significantly different from 0
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return {
            "trend_direction": trend_direction,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "n_periods": int(len(metric_values)),
        }
    except Exception as e:
        logging.error(f"Trend analysis failed for '{ticker}': {e}")
        return None


def generate_benchmarking_report(
    df: pd.DataFrame,
    metrics: List[str],
    sector_column: str = "sector",
    region_column: str = "region",
) -> Dict:
    """Generate comprehensive benchmarking report.

    Combines sector distributions, regional valuations, and summary statistics.

    Args:
        df: DataFrame with stock data
        metrics: List of metrics to analyze
        sector_column: Name of the sector column
        region_column: Name of the region column

    Returns:
        Dictionary with 'sector_distributions', 'regional_valuations', and 'summary'

    Example:
        >>> report = generate_benchmarking_report(df, metrics=['p_e', 'p_b'])
        >>> print(report['summary']['total_stocks'])
    """
    report = {}

    # Sector distributions
    try:
        sector_dist = compare_sector_distributions(df, metrics, sector_column=sector_column)
        report["sector_distributions"] = (
            sector_dist.to_dict(orient="records") if not sector_dist.empty else []
        )
    except Exception as e:
        logging.warning(f"Sector distribution analysis failed: {e}")
        report["sector_distributions"] = []

    # Regional valuations
    try:
        regional_val = compare_regional_valuations(df, metrics, region_column=region_column)
        report["regional_valuations"] = (
            regional_val.to_dict(orient="records") if not regional_val.empty else []
        )
    except Exception as e:
        logging.warning(f"Regional valuation analysis failed: {e}")
        report["regional_valuations"] = []

    # Summary statistics
    summary = {
        "total_stocks": int(len(df)),
        "n_sectors": int(df[sector_column].nunique()) if sector_column in df.columns else 0,
        "n_regions": int(df[region_column].nunique()) if region_column in df.columns else 0,
        "metrics_analyzed": metrics,
    }

    report["summary"] = summary

    return report
