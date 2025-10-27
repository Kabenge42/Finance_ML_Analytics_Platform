"""
finance_ml.advanced_eda - Advanced Exploratory Data Analysis for Phase 9.2

Provides comprehensive statistical analysis and visualization functions for financial data:
- Correlation analysis (Pearson, Spearman, Kendall)
- Distribution testing (normality, skewness, kurtosis)
- Feature importance analysis (Mutual Information, Random Forest)
- Automated EDA report generation
- Sector and region-specific analysis

Author: Finance ML Analytics Platform
Date: 2025-10-28
Phase: 9.2 - Advanced EDA with Statistical Analysis
"""

import warnings
from pathlib import Path
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


def calculate_correlation_matrix(
    df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to include in correlation matrix. If None, uses all numeric columns.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns
    -------
    pd.DataFrame
        Correlation matrix

    Examples
    --------
    >>> corr = calculate_correlation_matrix(df, columns=['p_e', 'p_b'], method='pearson')
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns found for correlation analysis")

    # Select only specified columns that exist
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns exist in dataframe")

    data = df[available_cols].copy()

    # Calculate correlation
    corr_matrix = data.corr(method=method)

    return corr_matrix


def find_high_correlations(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.7,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Find pairs of features with high correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to analyze. If None, uses all numeric columns.
    threshold : float
        Absolute correlation threshold (default: 0.7)
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature_1, feature_2, correlation
        Sorted by absolute correlation value (descending)

    Examples
    --------
    >>> high_corr = find_high_correlations(df, threshold=0.8)
    """
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df, columns=columns, method=method)

    # Extract upper triangle (avoid duplicates)
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                pairs.append(
                    {
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": corr_val,
                    }
                )

    result = pd.DataFrame(pairs)

    if len(result) > 0:
        # Sort by absolute correlation
        result = result.assign(abs_corr=result["correlation"].abs())
        result = result.sort_values("abs_corr", ascending=False)
        result = result.drop("abs_corr", axis=1).reset_index(drop=True)

    return result


def test_normality(
    df: pd.DataFrame, columns: Optional[List[str]] = None, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test normality of distributions using Shapiro-Wilk test.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to test. If None, uses all numeric columns.
    alpha : float
        Significance level (default: 0.05)

    Returns
    -------
    pd.DataFrame
        Test results with columns: column, statistic, p_value, is_normal

    Examples
    --------
    >>> normality = test_normality(df, columns=['p_e', 'p_b'])
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if len(data) < 3:
            # Not enough data for test
            results.append(
                {"column": col, "statistic": np.nan, "p_value": np.nan, "is_normal": False}
            )
            continue

        # Shapiro-Wilk test
        try:
            statistic, p_value = shapiro(data)
            is_normal = p_value > alpha
        except Exception:
            statistic, p_value, is_normal = np.nan, np.nan, False

        results.append(
            {"column": col, "statistic": statistic, "p_value": p_value, "is_normal": is_normal}
        )

    return pd.DataFrame(results)


def calculate_distribution_stats(
    df: pd.DataFrame, columns: Optional[List[str]] = None, group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate distribution statistics for columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to analyze. If None, uses all numeric columns.
    group_by : Optional[str]
        Column to group by (e.g., 'sector')

    Returns
    -------
    pd.DataFrame
        Statistics including mean, median, std, skewness, kurtosis, min, max

    Examples
    --------
    >>> stats = calculate_distribution_stats(df, columns=['p_e'], group_by='sector')
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        raise ValueError("No valid columns found for analysis")

    if group_by is not None:
        # Group by specified column
        if group_by not in df.columns:
            raise ValueError(f"Group column '{group_by}' not found in dataframe")

        results = []
        for col in available_cols:
            for group_name, group_data in df.groupby(group_by):
                data = group_data[col].dropna()
                if len(data) > 0:
                    stats_dict = {
                        "column": col,
                        "group": group_name,
                        "mean": data.mean(),
                        "median": data.median(),
                        "std": data.std(),
                        "skewness": data.skew(),
                        "kurtosis": data.kurtosis(),
                        "min": data.min(),
                        "max": data.max(),
                    }
                    results.append(stats_dict)

        result_df = pd.DataFrame(results)
        # Set multi-index
        if len(result_df) > 0:
            result_df = result_df.set_index(["column", "group"])
    else:
        # No grouping
        stats_list = []
        for col in available_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats_dict = {
                    "mean": data.mean(),
                    "median": data.median(),
                    "std": data.std(),
                    "skewness": data.skew(),
                    "kurtosis": data.kurtosis(),
                    "min": data.min(),
                    "max": data.max(),
                }
                stats_list.append(stats_dict)

        result_df = pd.DataFrame(stats_list, index=available_cols)

    return result_df


def calculate_mutual_information(
    df: pd.DataFrame, target: str, features: Optional[List[str]] = None, random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate mutual information scores for features vs target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Target variable column name
    features : Optional[List[str]]
        Feature columns. If None, uses all numeric columns except target.
    random_state : int
        Random state for reproducibility

    Returns
    -------
    pd.DataFrame
        Feature importance with columns: feature, importance
        Sorted by importance (descending)

    Examples
    --------
    >>> mi = calculate_mutual_information(df, target='price_target', features=['p_e', 'p_b'])
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != target]

    available_features = [f for f in features if f in df.columns]

    if not available_features:
        raise ValueError("No valid features found")

    # Prepare data
    X = df[available_features].copy()
    y = df[target].copy()

    # Drop rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) == 0:
        raise ValueError("No valid samples after removing missing values")

    # Calculate mutual information
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_regression(X, y, random_state=random_state)

    # Create result dataframe
    result = pd.DataFrame({"feature": available_features, "importance": mi_scores})

    result = result.sort_values("importance", ascending=False).reset_index(drop=True)

    return result


def calculate_rf_importance(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Target variable column name
    features : Optional[List[str]]
        Feature columns. If None, uses all numeric columns except target.
    n_estimators : int
        Number of trees in Random Forest
    random_state : int
        Random state for reproducibility

    Returns
    -------
    pd.DataFrame
        Feature importance with columns: feature, importance
        Sorted by importance (descending)

    Examples
    --------
    >>> rf_imp = calculate_rf_importance(df, target='price_target')
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != target]

    available_features = [f for f in features if f in df.columns]

    if not available_features:
        raise ValueError("No valid features found")

    # Prepare data
    X = df[available_features].copy()
    y = df[target].copy()

    # Drop rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) < 10:
        raise ValueError("Insufficient samples for Random Forest training")

    # Train Random Forest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)

    # Extract feature importance
    importances = rf.feature_importances_

    # Create result dataframe
    result = pd.DataFrame({"feature": available_features, "importance": importances})

    result = result.sort_values("importance", ascending=False).reset_index(drop=True)

    return result


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    title: str = "EDA Report",
    include_plots: bool = False,
) -> str:
    """
    Generate automated EDA report and save to file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    output_dir : Union[str, Path]
        Directory to save report
    title : str
        Report title
    include_plots : bool
        Whether to include visualization plots (not implemented yet)

    Returns
    -------
    str
        Path to generated report file

    Examples
    --------
    >>> report_path = generate_eda_report(df, output_dir='./reports')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report content
    report_lines = []
    report_lines.append(f"# {title}\n")
    report_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")

    # Dataset overview
    report_lines.append("## Dataset Overview\n")
    report_lines.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    report_lines.append(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")

    # Column types
    report_lines.append("## Column Types\n")
    dtypes_summary = df.dtypes.value_counts()
    for dtype, count in dtypes_summary.items():
        report_lines.append(f"- {dtype}: {count} columns\n")
    report_lines.append("\n")

    # Missing values
    report_lines.append("## Missing Values\n")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {"Missing Count": missing[missing > 0], "Missing %": missing_pct[missing > 0]}
    )
    if len(missing_df) > 0:
        report_lines.append(missing_df.to_string())
    else:
        report_lines.append("No missing values detected.\n")
    report_lines.append("\n\n")

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report_lines.append("## Numeric Columns Summary\n")
        summary_stats = df[numeric_cols].describe()
        report_lines.append(summary_stats.to_string())
        report_lines.append("\n\n")

    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        report_lines.append("## Categorical Columns Summary\n")
        for col in categorical_cols[:10]:  # Limit to first 10
            value_counts = df[col].value_counts().head(5)
            report_lines.append(f"\n### {col}\n")
            report_lines.append(f"Unique values: {df[col].nunique()}\n")
            report_lines.append("Top 5 values:\n")
            report_lines.append(value_counts.to_string())
            report_lines.append("\n")

    # Write report to file
    report_filename = f"eda_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = output_dir / report_filename

    with open(report_path, "w") as f:
        f.writelines(report_lines)

    return str(report_path)


def generate_eda_summary(df: pd.DataFrame) -> Dict:
    """
    Generate EDA summary as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    Dict
        Summary dictionary with keys: shape, numeric_columns, categorical_columns,
        missing_values, summary_statistics

    Examples
    --------
    >>> summary = generate_eda_summary(df)
    >>> print(summary['shape'])
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Missing values
    missing = df.isnull().sum()
    missing_dict = {col: int(count) for col, count in missing.items() if count > 0}

    # Summary statistics for numeric columns
    summary_stats = {}
    if numeric_cols:
        desc = df[numeric_cols].describe()
        summary_stats = desc.to_dict()

    summary = {
        "shape": df.shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing_dict,
        "summary_statistics": summary_stats,
    }

    return summary


def analyze_by_sector(
    df: pd.DataFrame, metrics: List[str], sector_column: str = "sector"
) -> pd.DataFrame:
    """
    Analyze metrics by sector.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    metrics : List[str]
        Metrics to analyze
    sector_column : str
        Column containing sector information

    Returns
    -------
    pd.DataFrame
        Sector-wise statistics with multi-level columns

    Examples
    --------
    >>> sector_stats = analyze_by_sector(df, metrics=['p_e', 'market_cap'])
    """
    if sector_column not in df.columns:
        raise ValueError(f"Sector column '{sector_column}' not found in dataframe")

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise ValueError("No valid metrics found in dataframe")

    # Group by sector and calculate statistics
    sector_groups = df.groupby(sector_column)[available_metrics]

    stats = pd.DataFrame()
    stats["count"] = sector_groups.size()

    for metric in available_metrics:
        stats[f"{metric}_mean"] = sector_groups[metric].mean()
        stats[f"{metric}_median"] = sector_groups[metric].median()
        stats[f"{metric}_std"] = sector_groups[metric].std()
        stats[f"{metric}_min"] = sector_groups[metric].min()
        stats[f"{metric}_max"] = sector_groups[metric].max()

    return stats


def analyze_by_region(
    df: pd.DataFrame, metrics: List[str], region_column: str = "region"
) -> pd.DataFrame:
    """
    Analyze metrics by region.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    metrics : List[str]
        Metrics to analyze
    region_column : str
        Column containing region information

    Returns
    -------
    pd.DataFrame
        Region-wise statistics

    Examples
    --------
    >>> region_stats = analyze_by_region(df, metrics=['p_e', 'revenue'])
    """
    if region_column not in df.columns:
        raise ValueError(f"Region column '{region_column}' not found in dataframe")

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise ValueError("No valid metrics found in dataframe")

    # Group by region and calculate statistics
    region_groups = df.groupby(region_column)[available_metrics]

    stats = pd.DataFrame()
    stats["count"] = region_groups.size()

    for metric in available_metrics:
        stats[f"{metric}_mean"] = region_groups[metric].mean()
        stats[f"{metric}_median"] = region_groups[metric].median()
        stats[f"{metric}_std"] = region_groups[metric].std()

    return stats


def compare_sector_distributions(
    df: pd.DataFrame,
    metric: str,
    test: str = "anova",
    sector_column: str = "sector",
    alpha: float = 0.05,
) -> Dict:
    """
    Statistically compare distributions across sectors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    metric : str
        Metric to compare
    test : str
        Statistical test: 'anova' (one-way ANOVA) or 'kruskal' (Kruskal-Wallis)
    sector_column : str
        Column containing sector information
    alpha : float
        Significance level

    Returns
    -------
    Dict
        Test results with keys: statistic, p_value, significant, test_name

    Examples
    --------
    >>> comparison = compare_sector_distributions(df, metric='p_e', test='anova')
    """
    if sector_column not in df.columns:
        raise ValueError(f"Sector column '{sector_column}' not found")

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found")

    # Prepare data by sector
    sector_groups = []
    for sector, group in df.groupby(sector_column):
        data = group[metric].dropna()
        if len(data) > 0:
            sector_groups.append(data.values)

    if len(sector_groups) < 2:
        raise ValueError("Need at least 2 sectors with valid data for comparison")

    # Perform statistical test
    if test == "anova":
        statistic, p_value = stats.f_oneway(*sector_groups)
        test_name = "One-way ANOVA"
    elif test == "kruskal":
        statistic, p_value = stats.kruskal(*sector_groups)
        test_name = "Kruskal-Wallis H-test"
    else:
        raise ValueError(f"Unknown test: {test}. Use 'anova' or 'kruskal'")

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "test_name": test_name,
        "alpha": alpha,
    }

    return result
