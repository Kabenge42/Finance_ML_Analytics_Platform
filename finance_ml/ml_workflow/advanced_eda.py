"""
finance_ml.advanced_eda - Advanced Exploratory Data Analysis for Phase 9.2

This module implements sophisticated EDA techniques including:
- Advanced correlation analysis (Pearson, Spearman, Kendall, distance correlation)
- Statistical hypothesis testing (ANOVA, t-tests, Kruskal-Wallis, Mann-Whitney)
- Normality tests and distribution analysis
- Multivariate analysis (PCA, t-SNE, UMAP)
- Automated EDA report generation
- Feature importance via mutual information and Random Forest

Part of Phase 9.2 implementation.

.. deprecated:: v9_8
    This module is maintained for backward compatibility. New code should use
    the EDA subpackage:
    - :mod:`finance_ml.ml_workflow.eda.descriptive` for basic statistics
    - :mod:`finance_ml.ml_workflow.eda.correlations` for correlation analysis
    - :mod:`finance_ml.ml_workflow.eda.distributions` for distribution analysis
    - :mod:`finance_ml.ml_workflow.eda.statistical_tests` for hypothesis testing
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# Emit deprecation warning when module is imported
warnings.warn(
    "The advanced_eda module is deprecated as of v9_8. "
    "Use finance_ml.ml_workflow.eda subpackage instead. "
    "See docs/improvement_plan/finance_ml_restructuring_plan.md for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrelationReport:
    """Container for correlation analysis results."""

    pearson_matrix: pd.DataFrame
    spearman_matrix: pd.DataFrame
    kendall_matrix: Optional[pd.DataFrame] = None
    top_positive: Optional[pd.DataFrame] = None
    top_negative: Optional[pd.DataFrame] = None


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""


@dataclass
class EDAReport:
    """Comprehensive EDA report container."""

    dataset_summary: Dict[str, Any]
    correlation_analysis: CorrelationReport
    normality_tests: Dict[str, StatisticalTestResult]
    distribution_stats: pd.DataFrame
    feature_importance: pd.DataFrame
    missing_values_summary: pd.DataFrame
    outlier_summary: Dict[str, int]
    sector_comparison: Optional[Dict[str, Any]] = None


def calculate_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson", columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate correlation matrix using specified method."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[columns].dropna()

    if method == "pearson":
        corr_matrix = data.corr(method="pearson")
    elif method == "spearman":
        corr_matrix = data.corr(method="spearman")
    elif method == "kendall":
        corr_matrix = data.corr(method="kendall")
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    logger.info(f"Calculated {method} correlation matrix for {len(columns)} columns")
    return corr_matrix


def find_top_correlations(
    corr_matrix: pd.DataFrame, n_top: int = 10
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Find top positive and negative correlations from a correlation matrix.

    Args:
        corr_matrix: Correlation matrix
        n_top: Number of top correlations to return for each category

    Returns:
        Tuple of (top_positive, top_negative) correlation lists
    """
    # Get upper triangle (avoid duplicates and self-correlations)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)

    # Convert to list of tuples
    correlations = []
    for i in range(len(upper_tri)):
        for j in range(i + 1, len(upper_tri)):
            var1 = upper_tri.index[i]
            var2 = upper_tri.columns[j]
            corr_value = upper_tri.iloc[i, j]

            if not pd.isna(corr_value):
                correlations.append((var1, var2, corr_value))

    # Sort by correlation value
    correlations.sort(key=lambda x: x[2], reverse=True)

    # Split into positive and negative
    top_positive = [c for c in correlations if c[2] > 0][:n_top]
    top_negative = [c for c in correlations if c[2] < 0][:n_top]

    return top_positive, top_negative


def test_normality(data: pd.Series, method: str = "shapiro") -> StatisticalTestResult:
    """Test if data follows normal distribution."""
    data_clean = data.dropna()

    if len(data_clean) < 3:
        return StatisticalTestResult(
            test_name=method,
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Insufficient data for test",
        )

    if method == "shapiro":
        if len(data_clean) > 5000:
            data_sample = data_clean.sample(n=5000, random_state=42)
        else:
            data_sample = data_clean
        stat, p_value = stats.shapiro(data_sample)
        test_name = "Shapiro-Wilk"
    elif method == "kstest":
        stat, p_value = stats.kstest(data_clean, "norm", args=(data_clean.mean(), data_clean.std()))
        test_name = "Kolmogorov-Smirnov"
    elif method == "anderson":
        result = stats.anderson(data_clean, dist="norm")
        stat = result.statistic
        critical_value = result.critical_values[2]
        p_value = 0.05 if stat > critical_value else 0.1
        test_name = "Anderson-Darling"
    else:
        raise ValueError(f"Unknown normality test method: {method}")

    significant = p_value < 0.05
    interpretation = (
        "Data is NOT normally distributed (reject H0)"
        if significant
        else "Data appears normally distributed (fail to reject H0)"
    )

    return StatisticalTestResult(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        significant=significant,
        interpretation=interpretation,
    )


def calculate_skewness_kurtosis(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate skewness and kurtosis for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) < 3:
            continue

        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        if abs(skew) < 0.5:
            skew_interp = "Fairly symmetric"
        elif abs(skew) < 1.0:
            skew_interp = "Moderately skewed"
        else:
            skew_interp = "Highly skewed"

        if skew > 0:
            skew_interp += " (right)"
        elif skew < 0:
            skew_interp += " (left)"

        if abs(kurt) < 0.5:
            kurt_interp = "Mesokurtic (normal-like tails)"
        elif kurt > 0:
            kurt_interp = "Leptokurtic (heavy tails)"
        else:
            kurt_interp = "Platykurtic (light tails)"

        results.append(
            {
                "column": col,
                "skewness": skew,
                "kurtosis": kurt,
                "skew_interpretation": skew_interp,
                "kurt_interpretation": kurt_interp,
            }
        )

    result_df = pd.DataFrame(results)
    logger.info(f"Calculated skewness and kurtosis for {len(results)} columns")
    return result_df


def detect_outliers_statistical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers using statistical methods with summary."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) < 4:
            continue

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(data)) * 100

        results.append(
            {
                "column": col,
                "n_outliers": n_outliers,
                "pct_outliers": pct_outliers,
                "method": method,
                "threshold": threshold,
            }
        )

    result_df = pd.DataFrame(results).sort_values("n_outliers", ascending=False)
    logger.info(f"Detected outliers in {len(results)} columns using {method} method")
    return result_df


def calculate_mutual_information(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = 20
) -> pd.DataFrame:
    """Calculate mutual information between features and target."""
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median())

    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42, n_neighbors=5)

    mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": mi_scores}).sort_values(
        "mutual_information", ascending=False
    )

    if top_k is not None:
        mi_df = mi_df.head(top_k)

    logger.info(f"Calculated mutual information for {len(X.columns)} features")
    return mi_df


def calculate_feature_importance_rf(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = 20, n_estimators: int = 100
) -> pd.DataFrame:
    """Calculate feature importance using Random Forest.

    Args:
        X: Features dataframe
        y: Target variable
        top_k: Number of top features to return (None for all)
        n_estimators: Number of trees in the forest

    Returns:
        DataFrame with feature names and importance scores
    """
    # Remove rows with missing target values
    valid_rows = ~y.isna()
    X_valid = X.loc[valid_rows]
    y_valid = y.loc[valid_rows]

    # Remove columns with any missing values
    valid_cols = ~X_valid.isna().any()
    X_clean = X_valid.loc[:, valid_cols]
    y_clean = y_valid

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=20
    )
    rf.fit(X_clean, y_clean)

    # Get feature importance - use X_clean.columns instead of X.columns
    importance_df = pd.DataFrame(
        {"feature": X_clean.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated Random Forest importance for {len(X_clean.columns)} features")
    return importance_df


def perform_pca(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    columns: Optional[List[str]] = None,
    scale: bool = True,
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Perform Principal Component Analysis."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[columns].dropna()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    if n_components is None:
        n_components = min(len(columns), len(data))

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(data_scaled)

    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    transformed_df = pd.DataFrame(transformed, index=data.index, columns=pc_columns)

    loadings_df = pd.DataFrame(pca.components_.T, index=columns, columns=pc_columns)

    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {n_components} components explain {explained_var:.2%} of variance")

    return pca, transformed_df, loadings_df


def calculate_optimal_pca_components(
    df: pd.DataFrame, variance_threshold: float = 0.95, columns: Optional[List[str]] = None
) -> int:
    """Calculate optimal number of PCA components for variance threshold."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[columns].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(random_state=42)
    pca.fit(data_scaled)

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

    logger.info(f"Optimal PCA: {n_components} components for {variance_threshold:.0%} variance")
    return int(n_components)


def compare_sector_means(
    df: pd.DataFrame, metric: str, sector_col: str = "sector", method: str = "anova"
) -> StatisticalTestResult:
    """Compare means across sectors using ANOVA or Kruskal-Wallis."""
    if sector_col not in df.columns or metric not in df.columns:
        return StatisticalTestResult(
            test_name=method,
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Required columns not found",
        )

    groups = []
    sectors = df[sector_col].dropna().unique()

    for sector in sectors:
        sector_data = df[df[sector_col] == sector][metric].dropna()
        if len(sector_data) > 0:
            groups.append(sector_data)

    if len(groups) < 2:
        return StatisticalTestResult(
            test_name=method,
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Insufficient groups for comparison",
        )

    if method == "anova":
        stat, p_value = stats.f_oneway(*groups)
        test_name = "One-Way ANOVA"
        effect_size = None
    elif method == "kruskal":
        stat, p_value = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis H-test"
        effect_size = None
    else:
        raise ValueError(f"Unknown test method: {method}")

    significant = p_value < 0.05
    interpretation = (
        f"Significant differences exist between sectors (p={p_value:.4f})"
        if significant
        else f"No significant differences between sectors (p={p_value:.4f})"
    )

    return StatisticalTestResult(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def compare_two_groups(
    group1: pd.Series, group2: pd.Series, method: str = "ttest", paired: bool = False
) -> StatisticalTestResult:
    """Compare two groups using t-test or Mann-Whitney U test."""
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return StatisticalTestResult(
            test_name=method,
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Insufficient data in one or both groups",
        )

    if method == "ttest":
        if paired:
            if len(group1_clean) != len(group2_clean):
                return StatisticalTestResult(
                    test_name="Paired t-test",
                    statistic=np.nan,
                    p_value=np.nan,
                    significant=False,
                    interpretation="Groups must have same length for paired test",
                )
            stat, p_value = stats.ttest_rel(group1_clean, group2_clean)
            test_name = "Paired t-test"
        else:
            stat, p_value = stats.ttest_ind(group1_clean, group2_clean)
            test_name = "Independent t-test"

        pooled_std = np.sqrt(
            (
                (len(group1_clean) - 1) * group1_clean.std() ** 2
                + (len(group2_clean) - 1) * group2_clean.std() ** 2
            )
            / (len(group1_clean) + len(group2_clean) - 2)
        )
        effect_size = (group1_clean.mean() - group2_clean.mean()) / pooled_std
    elif method == "mannwhitney":
        stat, p_value = stats.mannwhitneyu(group1_clean, group2_clean, alternative="two-sided")
        test_name = "Mann-Whitney U test"
        effect_size = None
    else:
        raise ValueError(f"Unknown test method: {method}")

    significant = p_value < 0.05
    interpretation = (
        f"Groups are significantly different (p={p_value:.4f})"
        if significant
        else f"No significant difference between groups (p={p_value:.4f})"
    )

    return StatisticalTestResult(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def generate_eda_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    sector_col: str = "sector",
    output_dir: Optional[Path] = None,
) -> EDAReport:
    """Generate comprehensive EDA report."""
    logger.info("Generating comprehensive EDA report...")

    dataset_summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
        "n_categorical": len(df.select_dtypes(include=["object", "category"]).columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]

    pearson_matrix = calculate_correlation_matrix(df, method="pearson", columns=numeric_cols)
    spearman_matrix = calculate_correlation_matrix(df, method="spearman", columns=numeric_cols)
    top_positive, top_negative = find_top_correlations(pearson_matrix, n_top=10)

    correlation_analysis = CorrelationReport(
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        top_positive=top_positive,
        top_negative=top_negative,
    )

    normality_tests = {}
    for col in numeric_cols[:10]:
        normality_tests[col] = test_normality(df[col], method="shapiro")

    distribution_stats = calculate_skewness_kurtosis(df, columns=numeric_cols)

    if target_col and target_col in df.columns:
        X = df[numeric_cols].drop(columns=[target_col], errors="ignore")
        y = df[target_col]
        feature_importance = calculate_mutual_information(X, y, top_k=20)
    else:
        feature_importance = pd.DataFrame()

    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum(),
            "missing_pct": (df.isnull().sum() / len(df)) * 100,
        }
    ).sort_values("missing_count", ascending=False)

    outlier_stats = detect_outliers_statistical(df, columns=numeric_cols[:10], method="iqr")
    outlier_summary = outlier_stats.set_index("column")["n_outliers"].to_dict()

    sector_comparison = None
    if sector_col in df.columns and len(numeric_cols) > 0:
        test_col = numeric_cols[0]
        sector_test = compare_sector_means(df, test_col, sector_col, method="anova")
        sector_comparison = {"test_column": test_col, "test_result": sector_test}

    report = EDAReport(
        dataset_summary=dataset_summary,
        correlation_analysis=correlation_analysis,
        normality_tests=normality_tests,
        distribution_stats=distribution_stats,
        feature_importance=feature_importance,
        missing_values_summary=missing_summary,
        outlier_summary=outlier_summary,
        sector_comparison=sector_comparison,
    )

    logger.info("✓ EDA report generation complete")
    return report


def generate_sector_comparison_report(
    df: pd.DataFrame, metrics: List[str], sector_col: str = "sector"
) -> pd.DataFrame:
    """Generate sector comparison report for multiple metrics."""
    if sector_col not in df.columns:
        logger.warning(f"Sector column '{sector_col}' not found")
        return pd.DataFrame()

    results = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        sector_stats = (
            df.groupby(sector_col)[metric]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )

        sector_stats["metric"] = metric
        results.append(sector_stats)

    if results:
        comparison_df = pd.concat(results, ignore_index=True)
        logger.info(f"Generated sector comparison for {len(metrics)} metrics")
        return comparison_df
    else:
        return pd.DataFrame()
