"""
Statistical hypothesis tests and distribution analysis.

Phase 8 (Restructuring Plan):
This module consolidates statistical testing functions that were previously
scattered across analytics/eval.py and other modules.

Functions:
- test_normality: Shapiro-Wilk and D'Agostino-Pearson normality tests
- test_homogeneity_of_variance: Levene and Bartlett tests
- test_correlation_significance: Pearson/Spearman correlation significance
- compare_distributions: Two-sample distribution comparison (t-test, Mann-Whitney)
- perform_anova: One-way ANOVA for group comparisons
- perform_kruskal_wallis: Non-parametric alternative to ANOVA
- calculate_effect_size: Cohen's d, Glass's delta, Hedge's g

Usage:
    from finance_ml.ml_workflow.eda.statistical_tests import (
        test_normality,
        compare_distributions,
        perform_anova,
    )

    # Test if returns are normally distributed
    normality_result = test_normality(df['returns'])

    # Compare distributions between two groups
    comparison = compare_distributions(group1_data, group2_data)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def test_normality(
    data: Union[pd.Series, np.ndarray],
    method: str = "shapiro",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test for normality of distribution.

    Args:
        data: 1D array-like of numeric values
        method: Test method - 'shapiro' (Shapiro-Wilk), 'dagostino' (D'Agostino-Pearson),
                'anderson' (Anderson-Darling), or 'all'
        alpha: Significance level for hypothesis test

    Returns:
        Dict with test statistic, p-value, and interpretation

    Example:
        >>> result = test_normality(pd.Series([1, 2, 3, 4, 5]))
        >>> print(f"Is normal: {result['is_normal']}")
    """
    # Convert to numpy and remove NaN
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 8:
        return {
            "method": method,
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "message": f"Insufficient data ({len(arr)} samples, need ≥8)",
        }

    results = {}

    if method in ("shapiro", "all"):
        # Shapiro-Wilk test (best for n < 50)
        stat, p = stats.shapiro(arr[:5000])  # Shapiro limited to 5000 samples
        results["shapiro"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "is_normal": bool(p > alpha),
        }

    if method in ("dagostino", "all"):
        # D'Agostino-Pearson test (requires n ≥ 20)
        if len(arr) >= 20:
            stat, p = stats.normaltest(arr)
            results["dagostino"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "is_normal": bool(p > alpha),
            }
        else:
            results["dagostino"] = {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "message": "Requires ≥20 samples",
            }

    if method in ("anderson", "all"):
        # Anderson-Darling test
        result = stats.anderson(arr, dist="norm")
        # Use 5% significance level (index 2)
        critical_value = result.critical_values[2]
        is_normal = result.statistic < critical_value
        results["anderson"] = {
            "statistic": float(result.statistic),
            "critical_value": float(critical_value),
            "is_normal": bool(is_normal),
        }

    # Return single result or all results
    if method == "all":
        return results
    else:
        return results.get(method, {"error": f"Unknown method: {method}"})


def test_homogeneity_of_variance(
    *groups: Union[pd.Series, np.ndarray],
    method: str = "levene",
    center: str = "median",
) -> Dict[str, Any]:
    """Test homogeneity of variance across groups.

    Args:
        *groups: Two or more groups of data
        method: 'levene' (robust) or 'bartlett' (assumes normality)
        center: For Levene's test - 'median' (default), 'mean', or 'trimmed'

    Returns:
        Dict with test statistic, p-value, and interpretation
    """
    # Clean each group
    clean_groups = []
    for g in groups:
        arr = np.asarray(g)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return {
            "method": method,
            "statistic": None,
            "p_value": None,
            "equal_variance": None,
            "message": "Need at least 2 non-empty groups",
        }

    if method == "levene":
        stat, p = stats.levene(*clean_groups, center=center)
    elif method == "bartlett":
        stat, p = stats.bartlett(*clean_groups)
    else:
        return {"error": f"Unknown method: {method}"}

    return {
        "method": method,
        "statistic": float(stat),
        "p_value": float(p),
        "equal_variance": bool(p > 0.05),
    }


def test_correlation_significance(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = "pearson",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test significance of correlation between two variables.

    Args:
        x: First variable
        y: Second variable
        method: 'pearson' or 'spearman'
        alpha: Significance level

    Returns:
        Dict with correlation coefficient, p-value, and significance
    """
    # Clean data - remove NaN from both
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]

    if len(x_clean) < 3:
        return {
            "method": method,
            "correlation": None,
            "p_value": None,
            "is_significant": None,
            "message": f"Insufficient paired data ({len(x_clean)} pairs)",
        }

    if method == "pearson":
        corr, p = stats.pearsonr(x_clean, y_clean)
    elif method == "spearman":
        corr, p = stats.spearmanr(x_clean, y_clean)
    else:
        return {"error": f"Unknown method: {method}"}

    return {
        "method": method,
        "correlation": float(corr),
        "p_value": float(p),
        "is_significant": bool(p < alpha),
        "n_pairs": len(x_clean),
    }


def compare_distributions(
    group1: Union[pd.Series, np.ndarray],
    group2: Union[pd.Series, np.ndarray],
    method: str = "auto",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare two distributions using appropriate statistical test.

    Args:
        group1: First group data
        group2: Second group data
        method: 'auto' (choose based on normality), 't_test', 'mann_whitney', or 'welch'
        alpha: Significance level

    Returns:
        Dict with test results including statistic, p-value, effect size
    """
    # Clean data
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return {
            "method": None,
            "statistic": None,
            "p_value": None,
            "significant": None,
            "message": "Insufficient data in one or both groups",
        }

    # Auto-select method based on normality
    if method == "auto":
        # Check normality of both groups
        norm1 = test_normality(g1, method="shapiro")
        norm2 = test_normality(g2, method="shapiro")

        both_normal = norm1.get("is_normal", False) and norm2.get("is_normal", False)
        method = "welch" if both_normal else "mann_whitney"

    # Perform selected test
    if method in ("t_test", "welch"):
        # Welch's t-test (does not assume equal variances)
        stat, p = stats.ttest_ind(g1, g2, equal_var=(method == "t_test"))
        test_name = "Welch's t-test" if method == "welch" else "Student's t-test"
    elif method == "mann_whitney":
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        return {"error": f"Unknown method: {method}"}

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
        / (len(g1) + len(g2) - 2)
    )
    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

    return {
        "method": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "effect_size_cohens_d": float(cohens_d),
        "group1_mean": float(np.mean(g1)),
        "group2_mean": float(np.mean(g2)),
        "group1_n": len(g1),
        "group2_n": len(g2),
    }


def perform_anova(
    *groups: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform one-way ANOVA to compare means across multiple groups.

    Args:
        *groups: Two or more groups of data
        alpha: Significance level

    Returns:
        Dict with F-statistic, p-value, and effect size (eta-squared)
    """
    # Clean groups
    clean_groups = []
    for g in groups:
        arr = np.asarray(g)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return {
            "method": "one-way ANOVA",
            "f_statistic": None,
            "p_value": None,
            "significant": None,
            "message": "Need at least 2 non-empty groups",
        }

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*clean_groups)

    # Calculate eta-squared (effect size)
    all_data = np.concatenate(clean_groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean_groups)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return {
        "method": "one-way ANOVA",
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "eta_squared": float(eta_squared),
        "n_groups": len(clean_groups),
    }


def perform_kruskal_wallis(
    *groups: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform Kruskal-Wallis H-test (non-parametric ANOVA alternative).

    Args:
        *groups: Two or more groups of data
        alpha: Significance level

    Returns:
        Dict with H-statistic, p-value, and interpretation
    """
    # Clean groups
    clean_groups = []
    for g in groups:
        arr = np.asarray(g)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return {
            "method": "Kruskal-Wallis H-test",
            "h_statistic": None,
            "p_value": None,
            "significant": None,
            "message": "Need at least 2 non-empty groups",
        }

    h_stat, p_value = stats.kruskal(*clean_groups)

    return {
        "method": "Kruskal-Wallis H-test",
        "h_statistic": float(h_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "n_groups": len(clean_groups),
    }


def calculate_effect_size(
    group1: Union[pd.Series, np.ndarray],
    group2: Union[pd.Series, np.ndarray],
    method: str = "cohens_d",
) -> Dict[str, float]:
    """Calculate effect size for difference between two groups.

    Args:
        group1: First group data
        group2: Second group data
        method: 'cohens_d', 'hedges_g', or 'glass_delta'

    Returns:
        Dict with effect size value and interpretation
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return {"effect_size": None, "interpretation": "Insufficient data"}

    mean_diff = np.mean(g1) - np.mean(g2)

    if method == "cohens_d":
        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
            / (len(g1) + len(g2) - 2)
        )
        effect = mean_diff / pooled_std if pooled_std > 0 else 0

    elif method == "hedges_g":
        # Cohen's d with small sample correction
        pooled_std = np.sqrt(
            ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
            / (len(g1) + len(g2) - 2)
        )
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        # Correction factor
        df = len(g1) + len(g2) - 2
        correction = 1 - (3 / (4 * df - 1))
        effect = d * correction

    elif method == "glass_delta":
        # Use control group (group2) standard deviation
        std2 = np.std(g2, ddof=1)
        effect = mean_diff / std2 if std2 > 0 else 0

    else:
        return {"error": f"Unknown method: {method}"}

    # Interpret effect size
    abs_effect = abs(effect)
    if abs_effect < 0.2:
        interpretation = "negligible"
    elif abs_effect < 0.5:
        interpretation = "small"
    elif abs_effect < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "method": method,
        "effect_size": float(effect),
        "interpretation": interpretation,
    }
