"""
Data quality assessment module for finance_ml.ml_workflow.preprocessing.

Part of Phase 9.1 refactor: Extracted from advanced_preprocessing.py.

This module provides:
- DataQualityReport: Container for quality metrics
- calculate_data_quality_score: Comprehensive data quality assessment

Functions ensure data is ready for modeling by checking completeness,
consistency, and validity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality metrics."""

    completeness_score: float  # % of non-null values
    consistency_score: float  # % of values within expected ranges
    validity_score: float  # % of valid data types and formats
    overall_score: float  # Weighted average of above
    issues: List[str]  # List of detected issues
    metrics: Dict[str, Any]  # Detailed metrics

    def __str__(self) -> str:
        """String representation of quality report."""
        return (
            f"Data Quality Report:\n"
            f"  Overall Score: {self.overall_score:.2%}\n"
            f"  Completeness: {self.completeness_score:.2%}\n"
            f"  Consistency: {self.consistency_score:.2%}\n"
            f"  Validity: {self.validity_score:.2%}\n"
            f"  Issues: {len(self.issues)}"
        )


def calculate_data_quality_score(df: pd.DataFrame) -> DataQualityReport:
    """Calculate comprehensive data quality metrics.

    Assesses three dimensions of data quality:
    1. Completeness: Percentage of non-null values
    2. Consistency: Values within expected ranges, no infinite values
    3. Validity: Expected columns present, correct data types

    Args:
        df: Input DataFrame to assess

    Returns:
        DataQualityReport with detailed quality metrics and issues

    Example:
        >>> report = calculate_data_quality_score(stocks_df)
        >>> print(f"Overall quality: {report.overall_score:.2%}")
        >>> if report.issues:
        ...     print("Issues found:", report.issues)
    """
    issues = []
    metrics = {}

    # 1. Completeness: % of non-null values
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0

    metrics["total_cells"] = total_cells
    metrics["non_null_cells"] = non_null_cells
    metrics["missing_cells"] = total_cells - non_null_cells

    if completeness < 0.8:
        issues.append(f"Low completeness: {completeness:.2%} (expected >80%)")

    # 2. Consistency: Check numeric ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    consistency_checks = 0
    consistency_passes = 0

    for col in numeric_cols:
        if col in df.columns and not df[col].isna().all():
            consistency_checks += 1

            # Check for infinite values
            if np.isinf(df[col]).any():
                issues.append(f"Column '{col}' contains infinite values")
            else:
                consistency_passes += 1

            # Check for negative values in typically positive columns
            if any(x in col.lower() for x in ["market_cap", "price", "revenue", "assets"]):
                if (df[col] < 0).any():
                    issues.append(f"Column '{col}' contains unexpected negative values")

    consistency = consistency_passes / consistency_checks if consistency_checks > 0 else 1.0
    metrics["consistency_checks"] = consistency_checks
    metrics["consistency_passes"] = consistency_passes

    # 3. Validity: Check data types and formats
    validity_checks = 0
    validity_passes = 0

    # Check if expected columns exist
    expected_cols = ["ticker", "sector", "region"]
    for col in expected_cols:
        validity_checks += 1
        if col in df.columns:
            validity_passes += 1
        else:
            issues.append(f"Missing expected column: '{col}'")

    validity = validity_passes / validity_checks if validity_checks > 0 else 0
    metrics["validity_checks"] = validity_checks
    metrics["validity_passes"] = validity_passes

    # Overall score (weighted average)
    overall = completeness * 0.4 + consistency * 0.3 + validity * 0.3

    report = DataQualityReport(
        completeness_score=completeness,
        consistency_score=consistency,
        validity_score=validity,
        overall_score=overall,
        issues=issues,
        metrics=metrics,
    )

    logger.info(f"Data quality assessment complete: {overall:.2%} overall score")
    return report
