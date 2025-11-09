"""
High-level preprocessing pipeline orchestration for finance_ml.ml_workflow.preprocessing.

Part of Phase 9.1 refactor: Provides unified entry points for preprocessing workflows.

This module provides:
- prepare_phase91_data: Complete Phase 9.1 preprocessing pipeline
  (4-step imputation, outlier detection, winsorization, scaling, quality assessment)

Future additions:
- prepare_phase95_data: Regression-specific preprocessing with classification features
- Additional specialized pipelines as needed
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from finance_ml.ml_workflow.preprocessing.imputation import apply_enhanced_imputation_strategy_4step
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
)
from finance_ml.ml_workflow.preprocessing.scaling import scale_features
from finance_ml.ml_workflow.preprocessing.quality import (
    calculate_data_quality_score,
    DataQualityReport,
)

logger = logging.getLogger(__name__)


def prepare_phase91_data(
    df: pd.DataFrame,
    sector_column: str = "sector",
    price_column: str = "last_price",
    n_neighbors: int = 5,
    apply_outlier_detection: bool = True,
    apply_winsorization: bool = True,
    apply_scaling: bool = False,
    scaler_type: str = "robust",
    return_stats: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete Phase 9.1 preprocessing pipeline.

    Applies the full preprocessing workflow:
    1. 4-step imputation strategy (zero, price, KNN, median)
    2. Optional outlier detection (IQR, Z-score, Isolation Forest)
    3. Optional winsorization by sector
    4. Optional feature scaling by sector
    5. Data quality assessment

    Args:
        df: Input DataFrame with financial data
        sector_column: Column name for sector (default: "sector")
        price_column: Column name for price-based imputation (default: "last_price")
        n_neighbors: Number of neighbors for KNN imputation (default: 5)
        apply_outlier_detection: Whether to detect outliers (default: True)
        apply_winsorization: Whether to winsorize extremes (default: True)
        apply_scaling: Whether to scale features (default: False, usually done later)
        scaler_type: Type of scaler if apply_scaling=True (default: "robust")
        return_stats: Whether to return quality stats dict (default: False)

    Returns:
        If return_stats=False: Preprocessed DataFrame
        If return_stats=True: Tuple of (preprocessed DataFrame, stats dict)

    Example:
        >>> # Basic usage
        >>> df_clean = prepare_phase91_data(df)
        >>>
        >>> # With statistics
        >>> df_clean, stats = prepare_phase91_data(df, return_stats=True)
        >>> print(f"Quality score: {stats['quality_report'].overall_score:.2%}")
    """
    logger.info("Starting Phase 9.1 preprocessing pipeline")
    stats = {}

    # Initial quality assessment
    initial_quality = calculate_data_quality_score(df)
    stats["initial_quality"] = initial_quality
    logger.info(f"Initial quality score: {initial_quality.overall_score:.2%}")

    # Step 1: Apply 4-step imputation strategy
    logger.info("Applying 4-step imputation strategy...")
    df_imputed = apply_enhanced_imputation_strategy_4step(
        df=df,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
        price_column=price_column,
    )
    stats["missing_before_imputation"] = df.isnull().sum().sum()
    stats["missing_after_imputation"] = df_imputed.isnull().sum().sum()
    logger.info(
        f"Imputation complete: {stats['missing_before_imputation']} → "
        f"{stats['missing_after_imputation']} missing values"
    )

    # Step 2: Optional outlier detection
    if apply_outlier_detection:
        logger.info("Detecting outliers...")
        numeric_cols = df_imputed.select_dtypes(include=["number"]).columns.tolist()
        financial_cols = [
            c for c in numeric_cols if c not in ["ticker", "isin"] and not c.endswith("_outlier")
        ][
            :20
        ]  # Limit to first 20 financial metrics

        outliers_iqr = detect_outliers_iqr(
            df_imputed, columns=financial_cols, by_sector=True, iqr_multiplier=1.5
        )
        outliers_zscore = detect_outliers_zscore(
            df_imputed, columns=financial_cols, by_sector=True, threshold=3.0
        )
        outliers_iforest = detect_outliers_isolation_forest(
            df_imputed, columns=financial_cols, contamination=0.1
        )

        stats["outliers_iqr"] = outliers_iqr.filter(like="_outlier").sum().sum()
        stats["outliers_zscore"] = outliers_zscore.filter(like="_outlier").sum().sum()
        stats["outliers_iforest"] = outliers_iforest.sum()

        logger.info(
            f"Outliers detected: IQR={stats['outliers_iqr']}, "
            f"Z-score={stats['outliers_zscore']}, IForest={stats['outliers_iforest']}"
        )

        # Merge outlier columns into main DataFrame
        outlier_cols = [c for c in outliers_iqr.columns if c.endswith("_outlier")]
        for col in outlier_cols:
            if col not in df_imputed.columns:
                df_imputed[col] = outliers_iqr[col]

        zscore_cols = [c for c in outliers_zscore.columns if c.endswith("_outlier")]
        for col in zscore_cols:
            if col not in df_imputed.columns:
                df_imputed[col] = outliers_zscore[col]

        if "iforest_outlier" not in df_imputed.columns:
            df_imputed["iforest_outlier"] = outliers_iforest

    # Step 3: Optional winsorization
    if apply_winsorization:
        logger.info("Applying sector-specific winsorization...")
        numeric_cols = df_imputed.select_dtypes(include=["number"]).columns.tolist()
        financial_cols = [
            c for c in numeric_cols if c not in ["ticker", "isin"] and not c.endswith("_outlier")
        ][:20]

        df_imputed = winsorize_by_sector(
            df_imputed,
            columns=financial_cols,
            lower_percentile=0.01,
            upper_percentile=0.99,
            by_sector=True,
        )
        logger.info("Winsorization complete")

    # Step 4: Optional scaling
    if apply_scaling:
        logger.info(f"Applying feature scaling ({scaler_type})...")
        numeric_cols = df_imputed.select_dtypes(include=["number"]).columns.tolist()
        scaling_cols = [
            c
            for c in numeric_cols
            if c not in ["ticker", "isin", "price_target", "price_target_median", "last_price"]
            and not c.endswith("_outlier")
        ][:30]

        df_imputed = scale_features(
            df_imputed, columns=scaling_cols, scaler_type=scaler_type, by_sector=True
        )
        logger.info("Scaling complete")

    # Final quality assessment
    final_quality = calculate_data_quality_score(df_imputed)
    stats["final_quality"] = final_quality
    stats["quality_improvement"] = final_quality.overall_score - initial_quality.overall_score
    logger.info(
        f"Final quality score: {final_quality.overall_score:.2%} "
        f"(improvement: {stats['quality_improvement']:+.2%})"
    )

    logger.info("Phase 9.1 preprocessing pipeline complete")

    if return_stats:
        return df_imputed, stats
    return df_imputed
