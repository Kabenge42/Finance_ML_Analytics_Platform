"""
Unified ETL (Extract, Transform, Load) Pipeline for Financial Data.

This module provides a simplified, orchestrated pipeline that consolidates
data loading, transformation, and validation operations from:
- finance_ml.ml_workflow.preprocessing.data (extraction & basic transforms)
- finance_ml.ml_workflow.data.schema (schema validation)
- finance_ml.ml_workflow.preprocessing.transforms (log transforms)

Architecture:
    Extract → Transform → Load

Usage:
    Basic usage:
        >>> from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline
        >>> df = run_etl_pipeline(source='csv', data_dir='data/')

    Advanced usage with configuration:
        >>> config = ETLConfig(
        ...     apply_log_transforms=True,
        ...     validate_quality=True,
        ...     drop_invalid_rows=True
        ... )
        >>> df, metrics = run_etl_pipeline(
        ...     source='db',
        ...     db_url='postgresql://...',
        ...     config=config,
        ...     return_metrics=True
        ... )

Aligned with code_guidelines.md ETL best practices.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    list_categorical_cols,
    list_numeric_feature_cols,
)

# Import feature engineering API (Section 9.3)
from finance_ml.ml_workflow.features.api import build_features

# Import feature selection API (Section 9.3 Task 1)
from finance_ml.ml_workflow.features.selection import select_features_auto
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    COUNT_COLUMNS,
    MARKET_VALUE_COLUMNS,
    PERCENTAGE_COLUMNS,
    PRICE_COLUMNS,
    RATIO_COLUMNS,
    classify_columns,
    get_log_transform_columns,
    get_scalable_columns,
    get_winsorizable_columns,
)

# Import from existing modules (no code duplication)
from finance_ml.ml_workflow.preprocessing.data import (
    load_from_all_stocks,
    load_from_csv,
    load_from_db,
    normalize_columns,
    perform_early_pipeline_validation,
    sanitize_dataframe_with_logging,
    validate_financial_data_quality,
    validate_schema,
)
from finance_ml.ml_workflow.preprocessing.dtypes import (
    detect_and_cast_dtypes,
    to_jsonable,
)

# Import financial metrics functions for unified ETL API
from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
    CONDITIONAL_METRICS,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_profitability_metrics,
    compute_sector_specific_ratios,
    compute_target_vs_price_metrics,
    compute_valuation_metrics,
    generate_data_quality_alerts,
    generate_metrics_dashboard,
    handle_sector_specific_metrics,
    # Post-metrics imputation utilities (schema-aware, sector-aware)
    impute_computed_metrics,
)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    apply_median_imputation,
    validate_imputation_completeness,
)
from finance_ml.ml_workflow.preprocessing.scaling import (
    scale_features,
)
from finance_ml.ml_workflow.preprocessing.transforms import (
    apply_log_transforms,
    get_skewness,
)

logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    """
    Configuration for ETL pipeline stages.

    Attributes:
        normalize_columns: Apply column name normalization (default: True)
        apply_dtype_casting: Apply schema-aware dtype casting from COLUMN_SCHEMA (default: True)
        track_dtype_diagnostics: Track detailed dtype coercion diagnostics in metrics (default: True)
        validate_schema: Validate against COLUMN_SCHEMA (default: True)
        require_target: Require price_target column in validation (default: False)
        sanitize_data: Apply data sanitization (inf, nan, extremes) (default: True)
        apply_log_transforms: Apply log transforms to skewed columns (default: False)
        log_transform_method: Method for log transforms ('log1p' or 'signed_log')
        validate_quality: Perform data quality checks (default: True)
        validate_pipeline: Perform early pipeline validation (default: True)
        drop_invalid_rows: Drop rows with missing critical fields (default: True)
        limit: Row limit for data loading (default: None = no limit)
    """

    normalize_columns: bool = True
    apply_dtype_casting: bool = True  # Apply schema-aware dtype casting (default: True)
    track_dtype_diagnostics: bool = (
        True  # Track dtype coercion diagnostics (default: True)
    )
    validate_schema: bool = True
    require_target: bool = False
    sanitize_data: bool = True
    apply_log_transforms: bool = False
    log_transform_method: Literal["log1p", "signed_log"] = "log1p"
    validate_quality: bool = True
    validate_pipeline: bool = True
    drop_invalid_rows: bool = True
    limit: Optional[int] = None

    # Imputation options
    apply_imputation: bool = True
    imputation_strategy: Literal["6step", "4step", "median_only"] = "6step"
    knn_neighbors: int = 5
    imputation_sector_column: str = "sector"
    imputation_price_column: str = "last_price"
    handle_categorical_imputation: bool = True
    handle_datetime_imputation: bool = True

    # Scaling options
    apply_scaling: bool = False  # Default OFF (backward compatible)
    scaler_type: Literal["robust", "standard", "minmax"] = "robust"
    scale_by_sector: bool = True
    scaling_columns: Optional[List[str]] = None
    exclude_price_columns_from_scaling: bool = True  # CRITICAL safety default

    # Advanced options
    log_transform_columns: Optional[List[str]] = None
    custom_validators: List[Any] = field(default_factory=list)

    # Financial metrics computation flags (default OFF for backward compatibility)
    compute_valuation_metrics: bool = False
    compute_profitability_metrics: bool = False
    compute_growth_metrics: bool = False
    compute_leverage_metrics: bool = False
    compute_target_vs_price: bool = False
    handle_sector_specific_metrics: bool = False

    # Quality reporting options
    generate_quality_alerts: bool = False
    generate_metrics_dashboard: bool = False
    output_subdir: str = "financial_metrics"

    # Semantic-aware transformation flags (Section 8.5)
    use_semantic_column_classification: bool = True
    preserve_price_columns: bool = True  # Never transform price columns
    log_transform_market_values: bool = True  # Apply log-transforms to skewed columns
    exclude_ratios_from_winsorization: bool = True  # Ratios are pre-normalized
    exclude_percentages_from_winsorization: bool = True  # Percentages are bounded
    exclude_counts_from_scaling: bool = False  # Optionally exclude discrete counts

    # Feature engineering integration (Section 9.3)
    apply_feature_engineering: bool = False  # Default OFF for backward compatibility
    feature_preset: str = (
        "standard"  # Options: "basic", "momentum", "quality", "comprehensive"
    )
    feature_categories: Optional[List[str]] = None  # Specific categories to engineer

    # Feature selection integration (Section 9.3 Task 1)
    apply_feature_selection: bool = False  # Default OFF for backward compatibility
    feature_selection_method: Literal["mutual_info", "correlation", "both"] = (
        "mutual_info"
    )
    importance_threshold: float = 0.01  # Min importance score to keep feature
    correlation_threshold: float = 0.95  # Max correlation before deduplication
    feature_selection_categories: Optional[List[str]] = None  # Category-based selection


@dataclass
class ETLMetrics:
    """
    Metrics collected during ETL pipeline execution.

    Attributes:
        source_type: Data source type (csv, db, all_stocks)
        extract_time_sec: Time spent in Extract stage
        transform_time_sec: Time spent in Transform stage
        load_time_sec: Time spent in Load stage
        total_time_sec: Total pipeline execution time
        rows_input: Number of rows after extraction
        rows_output: Number of rows after transformation
        columns_input: Number of columns after extraction
        columns_output: Number of columns after transformation
        quality_score: Data quality score (0-1)
        validation_score: Pipeline validation score (0-1)
        stages_executed: List of executed stages
        warnings: List of warning messages
        errors: List of error messages
    """

    source_type: str
    extract_time_sec: float = 0.0
    transform_time_sec: float = 0.0
    load_time_sec: float = 0.0
    total_time_sec: float = 0.0
    rows_input: int = 0
    rows_output: int = 0
    columns_input: int = 0
    columns_output: int = 0
    quality_score: float = 1.0
    validation_score: float = 1.0
    stages_executed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Dtype casting metrics
    dtype_casting_applied: bool = True
    dtype_diagnostics: Optional[Dict[str, Any]] = None
    dtype_coercion_warnings: int = 0
    dtype_unknown_columns: int = 0

    # Imputation metrics
    imputation_strategy: Optional[str] = None
    missing_values_before_imputation: int = 0
    missing_values_after_imputation: int = 0
    imputation_completeness: bool = False
    date_columns_ready: bool = False

    # Scaling metrics
    scaling_applied: bool = True
    scaler_type: Optional[str] = None
    scaled_columns_count: int = 0
    price_columns_protected: bool = True  # Always True with default settings

    # Financial metrics tracking
    valuation_metrics_added: int = 0
    profitability_metrics_added: int = 0
    growth_metrics_added: int = 0
    leverage_metrics_added: int = 0
    target_vs_price_metrics_added: int = 0
    sector_specific_metrics_added: int = 0

    # Semantic transformation metrics (Section 8.5)
    semantic_classification_applied: bool = False
    price_columns_count: int = 0
    market_value_columns_count: int = 0
    ratio_columns_count: int = 0
    percentage_columns_count: int = 0
    count_columns_count: int = 0
    log_transformed_columns: int = 0

    # Feature engineering metrics (Section 9.3)
    feature_engineering_applied: bool = False
    feature_preset_used: str = ""
    features_added: int = 0
    feature_categories_applied: List[str] = field(default_factory=list)

    # Feature selection metrics (Section 9.3 Task 1)
    feature_selection_applied: bool = False
    features_before_selection: int = 0
    features_after_selection: int = 0
    features_removed_by_selection: int = 0

    # Business rule validation metrics (Priority 1)
    business_rule_violations: int = 0  # Negative values in non-negative columns
    log_transforms_skipped: int = 0  # Log transforms skipped due to negative values

    # Schema validation metrics (code_guidelines.md v1.11)
    schema_alignment_score: float = 1.0  # Schema alignment quality [0.0-1.0]
    unknown_columns_count: int = 0  # Columns in df but not in COLUMN_SCHEMA
    missing_expected_columns_count: int = 0  # Expected columns not in df
    dtype_mismatches_count: int = 0  # Columns with dtype mismatches

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "source_type": self.source_type,
            "timings": {
                "extract_sec": self.extract_time_sec,
                "transform_sec": self.transform_time_sec,
                "load_sec": self.load_time_sec,
                "total_sec": self.total_time_sec,
            },
            "data_shape": {
                "rows_input": self.rows_input,
                "rows_output": self.rows_output,
                "columns_input": self.columns_input,
                "columns_output": self.columns_output,
            },
            "quality": {
                "quality_score": self.quality_score,
                "validation_score": self.validation_score,
            },
            "dtype_casting": {
                "applied": self.dtype_casting_applied,
                "coercion_warnings": self.dtype_coercion_warnings,
                "unknown_columns": self.dtype_unknown_columns,
            },
            "imputation": {
                "strategy": self.imputation_strategy,
                "missing_before": self.missing_values_before_imputation,
                "missing_after": self.missing_values_after_imputation,
                "completeness": self.imputation_completeness,
                "dates_ready": self.date_columns_ready,
            },
            "scaling": {
                "applied": self.scaling_applied,
                "scaler_type": self.scaler_type,
                "scaled_columns": self.scaled_columns_count,
                "price_columns_protected": self.price_columns_protected,
            },
            "financial_metrics": {
                "valuation_added": self.valuation_metrics_added,
                "profitability_added": self.profitability_metrics_added,
                "growth_added": self.growth_metrics_added,
                "leverage_added": self.leverage_metrics_added,
                "target_vs_price_added": self.target_vs_price_metrics_added,
                "sector_specific_added": self.sector_specific_metrics_added,
            },
            "semantic_transformations": {
                "applied": self.semantic_classification_applied,
                "price_columns": self.price_columns_count,
                "market_value_columns": self.market_value_columns_count,
                "ratio_columns": self.ratio_columns_count,
                "percentage_columns": self.percentage_columns_count,
                "count_columns": self.count_columns_count,
                "log_transformed": self.log_transformed_columns,
            },
            "feature_engineering": {
                "applied": self.feature_engineering_applied,
                "preset": self.feature_preset_used,
                "features_added": self.features_added,
                "categories_applied": self.feature_categories_applied,
            },
            "feature_selection": {
                "applied": self.feature_selection_applied,
                "features_before": self.features_before_selection,
                "features_after": self.features_after_selection,
                "features_removed": self.features_removed_by_selection,
            },
            "business_rules": {
                "violations": self.business_rule_violations,
                "log_transforms_skipped": self.log_transforms_skipped,
            },
            "schema_validation": {
                "alignment_score": self.schema_alignment_score,
                "unknown_columns": self.unknown_columns_count,
                "missing_expected_columns": self.missing_expected_columns_count,
                "dtype_mismatches": self.dtype_mismatches_count,
            },
            "stages_executed": self.stages_executed,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        dtype_info = ""
        if self.dtype_casting_applied:
            warning_icon = "⚠" if self.dtype_coercion_warnings > 0 else "✓"
            dtype_info = (
                f"\n  Dtype Casting: Applied "
                f"({self.dtype_coercion_warnings} coercion warnings, "
                f"{self.dtype_unknown_columns} unknown columns) {warning_icon}"
            )

        imputation_info = ""
        if self.imputation_strategy:
            status_icon = "✓" if self.imputation_completeness else "✗"
            imputation_info = (
                f"\n  Imputation: {self.imputation_strategy} "
                f"({self.missing_values_before_imputation} → "
                f"{self.missing_values_after_imputation} missing) "
                f"{status_icon}"
            )

        scaling_info = ""
        if self.scaling_applied:
            protection_icon = "✓" if self.price_columns_protected else "✗ WARNING"
            scaling_info = (
                f"\n  Scaling: {self.scaler_type} "
                f"({self.scaled_columns_count} columns) "
                f"Price Protected: {protection_icon}"
            )

        # Financial metrics info
        total_metrics = (
            self.valuation_metrics_added
            + self.profitability_metrics_added
            + self.growth_metrics_added
            + self.leverage_metrics_added
            + self.target_vs_price_metrics_added
            + self.sector_specific_metrics_added
        )
        financial_metrics_info = ""
        if total_metrics > 0:
            financial_metrics_info = (
                f"\n  Financial Metrics: {total_metrics} added "
                f"(valuation: {self.valuation_metrics_added}, "
                f"profitability: {self.profitability_metrics_added}, "
                f"growth: {self.growth_metrics_added}, "
                f"leverage: {self.leverage_metrics_added})"
            )

        # Semantic transformation info
        semantic_info = ""
        if self.semantic_classification_applied:
            semantic_info = (
                f"\n  Semantic Classification: ✓ "
                f"(Price Columns: {self.price_columns_count}, "
                f"Market Value: {self.market_value_columns_count}, "
                f"Ratios: {self.ratio_columns_count}, "
                f"Log-Transformed: {self.log_transformed_columns})"
            )

        # Feature engineering info
        feature_engineering_info = ""
        if self.feature_engineering_applied:
            feature_engineering_info = (
                f"\n  Feature Engineering: {self.feature_preset_used} "
                f"({self.features_added} features added)"
            )

        # Feature selection info
        feature_selection_info = ""
        if self.feature_selection_applied:
            reduction_pct = (
                100
                * self.features_removed_by_selection
                / self.features_before_selection
                if self.features_before_selection > 0
                else 0
            )
            feature_selection_info = (
                f"\n  Feature Selection: {self.features_before_selection} → "
                f"{self.features_after_selection} features "
                f"(removed {self.features_removed_by_selection}, {reduction_pct:.1f}% reduction)"
            )

        # Business rule validation info (Priority 1 Fix)
        business_rules_info = ""
        if self.business_rule_violations > 0 or self.log_transforms_skipped > 0:
            status_icon = "⚠" if self.business_rule_violations > 0 else "✓"
            business_rules_info = (
                f"\n  Business Rules: {status_icon} "
                f"({self.business_rule_violations} negative value violations sanitized, "
                f"{self.log_transforms_skipped} log-transforms skipped)"
            )

        # Schema validation info (code_guidelines.md v1.11)
        schema_validation_info = ""
        if (
            self.schema_alignment_score < 1.0
            or self.unknown_columns_count > 0
            or self.missing_expected_columns_count > 0
        ):
            status_icon = "✓" if self.schema_alignment_score >= 0.95 else "⚠"
            schema_validation_info = (
                f"\n  Schema Validation: {status_icon} "
                f"(alignment: {self.schema_alignment_score:.2%}, "
                f"unknown: {self.unknown_columns_count}, "
                f"missing: {self.missing_expected_columns_count}, "
                f"dtype mismatches: {self.dtype_mismatches_count})"
            )

        return (
            f"ETL Pipeline Summary:\n"
            f"  Source: {self.source_type}\n"
            f"  Duration: {self.total_time_sec:.2f}s "
            f"(extract: {self.extract_time_sec:.2f}s, "
            f"transform: {self.transform_time_sec:.2f}s, "
            f"load: {self.load_time_sec:.2f}s)\n"
            f"  Data: {self.rows_input} → {self.rows_output} rows, "
            f"{self.columns_input} → {self.columns_output} columns"
            f"{dtype_info}"
            f"{imputation_info}"
            f"{scaling_info}"
            f"{financial_metrics_info}"
            f"{semantic_info}"
            f"{feature_engineering_info}"
            f"{feature_selection_info}"
            f"{business_rules_info}"
            f"{schema_validation_info}\n"
            f"  Quality: {self.quality_score:.3f}, "
            f"Validation: {self.validation_score:.3f}\n"
            f"  Stages: {', '.join(self.stages_executed)}\n"
            f"  Warnings: {len(self.warnings)}, Errors: {len(self.errors)}"
        )


def validate_etl_output(
    df: pd.DataFrame,
    phase: str,
    expected_min_cols: int = 0,
    expected_min_rows: int = 0,
    check_nan: bool = True,
    check_inf: bool = True,
    critical_columns: Optional[List[str]] = None,
    raise_on_failure: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate DataFrame between ETL/feature engineering phases.

    This function provides intermediate validation checkpoints to ensure
    data quality is maintained throughout the pipeline. It checks:
    - Minimum expected column/row counts
    - NaN values presence and distribution
    - Infinite values in numeric columns
    - Critical column presence

    Args:
        df: DataFrame to validate
        phase: Name of the pipeline phase (for logging)
        expected_min_cols: Minimum expected column count (default: 0 = no check)
        expected_min_rows: Minimum expected row count (default: 0 = no check)
        check_nan: Check for NaN values (default: True)
        check_inf: Check for infinite values (default: True)
        critical_columns: List of columns that must exist (default: None)
        raise_on_failure: Raise ValueError on validation failure (default: False)

    Returns:
        Tuple of (is_valid, validation_report) where validation_report contains:
            - phase: Phase name
            - shape: DataFrame shape
            - passed: List of passed checks
            - failed: List of failed checks
            - nan_count: Total NaN values
            - inf_count: Total infinite values
            - missing_columns: List of missing critical columns

    Example:
        >>> # After ETL extraction
        >>> is_valid, report = validate_etl_output(
        ...     df, phase="phase_9.1_etl",
        ...     expected_min_cols=100,
        ...     critical_columns=["ticker", "sector", "last_price"]
        ... )
        >>> if not is_valid:
        ...     logger.warning(f"Validation issues: {report['failed']}")
    """
    report: Dict[str, Any] = {
        "phase": phase,
        "shape": df.shape,
        "passed": [],
        "failed": [],
        "nan_count": 0,
        "inf_count": 0,
        "missing_columns": [],
        "columns_with_nan": [],
        "columns_with_inf": [],
    }

    # Check minimum columns
    if expected_min_cols > 0:
        if df.shape[1] >= expected_min_cols:
            report["passed"].append(f"column_count >= {expected_min_cols}")
        else:
            report["failed"].append(
                f"column_count: {df.shape[1]} < expected {expected_min_cols}"
            )

    # Check minimum rows
    if expected_min_rows > 0:
        if df.shape[0] >= expected_min_rows:
            report["passed"].append(f"row_count >= {expected_min_rows}")
        else:
            report["failed"].append(
                f"row_count: {df.shape[0]} < expected {expected_min_rows}"
            )

    # Check critical columns
    if critical_columns:
        missing = [col for col in critical_columns if col not in df.columns]
        if missing:
            report["missing_columns"] = missing
            report["failed"].append(f"missing_critical_columns: {missing}")
        else:
            report["passed"].append("all_critical_columns_present")

    # Check NaN values
    if check_nan:
        nan_counts = df.isna().sum()
        total_nan = int(nan_counts.sum())
        report["nan_count"] = total_nan
        if total_nan > 0:
            cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
            report["columns_with_nan"] = cols_with_nan[:20]  # Limit to 20
            # Warning but not failure - NaN is often expected
            logger.debug(
                f"{phase}: {total_nan} NaN values in {len(cols_with_nan)} columns"
            )
        report["passed"].append("nan_check_completed")

    # Check infinite values
    if check_inf:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            try:
                inf_mask = np.isinf(numeric_df.to_numpy(dtype=np.float64))
                total_inf = int(inf_mask.sum())
                report["inf_count"] = total_inf
                if total_inf > 0:
                    inf_cols = numeric_df.columns[inf_mask.any(axis=0)].tolist()
                    report["columns_with_inf"] = inf_cols[:20]
                    report["failed"].append(
                        f"infinite_values: {total_inf} in columns {inf_cols[:5]}"
                    )
                else:
                    report["passed"].append("no_infinite_values")
            except (ValueError, TypeError):
                report["passed"].append("inf_check_skipped_non_numeric")
        else:
            report["passed"].append("inf_check_skipped_no_numeric_cols")

    # Determine overall validity
    is_valid = len(report["failed"]) == 0

    # Log summary
    status_icon = "✓" if is_valid else "✗"
    logger.info(
        f"{status_icon} {phase} validation: shape={df.shape}, "
        f"passed={len(report['passed'])}, failed={len(report['failed'])}"
    )

    if report["failed"]:
        logger.warning(f"{phase} validation failures: {report['failed']}")

    if raise_on_failure and not is_valid:
        raise ValueError(f"{phase} validation failed: {report['failed']}")

    return is_valid, report


def report_column_changes(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    operation: str,
) -> Dict[str, Any]:
    """
    Report column changes between two DataFrames.

    Useful for tracking feature engineering additions/removals.

    Args:
        df_before: DataFrame before operation
        df_after: DataFrame after operation
        operation: Name of the operation (for logging)

    Returns:
        Dict with 'added', 'removed', 'unchanged' column lists
    """
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)

    added = list(cols_after - cols_before)
    removed = list(cols_before - cols_after)
    unchanged = list(cols_before & cols_after)

    logger.info(
        f"{operation}: +{len(added)} added, -{len(removed)} removed, "
        f"={len(unchanged)} unchanged columns"
    )

    if added:
        logger.debug(f"  Added: {added[:10]}{'...' if len(added) > 10 else ''}")
    if removed:
        logger.debug(f"  Removed: {removed[:10]}{'...' if len(removed) > 10 else ''}")

    return {
        "operation": operation,
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
        "count_before": len(cols_before),
        "count_after": len(cols_after),
    }


class ETLPipeline:
    """
    Orchestrates Extract, Transform, Load operations for financial data.

    This class provides a unified interface for loading, transforming, and
    validating financial data from various sources.

    Example:
        >>> pipeline = ETLPipeline(config=ETLConfig())
        >>> df = pipeline.extract_from_csv('data/')
        >>> df = pipeline.transform(df)
        >>> result = pipeline.load(df)
    """

    def __init__(self, config: Optional[ETLConfig] = None):
        """
        Initialize ETL pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or ETLConfig()
        self.metrics = None

    def extract_from_csv(
        self,
        data_dir: Path | str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract data from CSV files.

        Loads from regional CSV files:
        - screening_us.csv
        - screening_eu.csv
        - screening_apac.csv
        - screening_rotw.csv

        Args:
            data_dir: Directory containing CSV files
            limit: Optional row limit

        Returns:
            Combined DataFrame from all regions

        Raises:
            FileNotFoundError: If no CSV files found
        """
        logger.info(f"Extracting data from CSV: {data_dir}")
        limit = limit or self.config.limit
        df = load_from_csv(Path(data_dir), limit=limit)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from CSV")
        return df

    def extract_from_db(
        self,
        db_url: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract data from PostgreSQL database (equities table).

        Args:
            db_url: SQLAlchemy database URL
            limit: Optional row limit

        Returns:
            DataFrame from database

        Raises:
            ImportError: If SQLAlchemy not available
        """
        logger.info(f"Extracting data from database: {db_url}")
        limit = limit or self.config.limit
        df = load_from_db(db_url, limit=limit)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from DB")
        return df

    def extract_from_all_stocks(
        self,
        db_url: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract data from unified all_stocks table.

        Recommended approach for database loading as it uses the unified
        all_stocks table created by all_stocks.sql.

        Args:
            db_url: SQLAlchemy database URL
            limit: Optional row limit

        Returns:
            DataFrame from all_stocks table

        Raises:
            ImportError: If SQLAlchemy not available
        """
        logger.info(f"Extracting data from all_stocks table: {db_url}")
        limit = limit or self.config.limit
        df = load_from_all_stocks(db_url, limit=limit)
        logger.info(
            f"Extracted {len(df)} rows, {len(df.columns)} columns from all_stocks"
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformation stages to DataFrame.

        Stages (configurable via ETLConfig):
        1. Column normalization
        1.5. Dtype casting (schema-aware, CSV-critical)
        2. Schema validation
        3. Drop invalid rows
        4. Data sanitization
        5. Imputation (6-step strategy)
        6. Log transforms (optional)
        7. Feature scaling (optional)
        8. Financial metrics computation (optional)

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If validation fails and errors are critical
        """
        logger.info("Starting transformation pipeline")
        result = df.copy()

        # Stage 1: Normalize columns
        if self.config.normalize_columns:
            logger.info("Stage 1: Normalizing column names")
            result = normalize_columns(result, preserve_schema=True)

        # Stage 1.5: Apply dtype casting (NEW - critical for CSV data)
        if self.config.apply_dtype_casting:
            logger.info("Stage 1.5: Applying schema-aware dtype casting")
            try:
                result, dtype_diagnostics = detect_and_cast_dtypes(
                    result, schema=COLUMN_SCHEMA
                )

                # Track metrics
                if self.metrics:
                    self.metrics.dtype_casting_applied = True
                    self.metrics.dtype_coercion_warnings = sum(
                        dtype_diagnostics.get("coercion_warnings", {}).values()
                    )
                    self.metrics.dtype_unknown_columns = len(
                        dtype_diagnostics.get("unknown_columns", [])
                    )

                    # Store diagnostics (convert numpy types to JSON-serializable)
                    if self.config.track_dtype_diagnostics:
                        self.metrics.dtype_diagnostics = to_jsonable(dtype_diagnostics)

                    # Add warnings for data quality issues
                    if self.metrics.dtype_coercion_warnings > 0:
                        self.metrics.warnings.append(
                            f"Dtype casting: {self.metrics.dtype_coercion_warnings} values coerced to NaN "
                            f"(check dtype_diagnostics for details)"
                        )
                    if self.metrics.dtype_unknown_columns > 0:
                        unknown_cols = dtype_diagnostics.get("unknown_columns", [])
                        self.metrics.warnings.append(
                            f"Dtype casting: {self.metrics.dtype_unknown_columns} columns not in schema: "
                            f"{', '.join(unknown_cols[:5])}{'...' if len(unknown_cols) > 5 else ''}"
                        )

                    self.metrics.stages_executed.append("dtype_casting")

                logger.info(
                    f"Dtype casting complete: {len(dtype_diagnostics.get('cast_applied', {}))} columns cast, "
                    f"{self.metrics.dtype_coercion_warnings if self.metrics else 0} coercion warnings, "
                    f"{self.metrics.dtype_unknown_columns if self.metrics else 0} unknown columns"
                )

            except Exception as e:
                error_msg = f"Dtype casting failed: {e}"
                logger.error(error_msg)
                if self.metrics:
                    self.metrics.errors.append(error_msg)
                # Don't raise - allow pipeline to continue without dtype casting
                logger.warning("Pipeline continuing without dtype casting")

        # Stage 1.6: Apply semantic column classification (Section 8.5)
        if self.config.use_semantic_column_classification:
            logger.info("Stage 1.6: Applying semantic column classification")
            result = self._apply_semantic_transformations(result)
            if self.metrics:
                self.metrics.stages_executed.append("semantic_classification")

        # Stage 2: Validate schema
        if self.config.validate_schema:
            logger.info("Stage 2: Validating schema")
            is_valid, errors = validate_schema(
                result, require_target=self.config.require_target
            )
            if not is_valid:
                error_msg = f"Schema validation failed: {', '.join(errors)}"
                logger.error(error_msg)
                if self.metrics:
                    self.metrics.errors.extend(errors)
                # Don't raise, just log - allow pipeline to continue with warnings
                if self.metrics:
                    self.metrics.warnings.append("Schema validation issues detected")

        # Stage 3: Drop invalid rows
        if self.config.drop_invalid_rows:
            logger.info("Stage 3: Dropping rows with missing critical fields")
            initial_rows = len(result)
            for col in ["ticker", "sector", "last_price"]:
                if col in result.columns:
                    result = result[~result[col].isna()]
            dropped_rows = initial_rows - len(result)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing critical fields")

        # Stage 4: Sanitize data
        if self.config.sanitize_data:
            logger.info("Stage 4: Sanitizing data (inf, nan, extremes)")
            result, sanitize_stats = sanitize_dataframe_with_logging(
                result, return_stats=True
            )

            # Track business rule violations in metrics (Priority 1 Fix)
            if self.metrics and sanitize_stats:
                self.metrics.business_rule_violations = sanitize_stats.get(
                    "business_rule_violations", 0
                )
                if self.metrics.business_rule_violations > 0:
                    self.metrics.warnings.append(
                        f"Business rules: {self.metrics.business_rule_violations} negative values "
                        f"sanitized in non-negative columns (converted to NaN for imputation)"
                    )

        # Stage 5: Apply imputation strategy
        if self.config.apply_imputation:
            logger.info(
                f"Stage 5: Applying {self.config.imputation_strategy} imputation strategy"
            )

            # Track missing values before imputation
            missing_before = result.isna().sum().sum()
            if self.metrics:
                self.metrics.missing_values_before_imputation = int(missing_before)
                self.metrics.imputation_strategy = self.config.imputation_strategy

            if self.config.imputation_strategy == "6step":
                result = apply_enhanced_imputation_strategy_6step(
                    result,
                    sector_column="sector",
                    n_neighbors=self.config.knn_neighbors,
                    price_column="last_price",
                    handle_categoricals=self.config.handle_categorical_imputation,
                    handle_dates=self.config.handle_datetime_imputation,
                )
            elif self.config.imputation_strategy == "4step":
                # Backward compatibility - numeric only
                result = apply_enhanced_imputation_strategy_6step(
                    result,
                    sector_column="sector",
                    n_neighbors=self.config.knn_neighbors,
                    price_column="last_price",
                    handle_categoricals=False,
                    handle_dates=False,
                )
            elif self.config.imputation_strategy == "median_only":
                # Simple median fallback
                result = apply_median_imputation(result)

            # Track missing values after imputation and validate completeness
            missing_after = result.isna().sum().sum()
            validation = validate_imputation_completeness(result)

            if self.metrics:
                self.metrics.missing_values_after_imputation = int(missing_after)
                self.metrics.imputation_completeness = validation["is_complete"]
                self.metrics.date_columns_ready = validation[
                    "ready_for_temporal_features"
                ]
                self.metrics.quality_score = min(
                    self.metrics.quality_score,
                    1.0 if validation["is_complete"] else 0.8,
                )
                if not validation["ready_for_temporal_features"]:
                    self.metrics.warnings.append(
                        "Date columns not properly formatted for temporal feature engineering"
                    )

            logger.info(
                f"After imputation: {missing_after} missing values remain "
                f"(reduced from {missing_before})"
            )

        # Stage 6: Apply log transforms (optional)
        if self.config.apply_log_transforms:
            logger.info(
                f"Stage 6: Applying log transforms (method: {self.config.log_transform_method})"
            )
            result = apply_log_transforms(
                result,
                columns=self.config.log_transform_columns,
                method=self.config.log_transform_method,
            )

        # Stage 7: Apply feature scaling (NEW)
        if self.config.apply_scaling:
            logger.info(f"Stage 7: Applying {self.config.scaler_type} scaling")

            # Determine columns to scale
            if self.config.scaling_columns:
                # User-specified columns
                columns_to_scale = self.config.scaling_columns
                logger.info(f"Scaling user-specified {len(columns_to_scale)} columns")
            else:
                # Auto-detect using column semantics (excludes PRICE columns by default)
                columns_to_scale = get_scalable_columns(result.columns.tolist())
                logger.info(
                    f"Auto-detected {len(columns_to_scale)} scalable columns "
                    f"(excluded {len([c for c in result.columns if c.lower() in PRICE_COLUMNS])} price columns)"
                )

            # Safety check: Verify no price columns in scaling list
            if self.config.exclude_price_columns_from_scaling:
                price_cols_to_remove = [
                    col for col in columns_to_scale if col.lower() in PRICE_COLUMNS
                ]
                if price_cols_to_remove:
                    logger.warning(
                        f"SAFETY: Removed {len(price_cols_to_remove)} price columns from scaling: "
                        f"{price_cols_to_remove[:3]}{'...' if len(price_cols_to_remove) > 3 else ''}"
                    )
                    columns_to_scale = [
                        col
                        for col in columns_to_scale
                        if col.lower() not in PRICE_COLUMNS
                    ]
                    if self.metrics:
                        self.metrics.warnings.append(
                            f"Excluded {len(price_cols_to_remove)} price columns from scaling for business metric preservation"
                        )

            # Apply scaling
            if columns_to_scale:
                try:
                    result = scale_features(
                        result,
                        columns=columns_to_scale,
                        scaler_type=self.config.scaler_type,
                        by_sector=self.config.scale_by_sector,
                        exclude_price_columns=self.config.exclude_price_columns_from_scaling,
                    )

                    # Track scaling in metrics
                    if self.metrics:
                        self.metrics.scaling_applied = True
                        self.metrics.scaler_type = self.config.scaler_type
                        self.metrics.scaled_columns_count = len(columns_to_scale)
                        self.metrics.price_columns_protected = (
                            self.config.exclude_price_columns_from_scaling
                        )
                        self.metrics.stages_executed.append("scaling")

                    logger.info(
                        f"Scaling complete: {len(columns_to_scale)} columns scaled using "
                        f"{self.config.scaler_type} scaler (sector-aware: {self.config.scale_by_sector})"
                    )

                except Exception as e:
                    error_msg = f"Scaling failed: {e}"
                    logger.error(error_msg)
                    if self.metrics:
                        self.metrics.errors.append(error_msg)
                    # Don't raise - allow pipeline to continue without scaling
                    logger.warning("Pipeline continuing without scaling")
            else:
                logger.warning("No columns to scale after applying exclusions")
                if self.metrics:
                    self.metrics.warnings.append(
                        "Scaling skipped: no valid columns found"
                    )

        # Stage 8: Compute financial metrics (optional)
        initial_cols = set(result.columns)

        if self.config.compute_valuation_metrics:
            logger.info("Stage 8a: Computing valuation metrics")
            result = compute_valuation_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.valuation_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("valuation_metrics")
            initial_cols = set(result.columns)

        if self.config.compute_profitability_metrics:
            logger.info("Stage 8b: Computing profitability metrics")
            result = compute_profitability_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.profitability_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("profitability_metrics")
            initial_cols = set(result.columns)

        if self.config.compute_growth_metrics:
            logger.info("Stage 8c: Computing growth metrics")
            result = compute_growth_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.growth_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("growth_metrics")
            initial_cols = set(result.columns)

        if self.config.compute_leverage_metrics:
            logger.info("Stage 8d: Computing leverage metrics")
            result = compute_leverage_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.leverage_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("leverage_metrics")
            initial_cols = set(result.columns)

        if self.config.compute_target_vs_price:
            logger.info("Stage 8e: Computing target vs price metrics")
            result = compute_target_vs_price_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.target_vs_price_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("target_vs_price_metrics")
            initial_cols = set(result.columns)

        if self.config.handle_sector_specific_metrics:
            logger.info("Stage 8f: Handling sector-specific metrics")
            result = handle_sector_specific_metrics(result)
            result = compute_sector_specific_ratios(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.sector_specific_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("sector_specific_metrics")

        # Stage 8g: Post-metrics imputation to ensure zero missing values
        # Rationale: Financial metrics computations can introduce NaNs. We apply
        # sector-aware imputation for imputable metrics and handle conditional
        # metrics separately to satisfy notebook assertion (zero NaNs) without
        # violating conditional metric semantics.
        try:
            missing_before_post = int(result.isna().sum().sum())
            if missing_before_post > 0:
                logger.info(
                    f"Stage 8g: Post-metrics imputation for computed metrics (missing={missing_before_post})"
                )

                # Impute only supported computed metrics; conditional are preserved inside
                result, post_imp_stats = impute_computed_metrics(
                    result, method="sector_median", sector_column="sector"
                )

                # Replace remaining NaNs in conditional metrics with 0 and add applicability flags
                # This preserves model-readiness (no NaNs) while signaling applicability.
                for col in [c for c in CONDITIONAL_METRICS if c in result.columns]:
                    applicable_flag = f"{col}_applicable"
                    # Create applicability flag BEFORE filling
                    result[applicable_flag] = result[col].notna()
                    # Fill remaining NaNs with 0 to satisfy zero-missing invariant
                    result[col] = result[col].fillna(0)

                missing_after_post = int(result.isna().sum().sum())
                if self.metrics:
                    self.metrics.stages_executed.append("post_metrics_imputation")
                    # If we previously tracked imputation, keep the stricter value
                    self.metrics.missing_values_after_imputation = missing_after_post
                    self.metrics.imputation_completeness = missing_after_post == 0
                    if not self.metrics.imputation_completeness:
                        self.metrics.warnings.append(
                            "Post-metrics imputation incomplete: remaining NaNs present"
                        )

                logger.info(
                    f"Post-metrics imputation complete: remaining missing={missing_after_post}"
                )
        except Exception as e:
            logger.warning(f"Post-metrics imputation step skipped due to error: {e}")

        # Stage 9: Apply feature engineering (Section 9.3)
        if self.config.apply_feature_engineering:
            logger.info("Stage 9: Applying feature engineering")
            result = self._apply_feature_engineering(result)
            if self.metrics:
                self.metrics.stages_executed.append("feature_engineering")

            # Stage 9b: Post-feature-engineering imputation sweep (GUIDELINES: must restore 0-missing invariant)
            # Rationale:
            # - Feature engineering can create NaNs (ratios, divide-by-zero, missing inputs).
            # - Median-only imputation is insufficient when:
            #   (a) engineered columns are all-NaN (median is NaN),
            #   (b) missingness exists in non-numeric columns.
            # Reference: code_guidelines.md Section 19.1, ml_workflow_guidelines.md Phase 9.1
            missing_after_features = int(result.isna().sum().sum())
            if missing_after_features > 0:
                logger.info(
                    f"Stage 9b: Post-feature-engineering imputation sweep (missing={missing_after_features})"
                )

                # Treat infinities as missing before imputation (common after ratio math)
                result = result.replace([np.inf, -np.inf], np.nan)

                # Re-run the full 6-step strategy to restore completeness post-feature engineering
                try:
                    result = apply_enhanced_imputation_strategy_6step(
                        result,
                        sector_column=self.config.imputation_sector_column,
                        n_neighbors=self.config.knn_neighbors,
                        price_column=self.config.imputation_price_column,
                        handle_categoricals=self.config.handle_categorical_imputation,
                        handle_dates=self.config.handle_datetime_imputation,
                    )
                    if self.metrics:
                        self.metrics.stages_executed.append(
                            "post_feature_imputation_6step"
                        )
                except Exception as e:
                    logger.warning(
                        f"Post-feature 6-step imputation failed ({e}); falling back to median imputation"
                    )
                    result = apply_median_imputation(result)
                    if self.metrics:
                        self.metrics.stages_executed.append(
                            "post_feature_imputation_median_fallback"
                        )

                # Final deterministic completeness pass for pathological columns (e.g., all-NaN columns)
                # This ensures the Section 19.1 invariant (0 missing) is always satisfied.
                remaining_missing = int(result.isna().sum().sum())
                if remaining_missing > 0:
                    logger.info(
                        f"Stage 9b-final: Deterministic completeness pass (remaining={remaining_missing})"
                    )

                    # Replace inf again just in case downstream steps reintroduced them
                    result = result.replace([np.inf, -np.inf], np.nan)

                    # Handle numeric columns: median then zero-fill for all-NaN columns
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        medians = result[numeric_cols].median(numeric_only=True)
                        result[numeric_cols] = result[numeric_cols].fillna(medians)
                        result[numeric_cols] = result[numeric_cols].fillna(0)

                    # Handle boolean columns
                    bool_cols = result.select_dtypes(include=["bool"]).columns
                    if len(bool_cols) > 0:
                        result[bool_cols] = result[bool_cols].fillna(False)

                    # Handle datetime columns
                    dt_cols = result.select_dtypes(
                        include=["datetime64[ns]", "datetimetz"]
                    ).columns
                    if len(dt_cols) > 0:
                        result[dt_cols] = result[dt_cols].fillna(
                            pd.Timestamp("1970-01-01")
                        )

                    # Handle object/string/category columns (FIX: categorical-aware)
                    obj_cols = result.select_dtypes(
                        include=["object", "string", "category"]
                    ).columns
                    if len(obj_cols) > 0:
                        # Iterate through each column to handle categorical dtypes correctly
                        for col in obj_cols:
                            if result[col].dtype.name == "category":
                                # Add 'UNKNOWN' to categories first if not already present
                                if "UNKNOWN" not in result[col].cat.categories:
                                    result[col] = result[col].cat.add_categories(
                                        ["UNKNOWN"]
                                    )
                                # Now safely fill NaN with 'UNKNOWN'
                                result[col] = result[col].fillna("UNKNOWN")
                            else:
                                # For object/string columns, direct fillna works
                                result[col] = result[col].fillna("UNKNOWN")

                    if self.metrics:
                        self.metrics.stages_executed.append(
                            "post_feature_imputation_final_pass"
                        )

                missing_after_sweep = int(result.isna().sum().sum())
                if self.metrics:
                    self.metrics.missing_values_after_imputation = missing_after_sweep
                    self.metrics.imputation_completeness = missing_after_sweep == 0
                    if not self.metrics.imputation_completeness:
                        self.metrics.warnings.append(
                            f"Post-feature-engineering imputation incomplete: {missing_after_sweep} NaNs remain"
                        )

                logger.info(
                    f"Post-feature-engineering imputation complete: "
                    f"{missing_after_features} → {missing_after_sweep} missing"
                )

        # Stage 10: Apply automated feature selection (Section 9.3 Task 1)
        if self.config.apply_feature_selection:
            logger.info("Stage 10: Applying automated feature selection")
            features_before = len(result.columns)

            # Need target column for feature selection
            target_col = None
            for col in ["price_target", "price_target_median"]:
                if col in result.columns:
                    target_col = col
                    break

            if target_col is not None:
                try:
                    # Separate features and target
                    y = result[target_col]
                    X = result.drop(columns=[target_col])

                    # Apply feature selection
                    X_selected = select_features_auto(
                        X,
                        y,
                        importance_threshold=self.config.importance_threshold,
                        correlation_threshold=self.config.correlation_threshold,
                        method=self.config.feature_selection_method,
                    )

                    # Reconstruct dataframe with selected features + target
                    result = X_selected.copy()
                    result[target_col] = y

                    features_after = len(result.columns) - 1  # Exclude target
                    features_removed = (
                        features_before - features_after - 1
                    )  # Exclude target from before count

                    if self.metrics:
                        self.metrics.feature_selection_applied = True
                        self.metrics.features_before_selection = features_before
                        self.metrics.features_after_selection = features_after
                        self.metrics.features_removed_by_selection = features_removed
                        self.metrics.stages_executed.append("feature_selection")

                    logger.info(
                        f"Feature selection complete: {features_before} -> {features_after} features "
                        f"(removed {features_removed}, reduction: {100 * features_removed / features_before:.1f}%)"
                    )
                except Exception as e:
                    error_msg = f"Feature selection failed: {e}"
                    logger.error(error_msg)
                    if self.metrics:
                        self.metrics.errors.append(error_msg)
                        self.metrics.warnings.append(
                            "Feature selection skipped due to error"
                        )
                    logger.warning("Pipeline continuing without feature selection")
            else:
                logger.warning(
                    "Feature selection skipped: no target column (price_target or price_target_median) found"
                )
                if self.metrics:
                    self.metrics.warnings.append(
                        "Feature selection skipped: no target column found"
                    )

        # Stage 11: Validate schema alignment (code_guidelines.md v1.11)
        if self.config.validate_quality:
            logger.info("Stage 11: Validating schema alignment")
            schema_validation = self._validate_schema_alignment(result)

            if self.metrics:
                self.metrics.schema_alignment_score = schema_validation.get(
                    "alignment_score", 1.0
                )
                self.metrics.unknown_columns_count = len(
                    schema_validation.get("unknown_columns", [])
                )
                self.metrics.missing_expected_columns_count = len(
                    schema_validation.get("missing_expected_columns", [])
                )
                self.metrics.dtype_mismatches_count = len(
                    schema_validation.get("dtype_mismatches", {})
                )
                self.metrics.stages_executed.append("schema_validation")

                # Add warnings for significant schema issues
                if schema_validation.get("alignment_score", 1.0) < 0.95:
                    self.metrics.warnings.append(
                        f"Schema alignment below 95%: {schema_validation.get('alignment_score', 1.0):.2%}"
                    )
                if len(schema_validation.get("unknown_columns", [])) > 10:
                    self.metrics.warnings.append(
                        f"Found {len(schema_validation.get('unknown_columns', []))} unknown columns"
                    )

        logger.info(
            f"Transformation complete: {len(result)} rows, {len(result.columns)} columns"
        )
        return result

    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return metrics.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Validating data quality")

        # Get region for quality check (default to 'ALL' if no region column)
        region = "ALL"
        if "region" in df.columns and len(df["region"].unique()) == 1:
            region = df["region"].iloc[0]

        quality_metrics = validate_financial_data_quality(df, region=region)

        if self.config.validate_pipeline:
            pipeline_metrics = perform_early_pipeline_validation(df)
            quality_metrics["pipeline_validation"] = pipeline_metrics

        return quality_metrics

    def _apply_semantic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column-semantics-aware transformations (Section 8.5).

        Transformation strategies by semantic category:
        - Price columns: NEVER transform (preserve original units)
        - Market value columns: Log-transform to handle skewness
        - Ratio columns: Skip winsorization (already normalized)
        - Percentage columns: Skip winsorization (bounded [0, 100])
        - Count columns: Optional discrete handling

        Args:
            df: DataFrame with normalized column names

        Returns:
            DataFrame with semantic-aware transformations applied
        """
        if not self.config.use_semantic_column_classification:
            logger.info("Semantic column classification disabled, skipping")
            return df

        # Classify columns by semantic type
        classification = classify_columns(list(df.columns))

        logger.info(
            f"Column classification: "
            f"price={len(classification['price'])}, "
            f"market_value={len(classification['market_value'])}, "
            f"ratio={len(classification['ratio'])}, "
            f"percentage={len(classification['percentage'])}, "
            f"count={len(classification['count'])}, "
            f"other={len(classification['other'])}"
        )

        # Update metrics with classification counts
        if self.metrics:
            self.metrics.semantic_classification_applied = True
            self.metrics.price_columns_count = len(classification["price"])
            self.metrics.market_value_columns_count = len(
                classification["market_value"]
            )
            self.metrics.ratio_columns_count = len(classification["ratio"])
            self.metrics.percentage_columns_count = len(classification["percentage"])
            self.metrics.count_columns_count = len(classification["count"])

        # Apply log-transforms to market value columns (high skewness)
        if self.config.log_transform_market_values:
            log_cols = get_log_transform_columns(list(df.columns))
            log_count = 0
            log_skipped = 0
            for col in log_cols:
                if col in df.columns:
                    # Check for negative values before log-transform (Priority 1 Fix)
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        # Skip log-transform for columns with negative values
                        # Add applicability flag for conditional metrics
                        df[f"log_{col}_applicable"] = df[col] >= 0
                        logger.warning(
                            f"Skipped log-transform for '{col}': {negative_count} negative values present "
                            f"(added applicability flag)"
                        )
                        log_skipped += 1
                        if self.metrics:
                            self.metrics.warnings.append(
                                f"Log-transform skipped for '{col}': {negative_count} negative values"
                            )
                    else:
                        # Safe to apply log-transform (all values >= 0)
                        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
                        log_count += 1
            logger.info(
                f"Applied log-transforms to {log_count} market value columns ({log_skipped} skipped due to negative values)"
            )
            if self.metrics:
                self.metrics.log_transformed_columns = log_count
                self.metrics.log_transforms_skipped = log_skipped

        return df

    def _get_winsorization_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns safe for winsorization based on semantic classification.

        Excludes:
        - Price columns (must preserve original units for business metric)
        - Ratio columns (already normalized)
        - Percentage columns (already bounded)
        - Count columns (discrete)

        Returns:
            List of column names safe for winsorization
        """
        if not self.config.use_semantic_column_classification:
            # Fallback: all numeric columns except explicit exclusions
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return [c for c in numeric_cols if c.lower() not in PRICE_COLUMNS]

        return get_winsorizable_columns(list(df.columns))

    def _get_scaling_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns safe for scaling based on semantic classification.

        Excludes price columns to preserve business metric integrity.

        Returns:
            List of column names safe for scaling
        """
        if not self.config.use_semantic_column_classification:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return [c for c in numeric_cols if c.lower() not in PRICE_COLUMNS]

        return get_scalable_columns(list(df.columns))

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering using the features.api module (Section 9.3).

        Integrates with build_features() for:
        - Valuation ratios (p_e_ratio, ev_ebitda_ratio, etc.)
        - Profitability metrics (roe, roa, margins)
        - Growth indicators (revenue_growth, earnings_growth)
        - Momentum features (price_momentum_1m, rsi_14d)
        - Quality scores (piotroski_f_score, altman_z_score)

        Args:
            df: Preprocessed DataFrame (post-imputation, pre-scaling)

        Returns:
            DataFrame with engineered features added
        """
        if not self.config.apply_feature_engineering:
            return df

        logger.info(
            f"Applying feature engineering with preset: {self.config.feature_preset}"
        )

        try:
            original_cols = set(df.columns)

            # Use the unified build_features API
            df_with_features = build_features(
                df,
                preset=self.config.feature_preset,
            )

            new_cols = set(df_with_features.columns) - original_cols
            logger.info(f"Added {len(new_cols)} engineered features")

            # Update metrics
            if self.metrics:
                self.metrics.feature_engineering_applied = True
                self.metrics.feature_preset_used = self.config.feature_preset
                self.metrics.features_added = len(new_cols)
                self.metrics.columns_output = len(df_with_features.columns)

            return df_with_features

        except Exception as e:
            logger.warning(
                f"Feature engineering failed: {e}, returning original DataFrame"
            )
            return df

    def _validate_schema_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame columns against COLUMN_SCHEMA registry.

        Checks for:
        - Unknown columns (present in df but not in COLUMN_SCHEMA)
        - Missing expected columns (in COLUMN_SCHEMA but not in df)
        - Dtype mismatches between actual and expected

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results:
            - unknown_columns: List of columns not in COLUMN_SCHEMA
            - missing_expected_columns: List of expected columns not in df
            - dtype_mismatches: Dict of columns with dtype mismatches
            - alignment_score: Float [0.0-1.0] indicating schema alignment quality
        """
        from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA, get_expected_dtype

        df_cols = set(df.columns)
        schema_cols = set(COLUMN_SCHEMA.keys())

        # Identify unknown columns (in df but not in schema)
        unknown_cols = sorted(df_cols - schema_cols)

        # Identify missing expected columns (in schema but not in df)
        # Only check for non-auxiliary columns that should be present
        expected_cols = {
            col
            for col, meta in COLUMN_SCHEMA.items()
            if meta.get("role") not in ["auxiliary", "label"]
        }
        missing_cols = sorted(expected_cols - df_cols)

        # Check dtype mismatches for columns present in both
        dtype_mismatches = {}
        common_cols = df_cols & schema_cols
        for col in common_cols:
            expected_dtype = get_expected_dtype(col)
            actual_dtype = str(df[col].dtype)

            # Normalize dtype comparison (float64/float32 -> float, int64/int32 -> int)
            expected_normalized = expected_dtype
            actual_normalized = actual_dtype

            if "float" in actual_dtype:
                actual_normalized = "float"
            elif "int" in actual_dtype:
                actual_normalized = "int"
            elif "object" in actual_dtype or "string" in actual_dtype:
                actual_normalized = "string"
            elif "category" in actual_dtype:
                actual_normalized = "category"
            elif "datetime" in actual_dtype:
                actual_normalized = "datetime64[ns]"
            elif "bool" in actual_dtype:
                actual_normalized = "bool"

            if expected_normalized != actual_normalized:
                dtype_mismatches[col] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype,
                }

        # Calculate alignment score
        total_expected = len(expected_cols)
        total_present = len(common_cols & expected_cols)
        alignment_score = total_present / total_expected if total_expected > 0 else 1.0

        validation_result = {
            "unknown_columns": unknown_cols,
            "missing_expected_columns": missing_cols,
            "dtype_mismatches": dtype_mismatches,
            "alignment_score": alignment_score,
            "total_columns": len(df_cols),
            "schema_columns": len(schema_cols),
            "common_columns": len(common_cols),
        }

        # Log validation results
        if unknown_cols:
            logger.warning(
                f"Found {len(unknown_cols)} unknown columns not in COLUMN_SCHEMA"
            )
            if len(unknown_cols) <= 10:
                logger.warning(f"Unknown columns: {unknown_cols}")

        if missing_cols:
            logger.info(
                f"Missing {len(missing_cols)} expected columns from COLUMN_SCHEMA"
            )
            if len(missing_cols) <= 10:
                logger.info(f"Missing columns: {missing_cols}")

        if dtype_mismatches:
            logger.warning(f"Found {len(dtype_mismatches)} dtype mismatches")
            if len(dtype_mismatches) <= 5:
                for col, mismatch in list(dtype_mismatches.items())[:5]:
                    logger.warning(
                        f"  {col}: expected {mismatch['expected']}, got {mismatch['actual']}"
                    )

        logger.info(f"Schema alignment score: {alignment_score:.2%}")

        return validation_result

    def load(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Load stage: final validation and return processed DataFrame.

        Args:
            df: Transformed DataFrame
            validate: Perform final quality validation (default: True)

        Returns:
            Final processed DataFrame
        """
        logger.info("Load stage: finalizing DataFrame")

        if validate and self.config.validate_quality:
            quality_metrics = self.validate_quality(df)
            if self.metrics:
                self.metrics.quality_score = quality_metrics.get(
                    "data_quality_score", 1.0
                )
                if "pipeline_validation" in quality_metrics:
                    self.metrics.validation_score = quality_metrics[
                        "pipeline_validation"
                    ].get("validation_score", 1.0)
                    self.metrics.warnings.extend(
                        quality_metrics["pipeline_validation"].get("warnings", [])
                    )

        logger.info(f"Load complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def run(
        self,
        source: Literal["csv", "db", "all_stocks"],
        data_dir: Optional[Path | str] = None,
        db_url: Optional[str] = None,
        return_metrics: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
        """
        Run complete ETL pipeline.

        Args:
            source: Data source type ('csv', 'db', 'all_stocks')
            data_dir: Directory for CSV files (required if source='csv')
            db_url: Database URL (required if source='db' or 'all_stocks')
            return_metrics: Return metrics along with DataFrame

        Returns:
            Processed DataFrame, or (DataFrame, ETLMetrics) if return_metrics=True

        Raises:
            ValueError: If required arguments missing

        Example:
            >>> pipeline = ETLPipeline()
            >>> df = pipeline.run(source='csv', data_dir='data/')
            >>>
            >>> # With metrics
            >>> df, metrics = pipeline.run(source='db', db_url='postgresql://...', return_metrics=True)
            >>> print(metrics.summary())
        """
        pipeline_start = time.time()

        # Initialize metrics
        self.metrics = ETLMetrics(source_type=source)

        try:
            # EXTRACT stage
            extract_start = time.time()
            if source == "csv":
                if data_dir is None:
                    raise ValueError("data_dir required for source='csv'")
                df = self.extract_from_csv(data_dir)
            elif source == "db":
                if db_url is None:
                    raise ValueError("db_url required for source='db'")
                df = self.extract_from_db(db_url)
            elif source == "all_stocks":
                if db_url is None:
                    raise ValueError("db_url required for source='all_stocks'")
                df = self.extract_from_all_stocks(db_url)
            else:
                raise ValueError(f"Unknown source type: {source}")

            self.metrics.extract_time_sec = time.time() - extract_start
            self.metrics.rows_input = len(df)
            self.metrics.columns_input = len(df.columns)
            self.metrics.stages_executed.append("extract")

            # TRANSFORM stage
            transform_start = time.time()
            df = self.transform(df)
            self.metrics.transform_time_sec = time.time() - transform_start
            self.metrics.stages_executed.append("transform")

            # LOAD stage
            load_start = time.time()
            df = self.load(df)
            self.metrics.load_time_sec = time.time() - load_start
            self.metrics.rows_output = len(df)
            self.metrics.columns_output = len(df.columns)
            self.metrics.stages_executed.append("load")

            # Finalize metrics
            self.metrics.total_time_sec = time.time() - pipeline_start

            logger.info(f"ETL pipeline complete in {self.metrics.total_time_sec:.2f}s")
            logger.info(self.metrics.summary())

            if return_metrics:
                return df, self.metrics
            return df

        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            if self.metrics:
                self.metrics.errors.append(str(e))
            raise


def run_etl_pipeline(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    config: Optional[ETLConfig] = None,
    return_metrics: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Convenience function to run ETL pipeline with minimal configuration.

    This is the main entry point for the ETL pipeline. It handles:
    - Data extraction from CSV or database
    - Column normalization and schema validation
    - Data sanitization and quality checks
    - Optional log transforms
    - Final validation

    Args:
        source: Data source type
            - 'csv': Load from regional CSV files
            - 'db': Load from PostgreSQL equities table
            - 'all_stocks': Load from unified all_stocks table (recommended)
        data_dir: Directory containing CSV files (required for source='csv')
        db_url: SQLAlchemy database URL (required for source='db' or 'all_stocks')
        config: ETLConfig object for advanced configuration (optional)
        return_metrics: Return ETLMetrics along with DataFrame

    Returns:
        Processed DataFrame ready for ML pipeline
        If return_metrics=True, returns (DataFrame, ETLMetrics)

    Raises:
        ValueError: If required arguments missing or invalid
        FileNotFoundError: If CSV files not found
        ImportError: If database dependencies not available

    Examples:
        Basic CSV loading:
            >>> from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline
            >>> df = run_etl_pipeline(source='csv', data_dir='data/')

        Database loading with imputation:
            >>> from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig
            >>> config = ETLConfig(
            ...     apply_imputation=True,
            ...     imputation_strategy='6step',
            ...     apply_log_transforms=True,
            ...     knn_neighbors=5
            ... )
            >>> df, metrics = run_etl_pipeline(
            ...     source='all_stocks',
            ...     db_url='postgresql://user:pass@localhost/postgres',
            ...     config=config,
            ...     return_metrics=True
            ... )
            >>> # Check imputation effectiveness
            >>> print(metrics.summary())
            >>> assert metrics.imputation_completeness, "Imputation incomplete"

        With metrics for monitoring:
            >>> df, metrics = run_etl_pipeline(
            ...     source='csv',
            ...     data_dir='data/',
            ...     return_metrics=True
            ... )
            >>> print(f"Quality score: {metrics.quality_score}")
            >>> print(f"Processing time: {metrics.total_time_sec:.2f}s")

        Simplified imputation workflow:
            >>> from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation
            >>> df, metrics = etl_with_imputation(source='csv', data_dir='data/')
            >>> print(f"Missing values: {metrics.missing_values_before_imputation} → "
            ...       f"{metrics.missing_values_after_imputation}")
    """
    pipeline = ETLPipeline(config=config)
    return pipeline.run(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        return_metrics=return_metrics,
    )


# Convenience aliases for common use cases
def etl_from_csv(
    data_dir: Path | str,
    config: Optional[ETLConfig] = None,
) -> pd.DataFrame:
    """
    Shorthand for loading from CSV files.

    Args:
        data_dir: Directory containing regional CSV files
        config: Optional ETL configuration

    Returns:
        Processed DataFrame

    Example:
        >>> from finance_ml.ml_workflow.preprocessing.etl import etl_from_csv
        >>> df = etl_from_csv('data/')
    """
    return run_etl_pipeline(source="csv", data_dir=data_dir, config=config)


def etl_from_database(
    db_url: str,
    unified_table: bool = True,
    config: Optional[ETLConfig] = None,
) -> pd.DataFrame:
    """
    Shorthand for loading from database.

    Args:
        db_url: SQLAlchemy database URL
        unified_table: Use all_stocks table (True) or equities table (False)
        config: Optional ETL configuration

    Returns:
        Processed DataFrame

    Example:
        >>> from finance_ml.ml_workflow.preprocessing.etl import etl_from_database
        >>> df = etl_from_database('postgresql://localhost/postgres')
    """
    source = "all_stocks" if unified_table else "db"
    return run_etl_pipeline(source=source, db_url=db_url, config=config)


def etl_with_imputation(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    imputation_strategy: Literal["6step", "4step", "median_only"] = "6step",
    return_metrics: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    ETL pipeline with imputation enabled and metrics tracking.

    Shorthand for common use case: complete data loading with full imputation.
    This function is optimized for ML workflows that require zero missing values.

    Args:
        source: Data source type
            - 'csv': Load from regional CSV files
            - 'db': Load from PostgreSQL equities table
            - 'all_stocks': Load from unified all_stocks table (recommended)
        data_dir: Directory for CSV files (required if source='csv')
        db_url: Database URL (required if source='db' or 'all_stocks')
        imputation_strategy: Imputation approach
            - '6step': Complete imputation (numeric, categorical, dates) - RECOMMENDED
            - '4step': Numeric-only imputation (backward compatibility)
            - 'median_only': Simple median filling
        return_metrics: Always returns metrics for monitoring imputation (default: True)

    Returns:
        (DataFrame, ETLMetrics) tuple with imputation metrics if return_metrics=True
        DataFrame only if return_metrics=False

    Examples:
        Basic CSV loading with imputation:
            >>> from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation
            >>> df, metrics = etl_with_imputation(source='csv', data_dir='data/')
            >>> print(f"Imputation reduced missing values: "
            ...       f"{metrics.missing_values_before_imputation} → "
            ...       f"{metrics.missing_values_after_imputation}")
            >>> assert metrics.imputation_completeness, "Imputation failed to complete"

        Database loading with 4-step strategy:
            >>> df, metrics = etl_with_imputation(
            ...     source='all_stocks',
            ...     db_url='postgresql://user:pass@localhost/postgres',
            ...     imputation_strategy='4step'
            ... )
            >>> print(metrics.summary())

        Quick median-only imputation:
            >>> df, metrics = etl_with_imputation(
            ...     source='csv',
            ...     data_dir='data/',
            ...     imputation_strategy='median_only'
            ... )
    """
    config = ETLConfig(
        apply_imputation=True,
        imputation_strategy=imputation_strategy,
        apply_log_transforms=False,  # User can enable separately if needed
        apply_scaling=False,  # Imputation only
    )

    return run_etl_pipeline(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        config=config,
        return_metrics=return_metrics,
    )


def etl_with_imputation_and_scaling(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    imputation_strategy: Literal["6step", "4step", "median_only"] = "6step",
    scaler_type: Literal["robust", "standard", "minmax"] = "robust",
    scale_by_sector: bool = True,
    return_metrics: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Complete ETL pipeline with imputation AND scaling.

    Shorthand for: Extract → Transform (with 6-step imputation + scaling) → Load

    RECOMMENDED for ML pipelines that require standardized features.
    This function ensures zero missing values AND standardized features while
    preserving price columns for business metrics.

    Args:
        source: Data source type
            - 'csv': Load from regional CSV files
            - 'db': Load from PostgreSQL equities table
            - 'all_stocks': Load from unified all_stocks table (recommended)
        data_dir: Directory for CSV files (required if source='csv')
        db_url: Database URL (required if source='db' or 'all_stocks')
        imputation_strategy: Imputation approach (default: '6step')
            - '6step': Complete imputation (numeric, categorical, dates) - RECOMMENDED
            - '4step': Numeric-only imputation (backward compatibility)
            - 'median_only': Simple median filling
        scaler_type: Scaler type (default: 'robust')
            - 'robust': Robust to outliers, best for financial data
            - 'standard': Z-score standardization
            - 'minmax': Scale to [0, 1] range
        scale_by_sector: Sector-aware scaling (default: True)
            If True, scales within each sector separately, preserving sector relationships
        return_metrics: Return metrics for monitoring (default: True)

    Returns:
        (DataFrame, ETLMetrics) tuple if return_metrics=True
        DataFrame only if return_metrics=False

    Examples:
        Basic usage with all defaults:
            >>> from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation_and_scaling
            >>> df, metrics = etl_with_imputation_and_scaling(
            ...     source='all_stocks',
            ...     db_url='postgresql://localhost/postgres'
            ... )
            >>> print(metrics.summary())
            >>> assert metrics.imputation_completeness
            >>> assert metrics.scaling_applied
            >>> assert metrics.price_columns_protected  # CRITICAL check

        CSV loading with standard scaler:
            >>> df, metrics = etl_with_imputation_and_scaling(
            ...     source='csv',
            ...     data_dir='data/',
            ...     scaler_type='standard',
            ...     scale_by_sector=True
            ... )
            >>> print(f"Scaled {metrics.scaled_columns_count} columns")
            >>> print(f"Price columns protected: {metrics.price_columns_protected}")

        MinMax scaling without sector grouping:
            >>> df, metrics = etl_with_imputation_and_scaling(
            ...     source='csv',
            ...     data_dir='data/',
            ...     scaler_type='minmax',
            ...     scale_by_sector=False
            ... )
            >>> # Verify scaling was applied
            >>> assert metrics.scaling_applied
            >>> assert metrics.scaler_type == 'minmax'

        Full ML preprocessing (imputation + log transforms + scaling):
            >>> from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig
            >>> config = ETLConfig(
            ...     apply_imputation=True,
            ...     imputation_strategy='6step',
            ...     apply_log_transforms=True,
            ...     log_transform_method='log1p',
            ...     apply_scaling=True,
            ...     scaler_type='robust',
            ...     scale_by_sector=True
            ... )
            >>> df, metrics = run_etl_pipeline(
            ...     source='all_stocks',
            ...     db_url='postgresql://localhost/postgres',
            ...     config=config,
            ...     return_metrics=True
            ... )
    """
    config = ETLConfig(
        apply_imputation=True,
        imputation_strategy=imputation_strategy,
        apply_scaling=True,
        scaler_type=scaler_type,
        scale_by_sector=scale_by_sector,
        exclude_price_columns_from_scaling=True,  # CRITICAL: Always protect
        apply_log_transforms=False,  # User can enable via full config if needed
    )

    return run_etl_pipeline(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        config=config,
        return_metrics=return_metrics,
    )


def etl_with_financial_metrics(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    compute_all_metrics: bool = True,
    output_dir: Optional[Path] = None,
    return_metrics: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Complete ETL pipeline with financial metrics computation.

    Unified entry point for: Extract → Transform (with imputation + scaling +
    financial metrics) → Load

    This function consolidates the functionality of run_etl_pipeline() and
    run_financial_metrics_etl() into a single call.

    Args:
        source: Data source type
            - 'csv': Load from regional CSV files
            - 'db': Load from PostgreSQL equities table
            - 'all_stocks': Load from unified all_stocks table (recommended)
        data_dir: Directory for CSV files (required if source='csv')
        db_url: Database URL (required if source='db' or 'all_stocks')
        compute_all_metrics: Enable all financial metrics computation (default: True)
            When True, computes:
            - Valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
            - Profitability metrics (margins, ROE, ROA)
            - Growth metrics (revenue, EBITDA, earnings growth)
            - Leverage metrics (debt ratios)
            - Target vs price metrics
            - Sector-specific ratios
        output_dir: Optional directory for quality alerts and dashboard JSON files
        return_metrics: Return metrics for monitoring (default: True)

    Returns:
        (DataFrame, ETLMetrics) tuple if return_metrics=True
        DataFrame only if return_metrics=False

    Examples:
        Basic usage with all metrics:
            >>> from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics
            >>> df, metrics = etl_with_financial_metrics(
            ...     source='csv',
            ...     data_dir='data/',
            ...     compute_all_metrics=True
            ... )
            >>> print(metrics.summary())
            >>> print(f"Valuation metrics added: {metrics.valuation_metrics_added}")

        Selective metrics computation:
            >>> from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig
            >>> config = ETLConfig(
            ...     apply_imputation=True,
            ...     compute_valuation_metrics=True,
            ...     compute_profitability_metrics=True,
            ...     compute_growth_metrics=False,
            ...     compute_leverage_metrics=False,
            ... )
            >>> df, metrics = run_etl_pipeline(
            ...     source='csv',
            ...     data_dir='data/',
            ...     config=config,
            ...     return_metrics=True
            ... )

    Note:
        This function replaces the two-step approach:
            # OLD (deprecated):
            # df = run_etl_pipeline(source='csv', data_dir='data/')
            # df, metrics = run_financial_metrics_etl(df, output_dir=output_dir)

            # NEW (recommended):
            df, metrics = etl_with_financial_metrics(
                source='csv',
                data_dir='data/',
                output_dir=output_dir
            )
    """
    config = ETLConfig(
        # Standard ETL options
        apply_imputation=True,
        imputation_strategy="6step",
        apply_scaling=False,  # Keep raw values for financial metrics
        # Financial metrics options
        compute_valuation_metrics=compute_all_metrics,
        compute_profitability_metrics=compute_all_metrics,
        compute_growth_metrics=compute_all_metrics,
        compute_leverage_metrics=compute_all_metrics,
        compute_target_vs_price=compute_all_metrics,
        handle_sector_specific_metrics=compute_all_metrics,
        # Quality reporting options
        generate_quality_alerts=True if output_dir else False,
        generate_metrics_dashboard=True if output_dir else False,
        output_subdir=str(output_dir) if output_dir else "financial_metrics",
    )

    return run_etl_pipeline(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        config=config,
        return_metrics=return_metrics,
    )


def etl_with_features(
    source: Literal["csv", "db", "all_stocks"],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    feature_preset: str = "comprehensive",
    feature_categories: Optional[List[str]] = None,
    auto_feature_selection: bool = False,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    config: Optional[ETLConfig] = None,
    return_metrics: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """
    Complete ETL pipeline with integrated feature engineering.

    Consolidates schema.py, column_semantics.py, and api.py functionality
    into a single entry point (Section 8.6, Section 9.3).

    Pipeline:
    1. Extract from source (CSV or database)
    2. Transform with semantic-aware strategies
    3. Apply feature engineering (Phase 9.3 features)
    4. Apply automated feature selection (optional, Task 1)
    5. Compute financial metrics
    6. Validate quality

    Args:
        source: Data source ('csv', 'db', 'all_stocks')
        data_dir: Directory for CSV files
        db_url: Database connection URL
        feature_preset: Feature engineering preset ('basic', 'momentum', 'quality',
            'standard', 'comprehensive')
        feature_categories: Specific feature categories to engineer
        auto_feature_selection: Enable automated feature selection (default: False)
        importance_threshold: Min importance score to keep feature (default: 0.01)
        correlation_threshold: Max correlation before deduplication (default: 0.95)
        config: Optional ETLConfig override
        return_metrics: Whether to return ETLMetrics

    Returns:
        DataFrame with all features, optionally with ETLMetrics

    Example:
        >>> # Basic usage with feature engineering
        >>> df, metrics = etl_with_features(
        ...     source='csv',
        ...     data_dir=Path('data'),
        ...     feature_preset='comprehensive',
        ...     return_metrics=True
        ... )
        >>> print(f"Shape: {df.shape}, Quality: {metrics.quality_score:.3f}")

        >>> # With automated feature selection (Phase 9.3 Task 1)
        >>> df, metrics = etl_with_features(
        ...     source='csv',
        ...     data_dir=Path('data'),
        ...     feature_preset='comprehensive',
        ...     auto_feature_selection=True,
        ...     importance_threshold=0.05,
        ...     correlation_threshold=0.95,
        ...     return_metrics=True
        ... )
    """
    # Build config with feature engineering enabled
    if config is None:
        config = ETLConfig()

    # Enable semantic-aware transformations
    config.use_semantic_column_classification = True
    config.preserve_price_columns = True
    config.log_transform_market_values = True

    # Enable feature engineering
    config.apply_feature_engineering = True
    config.feature_preset = feature_preset
    config.feature_categories = feature_categories

    # Enable feature selection (Phase 9.3 Task 1)
    config.apply_feature_selection = auto_feature_selection
    config.importance_threshold = importance_threshold
    config.correlation_threshold = correlation_threshold

    # Enable financial metrics
    config.compute_valuation_metrics = True
    config.compute_profitability_metrics = True
    config.compute_growth_metrics = True
    config.compute_leverage_metrics = True
    config.compute_target_vs_price = True

    return run_etl_pipeline(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        config=config,
        return_metrics=return_metrics,
    )
