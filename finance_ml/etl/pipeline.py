"""ETL Pipeline orchestration - extracted from etl.py."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from finance_ml.etl.config import ETLConfig
from finance_ml.etl.metrics import ETLMetrics
from finance_ml.etl.stages import (
    run_extraction_stage,
    run_dtype_casting_stage,
    run_semantic_classification_stage,
    run_validation_stage,
    run_row_dropping_stage,
    run_sanitization_stage,
    run_imputation_stage,
    run_semantic_transformations_stage,
    run_scaling_stage,
    run_financial_metrics_stage,
    run_post_metrics_imputation_stage,
    run_feature_engineering_stage,
    run_feature_selection_stage,
    run_schema_alignment_validation_stage,
    run_quality_validation_stage,
)
from finance_ml.ml_workflow.preprocessing.data import (
    load_from_all_stocks,
    load_from_csv,
    load_from_db,
)
from finance_ml.ml_workflow.preprocessing.dtypes import to_jsonable

logger = logging.getLogger(__name__)

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
            report["failed"].append(f"column_count: {df.shape[1]} < expected {expected_min_cols}")

    # Check minimum rows
    if expected_min_rows > 0:
        if df.shape[0] >= expected_min_rows:
            report["passed"].append(f"row_count >= {expected_min_rows}")
        else:
            report["failed"].append(f"row_count: {df.shape[0]} < expected {expected_min_rows}")

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
            logger.debug(f"{phase}: {total_nan} NaN values in {len(cols_with_nan)} columns")
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

        Args:
            data_dir: Directory containing CSV files
            limit: Optional row limit

        Returns:
            Combined DataFrame from all regions
        """
        logger.info(f"Extracting data from CSV: {data_dir}")
        limit = limit or self.config.extraction.limit
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
        """
        logger.info(f"Extracting data from database: {db_url}")
        limit = limit or self.config.extraction.limit
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

        Args:
            db_url: SQLAlchemy database URL
            limit: Optional row limit

        Returns:
            DataFrame from all_stocks table
        """
        logger.info(f"Extracting data from all_stocks table: {db_url}")
        limit = limit or self.config.extraction.limit
        df = load_from_all_stocks(db_url, limit=limit)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from all_stocks")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformation stages to DataFrame using modular components."""
        logger.info("Starting modular transformation pipeline")
        result = df.copy()

        # Stage 1: Normalize columns
        result = run_extraction_stage(result, normalize=self.config.extraction.normalize_column_names)

        # Stage 1.5: Dtype casting
        if self.config.dtype_casting.apply_dtype_casting:
            result, dtype_diagnostics = run_dtype_casting_stage(result, track_diagnostics=self.config.dtype_casting.track_diagnostics)
            if self.metrics:
                self.metrics.dtype_casting_applied = True
                self.metrics.dtype_diagnostics = to_jsonable(dtype_diagnostics)
                self.metrics.stages_executed.append("dtype_casting")

        # Stage 1.6: Semantic Classification
        if self.config.semantic_classification.enabled:
            classification_stats = run_semantic_classification_stage(result)
            if self.metrics:
                self.metrics.price_columns_count = classification_stats["price_columns_count"]
                self.metrics.market_value_columns_count = classification_stats["market_value_columns_count"]
                self.metrics.ratio_columns_count = classification_stats["ratio_columns_count"]
                self.metrics.percentage_columns_count = classification_stats["percentage_columns_count"]
                self.metrics.count_columns_count = classification_stats["count_columns_count"]
                self.metrics.semantic_classification_applied = True
                self.metrics.stages_executed.append("semantic_classification")

        # Stage 2: Validate schema
        if self.config.validation.validate_schema:
            is_valid, errors = run_validation_stage(result, require_target=self.config.validation.require_target_column)
            if not is_valid and self.metrics:
                self.metrics.errors.extend(errors)

        # Stage 3: Drop invalid rows
        if self.config.validation.drop_rows_with_missing_critical_fields:
            result = run_row_dropping_stage(result)

        # Stage 4: Sanitize data
        if self.config.sanitization.sanitize_data:
            result = run_sanitization_stage(result)
            if self.metrics:
                self.metrics.stages_executed.append("sanitization")

        # Stage 5: Apply imputation strategy
        if self.config.imputation.apply_imputation:
            result = run_imputation_stage(
                result, 
                strategy=self.config.imputation.strategy,
                sector_column=self.config.imputation.sector_column,
                reference_price_column=self.config.imputation.reference_price_column
            )
            if self.metrics:
                self.metrics.imputation_strategy = self.config.imputation.strategy
                self.metrics.stages_executed.append("imputation")

        # Stage 6: Apply semantic-aware transformations (log transforms)
        if self.config.semantic_transform.apply_log_transforms or self.config.semantic_transform.log_transform_market_values:
            result, trans_count, skip_count = run_semantic_transformations_stage(
                result,
                apply_log_transforms=self.config.semantic_transform.apply_log_transforms,
                log_transform_market_values=self.config.semantic_transform.log_transform_market_values,
                log_transform_target_columns=self.config.semantic_transform.log_transform_target_columns
            )
            if self.metrics:
                self.metrics.log_transformed_columns = trans_count
                self.metrics.log_transforms_skipped = skip_count
                self.metrics.stages_executed.append("semantic_transformations")

        # Stage 7: Apply feature scaling
        if self.config.scaling.enabled:
            result = run_scaling_stage(
                result, 
                scaler_type=self.config.scaling.scaler_type, 
                scale_by_sector=self.config.scaling.scale_by_sector
            )
            if self.metrics:
                self.metrics.scaling_applied = True
                self.metrics.stages_executed.append("scaling")

        # Stage 8: Compute financial metrics
        result, metrics_stats = run_financial_metrics_stage(
            result,
            compute_valuation=self.config.financial_metrics.compute_valuation_metrics,
            compute_profitability=self.config.financial_metrics.compute_profitability_metrics,
            compute_growth=self.config.financial_metrics.compute_growth_metrics,
            compute_leverage=self.config.financial_metrics.compute_leverage_metrics,
            compute_target_vs_price=self.config.financial_metrics.compute_target_vs_price_metrics,
            compute_sector_specific=self.config.financial_metrics.compute_sector_specific_metrics
        )
        if self.metrics:
            for k, v in metrics_stats.items():
                setattr(self.metrics, k, v)
            self.metrics.stages_executed.append("financial_metrics")

        # Stage 8g: Post-metrics imputation
        result, missing_after = run_post_metrics_imputation_stage(result)
        if self.metrics:
            self.metrics.missing_values_after_imputation = missing_after
            self.metrics.imputation_completeness = (missing_after == 0)
            self.metrics.stages_executed.append("post_metrics_imputation")

        # Stage 9: Feature engineering
        if self.config.feature_engineering.enabled:
            result = run_feature_engineering_stage(
                result, 
                preset=self.config.feature_engineering.preset,
                categories=self.config.feature_engineering.categories,
                engineer_earnings_analytics=self.config.feature_engineering.engineer_earnings_analytics
            )
            if self.metrics:
                self.metrics.stages_executed.append("feature_engineering")

        # Stage 10: Automated feature selection
        if self.config.feature_selection.enabled:
            result, f_before, f_after, f_removed = run_feature_selection_stage(
                result,
                method=self.config.feature_selection.method,
                importance_threshold=self.config.feature_selection.min_importance_threshold,
                correlation_threshold=self.config.feature_selection.max_correlation_threshold,
                preserve_columns=self.config.feature_selection.preserve_columns,
            )
            if self.metrics:
                self.metrics.feature_selection_applied = True
                self.metrics.features_before_selection = f_before
                self.metrics.features_after_selection = f_after
                self.metrics.features_removed_by_selection = f_removed
                self.metrics.stages_executed.append("feature_selection")

        # Stage 11: Validate schema alignment
        if self.config.validation.validate_schema_alignment:
            schema_validation = run_schema_alignment_validation_stage(result)
            if self.metrics:
                self.metrics.schema_alignment_score = schema_validation.get("alignment_score", 1.0)
                self.metrics.unknown_columns_count = len(schema_validation.get("unknown_columns", []))
                self.metrics.missing_expected_columns_count = len(schema_validation.get("missing_expected_columns", []))
                self.metrics.dtype_mismatches_count = len(schema_validation.get("dtype_mismatches", {}))
                self.metrics.recognized_columns_count = int(schema_validation.get("recognized_columns_count", 0))
                self.metrics.allowlisted_engineered_columns_count = len(schema_validation.get("allowlisted_engineered", []))
                self.metrics.stages_executed.append("schema_validation")

        # Stage 12: Quality validation
        if self.config.validation.validate_quality:
            quality_metrics = run_quality_validation_stage(result, validate_pipeline=self.config.validation.validate_pipeline)
            if self.metrics:
                self.metrics.quality_score = quality_metrics.get("overall_quality_score", 0.0)
                self.metrics.stages_executed.append("quality_validation")

        logger.info(f"Transformation complete: {len(result)} rows, {len(result.columns)} columns")
        return result

    def load(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Load processed data (final validation).

        Args:
            df: Processed DataFrame
            validate: Apply final validation checks

        Returns:
            Final DataFrame ready for downstream modules
        """
        if validate:
            is_valid, report = validate_etl_output(
                df,
                phase="final_load",
                expected_min_cols=50,
                critical_columns=["ticker", "sector", "last_price"],
            )
            if self.metrics:
                self.metrics.validation_score = 1.0 if is_valid else 0.0
                if not is_valid:
                    self.metrics.errors.append(f"Final validation failed: {report['failed']}")

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

            self.metrics.extract_duration = time.time() - extract_start
            self.metrics.rows_input = len(df)
            self.metrics.columns_input = len(df.columns)
            self.metrics.stages_executed.append("extract")

            # TRANSFORM stage
            transform_start = time.time()
            df = self.transform(df)
            self.metrics.transform_duration = time.time() - transform_start
            self.metrics.stages_executed.append("transform")

            # LOAD stage
            load_start = time.time()
            df = self.load(df)
            self.metrics.load_duration = time.time() - load_start
            self.metrics.rows_output = len(df)
            self.metrics.columns_output = len(df.columns)
            self.metrics.stages_executed.append("load")

            # Finalize metrics
            self.metrics.total_duration = time.time() - pipeline_start

            logger.info(f"ETL pipeline complete in {self.metrics.total_duration:.2f}s")
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
    """
    pipeline = ETLPipeline(config=config)
    return pipeline.run(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        return_metrics=return_metrics,
    )
