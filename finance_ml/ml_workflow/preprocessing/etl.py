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
from typing import Optional, Dict, Any, Tuple, List, Literal

import pandas as pd

# Import from existing modules (no code duplication)
from finance_ml.ml_workflow.preprocessing.data import (
    load_from_csv,
    load_from_db,
    load_from_all_stocks,
    normalize_columns,
    validate_schema,
    sanitize_dataframe_with_logging,
    validate_financial_data_quality,
    perform_early_pipeline_validation,
)

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    list_numeric_feature_cols,
    list_categorical_cols,
)

from finance_ml.ml_workflow.preprocessing.transforms import (
    apply_log_transforms,
    get_skewness,
)

from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    validate_imputation_completeness,
    apply_median_imputation,
)

from finance_ml.ml_workflow.preprocessing.scaling import (
    scale_features,
)

from finance_ml.ml_workflow.preprocessing.column_semantics import (
    get_scalable_columns,
    PRICE_COLUMNS,
)

# Import financial metrics functions for unified ETL API
from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
    compute_valuation_metrics,
    compute_profitability_metrics,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_target_vs_price_metrics,
    handle_sector_specific_metrics,
    compute_sector_specific_ratios,
    generate_data_quality_alerts,
    generate_metrics_dashboard,
)

logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    """
    Configuration for ETL pipeline stages.

    Attributes:
        normalize_columns: Apply column name normalization (default: True)
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

    # Imputation metrics
    imputation_strategy: Optional[str] = None
    missing_values_before_imputation: int = 0
    missing_values_after_imputation: int = 0
    imputation_completeness: bool = False
    date_columns_ready: bool = False

    # Scaling metrics
    scaling_applied: bool = False
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
            "stages_executed": self.stages_executed,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
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

        return (
            f"ETL Pipeline Summary:\n"
            f"  Source: {self.source_type}\n"
            f"  Duration: {self.total_time_sec:.2f}s "
            f"(extract: {self.extract_time_sec:.2f}s, "
            f"transform: {self.transform_time_sec:.2f}s, "
            f"load: {self.load_time_sec:.2f}s)\n"
            f"  Data: {self.rows_input} → {self.rows_output} rows, "
            f"{self.columns_input} → {self.columns_output} columns"
            f"{imputation_info}"
            f"{scaling_info}"
            f"{financial_metrics_info}\n"
            f"  Quality: {self.quality_score:.3f}, "
            f"Validation: {self.validation_score:.3f}\n"
            f"  Stages: {', '.join(self.stages_executed)}\n"
            f"  Warnings: {len(self.warnings)}, Errors: {len(self.errors)}"
        )


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
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from all_stocks")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformation stages to DataFrame.

        Stages (configurable via ETLConfig):
        1. Column normalization
        2. Schema validation
        3. Data sanitization
        4. Log transforms (optional)
        5. Quality validation
        6. Pipeline validation

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

        # Stage 2: Validate schema
        if self.config.validate_schema:
            logger.info("Stage 2: Validating schema")
            is_valid, errors = validate_schema(result, require_target=self.config.require_target)
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
            result = sanitize_dataframe_with_logging(result)

        # Stage 5: Apply imputation strategy
        if self.config.apply_imputation:
            logger.info(f"Stage 5: Applying {self.config.imputation_strategy} imputation strategy")

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
                self.metrics.date_columns_ready = validation["ready_for_temporal_features"]
                self.metrics.quality_score = min(
                    self.metrics.quality_score, 1.0 if validation["is_complete"] else 0.8
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
                        col for col in columns_to_scale if col.lower() not in PRICE_COLUMNS
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
                    self.metrics.warnings.append("Scaling skipped: no valid columns found")

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

        logger.info(f"Transformation complete: {len(result)} rows, {len(result.columns)} columns")
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
                self.metrics.quality_score = quality_metrics.get("data_quality_score", 1.0)
                if "pipeline_validation" in quality_metrics:
                    self.metrics.validation_score = quality_metrics["pipeline_validation"].get(
                        "validation_score", 1.0
                    )
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
