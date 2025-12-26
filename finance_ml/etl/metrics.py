"""ETL metrics tracking - extracted from etl.py."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ETLMetrics:
    """
    Metrics collected during ETL pipeline execution.

    Attributes:
        source_type: Data source type (csv, db, all_stocks)
        extract_duration: Time spent in Extract stage
        transform_duration: Time spent in Transform stage
        load_duration: Time spent in Load stage
        total_duration: Total pipeline execution time
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

    source_type: str = ""
    extract_duration: float = 0.0
    transform_duration: float = 0.0
    load_duration: float = 0.0
    total_duration: float = 0.0
    rows_input: int = 0
    rows_output: int = 0
    columns_input: int = 0
    columns_output: int = 0
    
    # Quality metrics
    quality_score: float = 1.0
    validation_score: float = 1.0
    quality_alerts_generated: int = 0
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

    # Semantic classification metrics (from column_semantics.py constants)
    semantic_classification_applied: bool = False
    price_columns_count: int = 0  # From PRICE_COLUMNS
    market_value_columns_count: int = 0  # From MARKET_VALUE_COLUMNS
    ratio_columns_count: int = 0  # From RATIO_COLUMNS
    percentage_columns_count: int = 0  # From PERCENTAGE_COLUMNS
    count_columns_count: int = 0  # From COUNT_COLUMNS
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
    recognized_columns_count: int = 0  # Columns recognized by schema or allowlist
    allowlisted_engineered_columns_count: int = 0  # Recognized engineered (not in schema)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "source_type": self.source_type,
            "timings": {
                "extract_sec": self.extract_duration,
                "transform_sec": self.transform_duration,
                "load_sec": self.load_duration,
                "total_sec": self.total_duration,
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
                "recognized_columns": self.recognized_columns_count,
                "allowlisted_engineered": self.allowlisted_engineered_columns_count,
            },
            "stages_executed": self.stages_executed,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate human-readable summary of ETL metrics."""
        dtype_info = ""
        if self.dtype_casting_applied:
            warning_icon = "⚠" if self.dtype_coercion_warnings > 0 else "✓"
            dtype_info = (
                f"\\n  Dtype Casting: Applied "
                f"({self.dtype_coercion_warnings} coercion warnings, "
                f"{self.dtype_unknown_columns} unknown columns) {warning_icon}"
            )

        imputation_info = ""
        if self.imputation_strategy:
            status_icon = "✓" if self.imputation_completeness else "✗"
            imputation_info = (
                f"\\n  Imputation: {self.imputation_strategy} "
                f"({self.missing_values_before_imputation} → "
                f"{self.missing_values_after_imputation} missing) "
                f"{status_icon}"
            )

        scaling_info = ""
        if self.scaling_applied:
            protection_icon = "✓" if self.price_columns_protected else "✗ WARNING"
            scaling_info = (
                f"\\n  Scaling: {self.scaler_type} "
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
                f"\\n  Financial Metrics: {total_metrics} added "
                f"(valuation: {self.valuation_metrics_added}, "
                f"profitability: {self.profitability_metrics_added}, "
                f"growth: {self.growth_metrics_added}, "
                f"leverage: {self.leverage_metrics_added})"
            )

        # Semantic transformation info
        semantic_info = ""
        if self.semantic_classification_applied:
            semantic_info = (
                f"\\n  Semantic Classification: ✓ "
                f"(Price Columns: {self.price_columns_count}, "
                f"Market Value: {self.market_value_columns_count}, "
                f"Ratios: {self.ratio_columns_count}, "
                f"Log-Transformed: {self.log_transformed_columns})"
            )

        # Feature engineering info
        feature_engineering_info = ""
        if self.feature_engineering_applied:
            feature_engineering_info = (
                f"\\n  Feature Engineering: {self.feature_preset_used} "
                f"({self.features_added} features added)"
            )

        # Feature selection info
        feature_selection_info = ""
        if self.feature_selection_applied:
            reduction_pct = (
                100 * self.features_removed_by_selection / self.features_before_selection
                if self.features_before_selection > 0
                else 0
            )
            feature_selection_info = (
                f"\\n  Feature Selection: {self.features_before_selection} → "
                f"{self.features_after_selection} features "
                f"(removed {self.features_removed_by_selection}, {reduction_pct:.1f}% reduction)"
            )

        # Business rule validation info (Priority 1 Fix)
        business_rules_info = ""
        if self.business_rule_violations > 0 or self.log_transforms_skipped > 0:
            status_icon = "⚠" if self.business_rule_violations > 0 else "✓"
            business_rules_info = (
                f"\\n  Business Rules: {status_icon} "
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
            status_icon = "✓" if self.schema_alignment_score >= 0.90 else "⚠"
            recognized_info = (
                f", recognized: {self.recognized_columns_count}"
                if self.recognized_columns_count > 0
                else ""
            )
            schema_validation_info = (
                f"\\n  Schema Validation: {status_icon} "
                f"(alignment: {self.schema_alignment_score:.2%}, "
                f"unknown extra: {self.unknown_columns_count}, "
                f"missing required: {self.missing_expected_columns_count}, "
                f"dtype mismatches: {self.dtype_mismatches_count}"
                f"{recognized_info})"
            )

        return (
            f"ETL Pipeline Summary:\\n"
            f"  Source: {self.source_type}\\n"
            f"  Duration: {self.total_duration:.2f}s "
            f"(extract: {self.extract_duration:.2f}s, "
            f"transform: {self.transform_duration:.2f}s, "
            f"load: {self.load_duration:.2f}s)\\n"
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
            f"{schema_validation_info}\\n"
            f"  Quality: {self.quality_score:.3f}, "
            f"Validation: {self.validation_score:.3f}"
        )
