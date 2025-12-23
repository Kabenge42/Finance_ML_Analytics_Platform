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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    # Add schema utility functions for Stage 3 classification
    list_categorical_cols,
    list_numeric_feature_cols,
)
# Import feature engineering API (Section 9.3)
from finance_ml.ml_workflow.features.api import build_features
# Import feature selection API (Section 9.3 Task 1)
from finance_ml.ml_workflow.features.selection import select_features_auto
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
    get_scalable_columns,
    get_winsorizable_columns,
    # Add semantic column constants for pipeline stages
    COUNT_COLUMNS,
    MARKET_VALUE_COLUMNS,
    PERCENTAGE_COLUMNS,
    PRICE_COLUMNS,
    RATIO_COLUMNS,
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

logger = logging.getLogger(__name__)


@dataclass
class DataExtractionConfig:
    """Configuration for ETL Stage 1: Data Extraction."""

    limit: Optional[int] = None
    normalize_column_names: bool = True


@dataclass
class SchemaValidationConfig:
    """Configuration for ETL Stages 3 & 11: Schema Validation."""

    validate_schema: bool = True
    require_target_column: bool = False
    drop_rows_with_missing_critical_fields: bool = True
    validate_schema_alignment: bool = True
    schema_alignment_threshold: float = 0.95
    validate_pipeline: bool = True
    validate_quality: bool = True
    custom_validators: List[Any] = field(default_factory=list)


@dataclass
class DtypeCastingConfig:
    """Configuration for ETL Stage 4: Dtype Casting."""

    apply_dtype_casting: bool = True
    track_diagnostics: bool = True


@dataclass
class SemanticClassificationConfig:
    """Configuration for ETL Stage 5: Semantic Column Classification."""

    enabled: bool = True
    preserve_price_columns: bool = True


@dataclass
class ImputationConfig:
    """Configuration for ETL Stages 6 & 10: Missing Value Imputation.
    
    Enhanced with categorical encoding support for ordinal and one-hot encoding.
    """

    apply_imputation: bool = True
    strategy: Literal["6step", "4step", "median_only"] = "6step"
    knn_neighbors: int = 5
    sector_column: str = "sector"
    reference_price_column: str = "last_price"
    impute_categorical_columns: bool = True
    impute_datetime_columns: bool = True
    # NEW: Categorical encoding options
    apply_categorical_encoding: bool = False
    ordinal_columns: Optional[List[str]] = None  # e.g., ['style_class', 'size_class']
    onehot_columns: Optional[List[str]] = None   # e.g., ['sector', 'region']
    onehot_drop_first: bool = True
    onehot_min_frequency: float = 0.01


@dataclass
class SemanticTransformConfig:
    """Configuration for ETL Stage 7: Semantic-Aware Transformations."""

    apply_log_transforms: bool = False
    log_transform_method: Literal["log1p", "signed_log"] = "log1p"
    log_transform_market_values: bool = True
    log_transform_target_columns: Optional[List[str]] = None
    exclude_ratios_from_winsorization: bool = True
    exclude_percentages_from_winsorization: bool = True
    exclude_counts_from_scaling: bool = True


@dataclass
class DataSanitizationConfig:
    """Configuration for ETL Stage 8: Data Sanitization & Winsorization."""

    sanitize_data: bool = True
    apply_winsorization: bool = False
    winsorize_lower_percentile: float = 0.05
    winsorize_upper_percentile: float = 0.95


@dataclass
class ScalingConfig:
    """Configuration for ETL Stage 9: Feature Scaling."""

    enabled: bool = False
    scaler_type: Literal["robust", "standard", "minmax"] = "robust"
    scale_by_sector: bool = True
    target_columns: Optional[List[str]] = None
    exclude_price_columns: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Configuration for Feature Engineering (Phase 9.3)."""

    enabled: bool = False
    preset: str = "comprehensive"
    categories: Optional[List[str]] = None
    engineer_earnings_analytics: bool = (
        True  # Enable Estimated vs. Actual and GAAP vs. Adjusted analytics
    )


@dataclass
class FeatureSelectionConfig:
    """Configuration for Feature Selection (Phase 9.3 Task 1)."""

    enabled: bool = False
    method: Literal["mutual_info", "correlation", "both"] = "mutual_info"
    min_importance_threshold: float = 0.01
    max_correlation_threshold: float = 0.95
    categories: Optional[List[str]] = None


@dataclass
class FinancialMetricsConfig:
    """Configuration for Financial Metrics Computation (Optional Enhancement)."""

    compute_valuation_metrics: bool = False
    compute_profitability_metrics: bool = False
    compute_growth_metrics: bool = False
    compute_leverage_metrics: bool = False
    compute_target_vs_price_metrics: bool = False
    compute_sector_specific_metrics: bool = False
    generate_quality_alerts: bool = False
    generate_metrics_dashboard: bool = False
    output_directory: str = "financial_metrics"


@dataclass
class ETLConfig:
    """
    Unified ETL Pipeline Configuration (11 Stages).

    Aligns with ml_workflow_guidelines.md Phase 9.1.
    """

    extraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
    validation: SchemaValidationConfig = field(default_factory=SchemaValidationConfig)
    dtype_casting: DtypeCastingConfig = field(default_factory=DtypeCastingConfig)
    semantic_classification: SemanticClassificationConfig = field(
        default_factory=SemanticClassificationConfig
    )
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    semantic_transform: SemanticTransformConfig = field(default_factory=SemanticTransformConfig)
    sanitization: DataSanitizationConfig = field(default_factory=DataSanitizationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    financial_metrics: FinancialMetricsConfig = field(default_factory=FinancialMetricsConfig)

    def __init__(
        self,
        extraction: Optional[DataExtractionConfig] = None,
        validation: Optional[SchemaValidationConfig] = None,
        dtype_casting: Optional[DtypeCastingConfig] = None,
        semantic_classification: Optional[SemanticClassificationConfig] = None,
        imputation: Optional[ImputationConfig] = None,
        semantic_transform: Optional[SemanticTransformConfig] = None,
        sanitization: Optional[DataSanitizationConfig] = None,
        scaling: Optional[ScalingConfig] = None,
        feature_engineering: Optional[FeatureEngineeringConfig] = None,
        feature_selection: Optional[FeatureSelectionConfig] = None,
        financial_metrics: Optional[FinancialMetricsConfig] = None,
        **legacy_kwargs: Any,
    ) -> None:
        self.extraction = extraction or DataExtractionConfig()
        self.validation = validation or SchemaValidationConfig()
        self.dtype_casting = dtype_casting or DtypeCastingConfig()
        self.semantic_classification = semantic_classification or SemanticClassificationConfig()
        self.imputation = imputation or ImputationConfig()
        self.semantic_transform = semantic_transform or SemanticTransformConfig()
        self.sanitization = sanitization or DataSanitizationConfig()
        self.scaling = scaling or ScalingConfig()
        self.feature_engineering = feature_engineering or FeatureEngineeringConfig()
        self.feature_selection = feature_selection or FeatureSelectionConfig()
        self.financial_metrics = financial_metrics or FinancialMetricsConfig()

        # Map legacy flat arguments into nested configs for backward compatibility
        self._apply_legacy_overrides(legacy_kwargs)

    def _apply_legacy_overrides(self, legacy_kwargs: Dict[str, Any]) -> None:
        legacy_map = {
            "normalize_columns": ("extraction", "normalize_column_names"),
            "limit": ("extraction", "limit"),
            "validate_schema": ("validation", "validate_schema"),
            "require_target": ("validation", "require_target_column"),
            "drop_invalid_rows": (
                "validation",
                "drop_rows_with_missing_critical_fields",
            ),
            "validate_quality": ("validation", "validate_quality"),
            "validate_pipeline": ("validation", "validate_pipeline"),
            "custom_validators": ("validation", "custom_validators"),
            "apply_dtype_casting": ("dtype_casting", "apply_dtype_casting"),
            "track_dtype_diagnostics": ("dtype_casting", "track_diagnostics"),
            "use_semantic_column_classification": (
                "semantic_classification",
                "enabled",
            ),
            "preserve_price_columns": (
                "semantic_classification",
                "preserve_price_columns",
            ),
            "apply_imputation": ("imputation", "apply_imputation"),
            "imputation_strategy": ("imputation", "strategy"),
            "knn_neighbors": ("imputation", "knn_neighbors"),
            "imputation_sector_column": ("imputation", "sector_column"),
            "imputation_price_column": (
                "imputation",
                "reference_price_column",
            ),
            "handle_categorical_imputation": (
                "imputation",
                "impute_categorical_columns",
            ),
            "handle_datetime_imputation": (
                "imputation",
                "impute_datetime_columns",
            ),
            "apply_log_transforms": (
                "semantic_transform",
                "apply_log_transforms",
            ),
            "log_transform_method": (
                "semantic_transform",
                "log_transform_method",
            ),
            "log_transform_market_values": (
                "semantic_transform",
                "log_transform_market_values",
            ),
            "log_transform_columns": (
                "semantic_transform",
                "log_transform_target_columns",
            ),
            "exclude_ratios_from_winsorization": (
                "semantic_transform",
                "exclude_ratios_from_winsorization",
            ),
            "exclude_percentages_from_winsorization": (
                "semantic_transform",
                "exclude_percentages_from_winsorization",
            ),
            "exclude_counts_from_scaling": (
                "semantic_transform",
                "exclude_counts_from_scaling",
            ),
            "sanitize_data": ("sanitization", "sanitize_data"),
            "apply_winsorization": ("sanitization", "apply_winsorization"),
            "winsorize_lower_percentile": (
                "sanitization",
                "winsorize_lower_percentile",
            ),
            "winsorize_upper_percentile": (
                "sanitization",
                "winsorize_upper_percentile",
            ),
            "apply_scaling": ("scaling", "enabled"),
            "scaler_type": ("scaling", "scaler_type"),
            "scale_by_sector": ("scaling", "scale_by_sector"),
            "scaling_columns": ("scaling", "target_columns"),
            "exclude_price_columns_from_scaling": (
                "scaling",
                "exclude_price_columns",
            ),
            "apply_feature_engineering": ("feature_engineering", "enabled"),
            "feature_preset": ("feature_engineering", "preset"),
            "feature_categories": ("feature_engineering", "categories"),
            "apply_feature_selection": ("feature_selection", "enabled"),
            "feature_selection_method": ("feature_selection", "method"),
            "importance_threshold": (
                "feature_selection",
                "min_importance_threshold",
            ),
            "correlation_threshold": (
                "feature_selection",
                "max_correlation_threshold",
            ),
            "feature_selection_categories": (
                "feature_selection",
                "categories",
            ),
            "compute_valuation_metrics": (
                "financial_metrics",
                "compute_valuation_metrics",
            ),
            "compute_profitability_metrics": (
                "financial_metrics",
                "compute_profitability_metrics",
            ),
            "compute_growth_metrics": (
                "financial_metrics",
                "compute_growth_metrics",
            ),
            "compute_leverage_metrics": (
                "financial_metrics",
                "compute_leverage_metrics",
            ),
            "compute_target_vs_price": (
                "financial_metrics",
                "compute_target_vs_price_metrics",
            ),
            "handle_sector_specific_metrics": (
                "financial_metrics",
                "compute_sector_specific_metrics",
            ),
            "generate_quality_alerts": (
                "financial_metrics",
                "generate_quality_alerts",
            ),
            "generate_metrics_dashboard": (
                "financial_metrics",
                "generate_metrics_dashboard",
            ),
            "output_subdir": ("financial_metrics", "output_directory"),
        }

        for key, value in legacy_kwargs.items():
            if key in legacy_map and value is not None:
                section, attr = legacy_map[key]
                setattr(getattr(self, section), attr, value)

    # Backward compatibility properties (delegate to nested configs)
    @property
    def normalize_columns(self) -> bool:
        return self.extraction.normalize_column_names

    @normalize_columns.setter
    def normalize_columns(self, value: bool) -> None:
        self.extraction.normalize_column_names = value

    @property
    def limit(self) -> Optional[int]:
        return self.extraction.limit

    @limit.setter
    def limit(self, value: Optional[int]) -> None:
        self.extraction.limit = value

    @property
    def validate_schema(self) -> bool:
        return self.validation.validate_schema

    @validate_schema.setter
    def validate_schema(self, value: bool) -> None:
        self.validation.validate_schema = value

    @property
    def require_target(self) -> bool:
        return self.validation.require_target_column

    @require_target.setter
    def require_target(self, value: bool) -> None:
        self.validation.require_target_column = value

    @property
    def validate_quality(self) -> bool:
        return self.validation.validate_quality

    @validate_quality.setter
    def validate_quality(self, value: bool) -> None:
        self.validation.validate_quality = value

    @property
    def validate_pipeline(self) -> bool:
        return self.validation.validate_pipeline

    @validate_pipeline.setter
    def validate_pipeline(self, value: bool) -> None:
        self.validation.validate_pipeline = value

    @property
    def drop_invalid_rows(self) -> bool:
        return self.validation.drop_rows_with_missing_critical_fields

    @drop_invalid_rows.setter
    def drop_invalid_rows(self, value: bool) -> None:
        self.validation.drop_rows_with_missing_critical_fields = value

    @property
    def custom_validators(self) -> List[Any]:
        return self.validation.custom_validators

    @custom_validators.setter
    def custom_validators(self, value: List[Any]) -> None:
        self.validation.custom_validators = value

    @property
    def apply_dtype_casting(self) -> bool:
        return self.dtype_casting.apply_dtype_casting

    @apply_dtype_casting.setter
    def apply_dtype_casting(self, value: bool) -> None:
        self.dtype_casting.apply_dtype_casting = value

    @property
    def track_dtype_diagnostics(self) -> bool:
        return self.dtype_casting.track_diagnostics

    @track_dtype_diagnostics.setter
    def track_dtype_diagnostics(self, value: bool) -> None:
        self.dtype_casting.track_diagnostics = value

    @property
    def use_semantic_column_classification(self) -> bool:
        return self.semantic_classification.enabled

    @use_semantic_column_classification.setter
    def use_semantic_column_classification(self, value: bool) -> None:
        self.semantic_classification.enabled = value

    @property
    def preserve_price_columns(self) -> bool:
        return self.semantic_classification.preserve_price_columns

    @preserve_price_columns.setter
    def preserve_price_columns(self, value: bool) -> None:
        self.semantic_classification.preserve_price_columns = value

    @property
    def apply_imputation(self) -> bool:
        return self.imputation.apply_imputation

    @apply_imputation.setter
    def apply_imputation(self, value: bool) -> None:
        self.imputation.apply_imputation = value

    @property
    def imputation_strategy(self) -> str:
        return self.imputation.strategy

    @imputation_strategy.setter
    def imputation_strategy(self, value: str) -> None:
        self.imputation.strategy = value

    @property
    def knn_neighbors(self) -> int:
        return self.imputation.knn_neighbors

    @knn_neighbors.setter
    def knn_neighbors(self, value: int) -> None:
        self.imputation.knn_neighbors = value

    @property
    def imputation_sector_column(self) -> str:
        return self.imputation.sector_column

    @imputation_sector_column.setter
    def imputation_sector_column(self, value: str) -> None:
        self.imputation.sector_column = value

    @property
    def imputation_price_column(self) -> str:
        return self.imputation.reference_price_column

    @imputation_price_column.setter
    def imputation_price_column(self, value: str) -> None:
        self.imputation.reference_price_column = value

    @property
    def handle_categorical_imputation(self) -> bool:
        return self.imputation.impute_categorical_columns

    @handle_categorical_imputation.setter
    def handle_categorical_imputation(self, value: bool) -> None:
        self.imputation.impute_categorical_columns = value

    @property
    def handle_datetime_imputation(self) -> bool:
        return self.imputation.impute_datetime_columns

    @handle_datetime_imputation.setter
    def handle_datetime_imputation(self, value: bool) -> None:
        self.imputation.impute_datetime_columns = value

    @property
    def apply_log_transforms(self) -> bool:
        return self.semantic_transform.apply_log_transforms

    @apply_log_transforms.setter
    def apply_log_transforms(self, value: bool) -> None:
        self.semantic_transform.apply_log_transforms = value

    @property
    def log_transform_method(self) -> Literal["log1p", "signed_log"]:
        return self.semantic_transform.log_transform_method

    @log_transform_method.setter
    def log_transform_method(self, value: Literal["log1p", "signed_log"]) -> None:
        self.semantic_transform.log_transform_method = value

    @property
    def log_transform_columns(self) -> Optional[List[str]]:
        return self.semantic_transform.log_transform_target_columns

    @log_transform_columns.setter
    def log_transform_columns(self, value: Optional[List[str]]) -> None:
        self.semantic_transform.log_transform_target_columns = value

    @property
    def log_transform_market_values(self) -> bool:
        return self.semantic_transform.log_transform_market_values

    @log_transform_market_values.setter
    def log_transform_market_values(self, value: bool) -> None:
        self.semantic_transform.log_transform_market_values = value

    @property
    def exclude_ratios_from_winsorization(self) -> bool:
        return self.semantic_transform.exclude_ratios_from_winsorization

    @exclude_ratios_from_winsorization.setter
    def exclude_ratios_from_winsorization(self, value: bool) -> None:
        self.semantic_transform.exclude_ratios_from_winsorization = value

    @property
    def exclude_percentages_from_winsorization(self) -> bool:
        return self.semantic_transform.exclude_percentages_from_winsorization

    @exclude_percentages_from_winsorization.setter
    def exclude_percentages_from_winsorization(self, value: bool) -> None:
        self.semantic_transform.exclude_percentages_from_winsorization = value

    @property
    def exclude_counts_from_scaling(self) -> bool:
        return self.semantic_transform.exclude_counts_from_scaling

    @exclude_counts_from_scaling.setter
    def exclude_counts_from_scaling(self, value: bool) -> None:
        self.semantic_transform.exclude_counts_from_scaling = value

    @property
    def sanitize_data(self) -> bool:
        return self.sanitization.sanitize_data

    @sanitize_data.setter
    def sanitize_data(self, value: bool) -> None:
        self.sanitization.sanitize_data = value

    @property
    def apply_scaling(self) -> bool:
        return self.scaling.enabled

    @apply_scaling.setter
    def apply_scaling(self, value: bool) -> None:
        self.scaling.enabled = value

    @property
    def scaler_type(self) -> Literal["robust", "standard", "minmax"]:
        return self.scaling.scaler_type

    @scaler_type.setter
    def scaler_type(self, value: Literal["robust", "standard", "minmax"]) -> None:
        self.scaling.scaler_type = value

    @property
    def scale_by_sector(self) -> bool:
        return self.scaling.scale_by_sector

    @scale_by_sector.setter
    def scale_by_sector(self, value: bool) -> None:
        self.scaling.scale_by_sector = value

    @property
    def scaling_columns(self) -> Optional[List[str]]:
        return self.scaling.target_columns

    @scaling_columns.setter
    def scaling_columns(self, value: Optional[List[str]]) -> None:
        self.scaling.target_columns = value

    @property
    def exclude_price_columns_from_scaling(self) -> bool:
        return self.scaling.exclude_price_columns

    @exclude_price_columns_from_scaling.setter
    def exclude_price_columns_from_scaling(self, value: bool) -> None:
        self.scaling.exclude_price_columns = value

    @property
    def apply_feature_engineering(self) -> bool:
        return self.feature_engineering.enabled

    @apply_feature_engineering.setter
    def apply_feature_engineering(self, value: bool) -> None:
        self.feature_engineering.enabled = value

    @property
    def feature_preset(self) -> str:
        return self.feature_engineering.preset

    @feature_preset.setter
    def feature_preset(self, value: str) -> None:
        self.feature_engineering.preset = value

    @property
    def feature_categories(self) -> Optional[List[str]]:
        return self.feature_engineering.categories

    @feature_categories.setter
    def feature_categories(self, value: Optional[List[str]]) -> None:
        self.feature_engineering.categories = value

    @property
    def apply_feature_selection(self) -> bool:
        return self.feature_selection.enabled

    @apply_feature_selection.setter
    def apply_feature_selection(self, value: bool) -> None:
        self.feature_selection.enabled = value

    @property
    def feature_selection_method(self) -> Literal["mutual_info", "correlation", "both"]:
        return self.feature_selection.method

    @feature_selection_method.setter
    def feature_selection_method(
        self, value: Literal["mutual_info", "correlation", "both"]
    ) -> None:
        self.feature_selection.method = value

    @property
    def importance_threshold(self) -> float:
        return self.feature_selection.min_importance_threshold

    @importance_threshold.setter
    def importance_threshold(self, value: float) -> None:
        self.feature_selection.min_importance_threshold = value

    @property
    def correlation_threshold(self) -> float:
        return self.feature_selection.max_correlation_threshold

    @correlation_threshold.setter
    def correlation_threshold(self, value: float) -> None:
        self.feature_selection.max_correlation_threshold = value

    @property
    def feature_selection_categories(self) -> Optional[List[str]]:
        return self.feature_selection.categories

    @feature_selection_categories.setter
    def feature_selection_categories(self, value: Optional[List[str]]) -> None:
        self.feature_selection.categories = value

    @property
    def compute_valuation_metrics(self) -> bool:
        return self.financial_metrics.compute_valuation_metrics

    @compute_valuation_metrics.setter
    def compute_valuation_metrics(self, value: bool) -> None:
        self.financial_metrics.compute_valuation_metrics = value

    @property
    def compute_profitability_metrics(self) -> bool:
        return self.financial_metrics.compute_profitability_metrics

    @compute_profitability_metrics.setter
    def compute_profitability_metrics(self, value: bool) -> None:
        self.financial_metrics.compute_profitability_metrics = value

    @property
    def compute_growth_metrics(self) -> bool:
        return self.financial_metrics.compute_growth_metrics

    @compute_growth_metrics.setter
    def compute_growth_metrics(self, value: bool) -> None:
        self.financial_metrics.compute_growth_metrics = value

    @property
    def compute_leverage_metrics(self) -> bool:
        return self.financial_metrics.compute_leverage_metrics

    @compute_leverage_metrics.setter
    def compute_leverage_metrics(self, value: bool) -> None:
        self.financial_metrics.compute_leverage_metrics = value

    @property
    def compute_target_vs_price(self) -> bool:
        return self.financial_metrics.compute_target_vs_price_metrics

    @compute_target_vs_price.setter
    def compute_target_vs_price(self, value: bool) -> None:
        self.financial_metrics.compute_target_vs_price_metrics = value

    @property
    def handle_sector_specific_metrics(self) -> bool:
        return self.financial_metrics.compute_sector_specific_metrics

    @handle_sector_specific_metrics.setter
    def handle_sector_specific_metrics(self, value: bool) -> None:
        self.financial_metrics.compute_sector_specific_metrics = value

    @property
    def generate_quality_alerts(self) -> bool:
        return self.financial_metrics.generate_quality_alerts

    @generate_quality_alerts.setter
    def generate_quality_alerts(self, value: bool) -> None:
        self.financial_metrics.generate_quality_alerts = value

    @property
    def generate_metrics_dashboard(self) -> bool:
        return self.financial_metrics.generate_metrics_dashboard

    @generate_metrics_dashboard.setter
    def generate_metrics_dashboard(self, value: bool) -> None:
        self.financial_metrics.generate_metrics_dashboard = value

    @property
    def output_subdir(self) -> str:
        return self.financial_metrics.output_directory

    @output_subdir.setter
    def output_subdir(self, value: str) -> None:
        self.financial_metrics.output_directory = value


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

    source_type: str
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
                100 * self.features_removed_by_selection / self.features_before_selection
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
            status_icon = "✓" if self.schema_alignment_score >= 0.90 else "⚠"
            recognized_info = (
                f", recognized: {self.recognized_columns_count}"
                if self.recognized_columns_count > 0
                else ""
            )
            schema_validation_info = (
                f"\n  Schema Validation: {status_icon} "
                f"(alignment: {self.schema_alignment_score:.2%}, "
                f"unknown extra: {self.unknown_columns_count}, "
                f"missing required: {self.missing_expected_columns_count}, "
                f"dtype mismatches: {self.dtype_mismatches_count}"
                f"{recognized_info})"
            )

        return (
            f"ETL Pipeline Summary:\n"
            f"  Source: {self.source_type}\n"
            f"  Duration: {self.total_duration:.2f}s "
            f"(extract: {self.extract_duration:.2f}s, "
            f"transform: {self.transform_duration:.2f}s, "
            f"load: {self.load_duration:.2f}s)\n"
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
            f"Validation: {self.validation_score:.3f}"
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
        if self.config.extraction.normalize_column_names:
            logger.info("Stage 1: Normalizing column names")
            result = normalize_columns(result, preserve_schema=True)

        # Stage 1.5: Apply dtype casting (NEW - critical for CSV data)
        if self.config.dtype_casting.apply_dtype_casting:
            logger.info("Stage 1.5: Applying schema-aware dtype casting")
            try:
                result, dtype_diagnostics = detect_and_cast_dtypes(result, schema=COLUMN_SCHEMA)

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
                    if self.config.dtype_casting.track_diagnostics:
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
        if self.config.semantic_classification.enabled:
            result = self._apply_semantic_classification(result)
            if self.metrics:
                self.metrics.stages_executed.append("semantic_classification")

        # Stage 2: Validate schema
        if self.config.validation.validate_schema:
            logger.info("Stage 2: Validating schema")
            is_valid, errors = validate_schema(
                result, require_target=self.config.validation.require_target_column
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
        if self.config.validation.drop_rows_with_missing_critical_fields:
            logger.info("Stage 3: Dropping rows with missing critical fields")
            initial_rows = len(result)
            for col in ["ticker", "sector", "last_price"]:
                if col in result.columns:
                    result = result[~result[col].isna()]
            dropped_rows = initial_rows - len(result)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing critical fields")

        # Stage 4: Sanitize data
        if self.config.sanitization.sanitize_data:
            logger.info("Stage 4: Sanitizing data (inf, nan, extremes)")
            result, sanitize_stats = sanitize_dataframe_with_logging(result, return_stats=True)

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
        if self.config.imputation.apply_imputation:
            logger.info(f"Stage 5: Applying {self.config.imputation.strategy} imputation strategy")

            # Track missing values before imputation
            missing_before = result.isna().sum().sum()
            if self.metrics:
                self.metrics.missing_values_before_imputation = int(missing_before)
                self.metrics.imputation_strategy = self.config.imputation.strategy

            if self.config.imputation.strategy == "6step":
                result = apply_enhanced_imputation_strategy_6step(
                    result,
                    sector_column=self.config.imputation.sector_column,
                    n_neighbors=self.config.imputation.knn_neighbors,
                    price_column=self.config.imputation.reference_price_column,
                    handle_categoricals=self.config.imputation.impute_categorical_columns,
                    handle_dates=self.config.imputation.impute_datetime_columns,
                    apply_categorical_encoding=self.config.imputation.apply_categorical_encoding,
                )
            elif self.config.imputation.strategy == "4step":
                # Backward compatibility - numeric only
                result = apply_enhanced_imputation_strategy_6step(
                    result,
                    sector_column=self.config.imputation.sector_column,
                    n_neighbors=self.config.imputation.knn_neighbors,
                    price_column=self.config.imputation.reference_price_column,
                    handle_categoricals=False,
                    handle_dates=False,
                    apply_categorical_encoding=False,
                )
            elif self.config.imputation.strategy == "median_only":
                # Simple median fallback
                result = apply_median_imputation(
                    result, price_column=self.config.imputation.reference_price_column
                )

            # Track missing values after imputation and validate completeness
            missing_after = result.isna().sum().sum()
            validation = validate_imputation_completeness(result)

            if self.metrics:
                self.metrics.missing_values_after_imputation = int(missing_after)
                self.metrics.imputation_completeness = validation["is_complete"]
                self.metrics.date_columns_ready = validation["ready_for_temporal_features"]
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

        # Stage 6: Apply semantic-aware transformations (log transforms)
        result = self._apply_semantic_transformations(result)
        if self.metrics:
            self.metrics.stages_executed.append("semantic_transformations")

        # Stage 7: Apply feature scaling (NEW)
        if self.config.scaling.enabled:
            logger.info(f"Stage 7: Applying {self.config.scaling.scaler_type} scaling")

            exclude_counts = self.config.exclude_counts_from_scaling

            # Determine columns to scale
            if self.config.scaling.target_columns:
                # User-specified columns
                columns_to_scale = self.config.scaling.target_columns
                logger.info(f"Scaling user-specified {len(columns_to_scale)} columns")
            else:
                # Auto-detect using column semantics (excludes PRICE columns by default)
                columns_to_scale = get_scalable_columns(result.columns.tolist())
                logger.info(
                    f"Auto-detected {len(columns_to_scale)} scalable columns "
                    f"(excluded {len([c for c in result.columns if c.lower() in PRICE_COLUMNS])} price columns)"
                )

            # Safety check: Verify no price columns in scaling list
            if self.config.scaling.exclude_price_columns:
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
                        scaler_type=self.config.scaling.scaler_type,
                        by_sector=self.config.scaling.scale_by_sector,
                        exclude_price_columns=self.config.scaling.exclude_price_columns,
                        exclude_count_columns=exclude_counts,
                    )

                    # Track scaling in metrics
                    if self.metrics:
                        self.metrics.scaling_applied = True
                        self.metrics.scaler_type = self.config.scaling.scaler_type
                        self.metrics.scaled_columns_count = len(columns_to_scale)
                        self.metrics.price_columns_protected = (
                            self.config.scaling.exclude_price_columns
                        )
                        self.metrics.stages_executed.append("scaling")

                    logger.info(
                        f"Scaling complete: {len(columns_to_scale)} columns scaled using "
                        f"{self.config.scaling.scaler_type} scaler (sector-aware: {self.config.scaling.scale_by_sector})"
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

        if self.config.financial_metrics.compute_valuation_metrics:
            logger.info("Stage 8a: Computing valuation metrics")
            result = compute_valuation_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.valuation_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("valuation_metrics")
            initial_cols = set(result.columns)

        if self.config.financial_metrics.compute_profitability_metrics:
            logger.info("Stage 8b: Computing profitability metrics")
            result = compute_profitability_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.profitability_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("profitability_metrics")
            initial_cols = set(result.columns)

        if self.config.financial_metrics.compute_growth_metrics:
            logger.info("Stage 8c: Computing growth metrics")
            result = compute_growth_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.growth_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("growth_metrics")
            initial_cols = set(result.columns)

        if self.config.financial_metrics.compute_leverage_metrics:
            logger.info("Stage 8d: Computing leverage metrics")
            result = compute_leverage_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.leverage_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("leverage_metrics")
            initial_cols = set(result.columns)

        if self.config.financial_metrics.compute_target_vs_price_metrics:
            logger.info("Stage 8e: Computing target vs price metrics")
            result = compute_target_vs_price_metrics(result)
            new_cols = set(result.columns) - initial_cols
            if self.metrics:
                self.metrics.target_vs_price_metrics_added = len(new_cols)
                self.metrics.stages_executed.append("target_vs_price_metrics")
            initial_cols = set(result.columns)

        if self.config.financial_metrics.compute_sector_specific_metrics:
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
        if self.config.feature_engineering.enabled:
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
                        sector_column=self.config.imputation.sector_column,
                        n_neighbors=self.config.imputation.knn_neighbors,
                        price_column=self.config.imputation.reference_price_column,
                        handle_categoricals=self.config.imputation.impute_categorical_columns,
                        handle_dates=self.config.imputation.impute_datetime_columns,
                        apply_categorical_encoding=self.config.imputation.apply_categorical_encoding,
                    )
                    if self.metrics:
                        self.metrics.stages_executed.append("post_feature_imputation_6step")
                except Exception as e:
                    logger.warning(
                        f"Post-feature 6-step imputation failed ({e}); falling back to median imputation"
                    )
                    result = apply_median_imputation(
                        result, price_column=self.config.imputation.reference_price_column
                    )
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
                    dt_cols = result.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
                    if len(dt_cols) > 0:
                        result[dt_cols] = result[dt_cols].fillna(pd.Timestamp("1970-01-01"))

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
                                    result[col] = result[col].cat.add_categories(["UNKNOWN"])
                                # Now safely fill NaN with 'UNKNOWN'
                                result[col] = result[col].fillna("UNKNOWN")
                            else:
                                # For object/string columns, direct fillna works
                                result[col] = result[col].fillna("UNKNOWN")

                    if self.metrics:
                        self.metrics.stages_executed.append("post_feature_imputation_final_pass")

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
        if self.config.feature_selection.enabled:
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
                        importance_threshold=self.config.feature_selection.min_importance_threshold,
                        correlation_threshold=self.config.feature_selection.max_correlation_threshold,
                        method=self.config.feature_selection.method,
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
                        self.metrics.warnings.append("Feature selection skipped due to error")
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
        if self.config.validation.validate_schema_alignment:
            logger.info("Stage 11: Validating schema alignment")
            schema_validation = self._validate_schema_alignment(result)

            if self.metrics:
                self.metrics.schema_alignment_score = schema_validation.get("alignment_score", 1.0)
                self.metrics.unknown_columns_count = len(
                    schema_validation.get("unknown_columns", [])
                )
                self.metrics.missing_expected_columns_count = len(
                    schema_validation.get("missing_expected_columns", [])
                )
                self.metrics.dtype_mismatches_count = len(
                    schema_validation.get("dtype_mismatches", {})
                )
                self.metrics.recognized_columns_count = int(
                    schema_validation.get("recognized_columns_count", 0)
                )
                self.metrics.allowlisted_engineered_columns_count = len(
                    schema_validation.get("allowlisted_engineered", [])
                )
                self.metrics.stages_executed.append("schema_validation")

                # Add warnings for significant schema issues
                alignment_threshold = self.config.validation.schema_alignment_threshold
                if schema_validation.get("alignment_score", 1.0) < alignment_threshold:
                    self.metrics.warnings.append(
                        f"Schema alignment below {alignment_threshold:.0%}: {schema_validation.get('alignment_score', 1.0):.2%}"
                    )
                if len(schema_validation.get("unknown_columns", [])) > 50:
                    self.metrics.warnings.append(
                        f"Found {len(schema_validation.get('unknown_columns', []))} unknown columns"
                    )

        # Stage 12: Quality validation (Stage 9 in requirements)
        if self.config.validation.validate_quality:
            result = self._validate_quality(result)
            if self.metrics:
                self.metrics.stages_executed.append("quality_validation")

        logger.info(f"Transformation complete: {len(result)} rows, {len(result.columns)} columns")
        return result

    def _validate_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 9: Quality validation with alerts and metrics dashboard generation.

        Generates data quality alerts and metrics dashboard JSON files
        when output_dir is configured (Section 8.6).
        """
        logger.info("Stage 9: Validating data quality")

        # Calculate quality score
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        quality_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

        if self.metrics:
            self.metrics.quality_score = quality_score

        # Generate quality alerts if output directory is configured
        if self.config.financial_metrics.output_directory:
            output_path = Path(self.config.financial_metrics.output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate data quality alerts (Section 19)
            alerts = generate_data_quality_alerts(
                df=df,
            )
            logger.info(f"Generated {len(alerts)} quality alerts")

            # Save alerts to JSON
            try:
                with open(output_path / "quality_alerts.json", "w") as f:
                    json.dump(to_jsonable(alerts), f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save quality alerts: {e}")

            if self.metrics:
                self.metrics.quality_alerts_generated = len(alerts)

            # Generate metrics dashboard (Section 20.2)
            dashboard = generate_metrics_dashboard(
                df=df,
            )

            # Enrich dashboard with semantic classification
            if "semantic_classification" not in dashboard:
                dashboard["semantic_classification"] = {}

            if self.metrics:
                dashboard["semantic_classification"].update(
                    {
                        "price_columns": self.metrics.price_columns_count,
                        "market_value_columns": self.metrics.market_value_columns_count,
                        "ratio_columns": self.metrics.ratio_columns_count,
                        "percentage_columns": self.metrics.percentage_columns_count,
                        "count_columns": self.metrics.count_columns_count,
                    }
                )
            dashboard["model_version"] = getattr(self.config, "model_version", "unknown")

            # Save dashboard to JSON
            try:
                with open(output_path / "metrics_dashboard.json", "w") as f:
                    json.dump(to_jsonable(dashboard), f, indent=4)
                logger.info(f"Metrics dashboard saved to {output_path / 'metrics_dashboard.json'}")
            except Exception as e:
                logger.error(f"Failed to save metrics dashboard: {e}")

        # Validation checks
        validation_score = 1.0
        issues = []

        # Check for remaining missing values (should be 0 after 6-step imputation)
        if missing_cells > 0:
            validation_score -= 0.1
            issues.append(f"Missing values remain: {missing_cells}")

        # Check price column preservation
        preserved_price_cols = [col for col in PRICE_COLUMNS if col in df.columns]
        if len(preserved_price_cols) < 21:
            logger.warning(f"Only {len(preserved_price_cols)}/21 price columns present")

        if self.metrics:
            self.metrics.validation_score = validation_score

        return df

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

        if self.config.validation.validate_pipeline:
            pipeline_metrics = perform_early_pipeline_validation(df)
            quality_metrics["pipeline_validation"] = pipeline_metrics

        return quality_metrics

    def _apply_semantic_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Semantic column classification (price, market_value, ratio, percentage, count).

        Uses constants from column_semantics.py to categorize columns for downstream
        transformation decisions (Section 8.5.1).
        """
        logger.info("Stage 1.6: Applying semantic column classification")

        # Classify columns using both hardcoded sets and pattern-based inference
        classification_result = classify_columns(df.columns.tolist())

        # Track column counts by semantic category using imported constants
        price_cols = [col for col in PRICE_COLUMNS if col in df.columns]
        market_value_cols = [col for col in MARKET_VALUE_COLUMNS if col in df.columns]
        ratio_cols = [col for col in RATIO_COLUMNS if col in df.columns]
        percentage_cols = [col for col in PERCENTAGE_COLUMNS if col in df.columns]
        count_cols = [col for col in COUNT_COLUMNS if col in df.columns]

        # Update metrics with semantic classification counts
        if self.metrics:
            self.metrics.price_columns_count = len(price_cols)
            self.metrics.market_value_columns_count = len(market_value_cols)
            self.metrics.ratio_columns_count = len(ratio_cols)
            self.metrics.percentage_columns_count = len(percentage_cols)
            self.metrics.count_columns_count = len(count_cols)
            self.metrics.semantic_classification_applied = True

        # Use schema utility functions for additional classification
        categorical_cols = list_categorical_cols()
        numeric_feature_cols = list_numeric_feature_cols()

        # Validate against schema expectations
        expected_categorical = [col for col in categorical_cols if col in df.columns]
        expected_numeric = [col for col in numeric_feature_cols if col in df.columns]

        logger.info(
            f"Column classification: price={len(price_cols)}, "
            f"market_value={len(market_value_cols)}, ratio={len(ratio_cols)}, "
            f"percentage={len(percentage_cols)}, count={len(count_cols)}, "
            f"other={len(classification_result.get('other', set()))}"
        )
        logger.info(f"Identified {len(market_value_cols)} columns for log-transform")

        return df

    def _apply_semantic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 5: Semantic-aware transformations (log-transforms for market values).

        Applies log-transforms to MARKET_VALUE_COLUMNS while preserving
        PRICE_COLUMNS, RATIO_COLUMNS, and PERCENTAGE_COLUMNS (Section 8.5.3).
        """
        logger.info("Stage 5: Applying semantic-aware transformations")

        if (
            not self.config.semantic_transform.apply_log_transforms
            and not self.config.semantic_transform.log_transform_market_values
        ):
            return df

        # Get columns eligible for log-transform using semantic constants
        log_transform_cols = []
        if (
            self.config.semantic_transform.apply_log_transforms
            or self.config.semantic_transform.log_transform_market_values
        ):
            log_transform_cols.extend(
                [
                    col
                    for col in MARKET_VALUE_COLUMNS
                    if col in df.columns and col not in PRICE_COLUMNS
                ]
            )

        if self.config.semantic_transform.log_transform_target_columns:
            for col in self.config.semantic_transform.log_transform_target_columns:
                if col in df.columns and col not in log_transform_cols:
                    log_transform_cols.append(col)

        transformed_count = 0
        skipped_negative = 0

        for col in log_transform_cols:
            # Check for negative values (skip log-transform if present)
            if (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                logger.warning(
                    f"Skipped log-transform for '{col}': "
                    f"{negative_count} negative values present (added applicability flag)"
                )
                df[f"log_{col}_applicable"] = df[col] >= 0
                skipped_negative += 1
                continue

            # Apply log1p transform (handles zeros)
            log_col_name = f"log_{col}"
            df[log_col_name] = np.log1p(df[col].clip(lower=0))
            transformed_count += 1

        if self.metrics:
            self.metrics.log_transformed_columns = transformed_count
            self.metrics.log_transforms_skipped = skipped_negative

        logger.info(
            f"Log-transformed {transformed_count} columns, "
            f"skipped {skipped_negative} due to negative values"
        )

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
        if not self.config.semantic_classification.enabled:
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
        if not self.config.semantic_classification.enabled:
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
        if not self.config.feature_engineering.enabled:
            return df

        logger.info(
            f"Applying feature engineering with preset: {self.config.feature_engineering.preset}"
        )

        try:
            original_cols = set(df.columns)

            # Use the unified build_features API
            df_with_features = build_features(
                df,
                preset=self.config.feature_engineering.preset,
            )

            # Apply earnings analytics if enabled
            if self.config.feature_engineering.engineer_earnings_analytics:
                logger.info(
                    "Applying earnings analytics features (Estimated vs. Actual and GAAP vs. Adjusted)"
                )
                from finance_ml.ml_workflow.features.advanced import (
                    engineer_estimated_vs_actual_analytics,
                    engineer_gaap_vs_adjusted_analytics,
                )

                df_with_features = engineer_estimated_vs_actual_analytics(df_with_features)
                df_with_features = engineer_gaap_vs_adjusted_analytics(df_with_features)

            new_cols = set(df_with_features.columns) - original_cols
            logger.info(f"Added {len(new_cols)} engineered features")

            # Update metrics
            if self.metrics:
                self.metrics.feature_engineering_applied = True
                self.metrics.feature_preset_used = self.config.feature_engineering.preset
                self.metrics.features_added = len(new_cols)
                self.metrics.columns_output = len(df_with_features.columns)

            return df_with_features

        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}, returning original DataFrame")
            return df

    def _validate_schema_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame columns against the schema registry with scope-aware rules.

        This validator is designed to run after feature engineering (Stage 9) and
        optional feature selection (Stage 10). As a result:

        - The DataFrame may legitimately contain engineered feature columns that are
          not part of the raw source schema.
        - Not all source-schema columns are guaranteed to exist in every dataset
          variant (CSV extracts may lag schema evolution).

        Therefore, we evaluate alignment primarily on the *required ETL schema*
        (Phase 9.1 critical inputs) and on whether all columns are either:
          (a) registered in COLUMN_SCHEMA, or
          (b) allowlisted engineered features (Phase 9.3 outputs).

        Checks for:
        - Unknown columns (present in df but not in schema and not allowlisted)
        - Missing expected columns (missing from the required ETL schema)
        - Dtype mismatches (for columns that are in COLUMN_SCHEMA)

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results:
            - unknown_columns: List of columns not in schema and not allowlisted
            - missing_expected_columns: Missing required ETL schema columns
            - dtype_mismatches: Dict of columns with dtype mismatches
            - alignment_score: Float [0.0-1.0] indicating schema alignment quality
        """
        from finance_ml.ml_workflow.data.schema import (
            COLUMN_SCHEMA,
            get_expected_dtype,
            list_required_schema_columns_for_etl,
        )

        df_cols = set(df.columns)
        schema_cols = set(COLUMN_SCHEMA.keys())

        # Allowlist: Phase 9.3 engineered feature outputs (registered separately for EDA/reporting)
        engineered_allowlist: set[str] = set()
        try:
            from finance_ml.ml_workflow.eda.phase93_categories import (
                PHASE93_FEATURE_CATEGORIES,
            )

            for _cat, feats in PHASE93_FEATURE_CATEGORIES.items():
                engineered_allowlist.update(feats)
        except Exception:
            # If registry isn't available, fall back to strict behavior.
            engineered_allowlist = set()

        def _is_allowlisted_engineered(col: str) -> bool:
            """Return True if `col` is a legitimate ETL/feature-engineering output."""

            # 1) Explicit allowlist from Phase 9.3 registry
            if col in engineered_allowlist:
                return True

            # 2) Known meta-feature prefixes
            if col.startswith("event_prob_"):
                return True

            # 3) ETL-generated column conventions
            # Log transforms: only allow when base is a schema column.
            if col.startswith("log_") and col[4:] in schema_cols:
                return True

            # Conditional metric applicability flags: allow when base metric is known.
            if col.endswith("_applicable"):
                base = col[: -len("_applicable")]
                if base in schema_cols or base in CONDITIONAL_METRICS:
                    return True
                # Also allow log_* applicability flags
                if base.startswith("log_"):
                    return True

            # Growth / YoY metrics produced by metrics ETL.
            if col.endswith("_growth") or col.endswith("_yoy"):
                return True

            # Sector interaction features (dynamically generated).
            if col.startswith("sector_") and "_x_" in col:
                return True

            # Sector-relative features (percentile, zscore, vs_sector_median, vs_sector_top_quartile)
            if "_sector_percentile" in col or "_sector_zscore" in col:
                return True
            if "_vs_sector_median" in col or "_vs_sector_top_quartile" in col:
                return True

            # Squared/polynomial features
            if col.endswith("_squared") or col.endswith("_cubed"):
                return True

            # FTE (full-time employee) derived metrics
            if col.startswith("fte_"):
                return True

            # Momentum features
            if "_momentum_" in col or col.endswith("_momentum"):
                return True

            # Volatility features
            if col.endswith("_volatility"):
                return True

            # Reference date column
            if col == "_reference_date":
                return True

            # Common semantic/derived suffixes used by the pipeline.
            if col.endswith(("_ratio", "_pct", "_margin")):
                tokens = (
                    "p_",
                    "ev_",
                    "market_",
                    "gross_",
                    "operating_",
                    "net_",
                    "revenue",
                    "ebitda",
                    "assets",
                    "equity",
                    "cash",
                    "debt",
                    "dividend",
                    "price_",
                    "target_",
                    "efficiency",
                    "payout",
                )
                if any(t in col for t in tokens):
                    return True

            return False

        # Identify unknown columns (in df but neither in schema nor allowlisted engineered features)
        unknown_cols = sorted(
            col
            for col in df_cols
            if (col not in schema_cols and not _is_allowlisted_engineered(col))
        )

        # Expected columns: required raw ETL inputs (scope-aware, source-agnostic)
        expected_cols = set(list_required_schema_columns_for_etl(include_extended_financials=False))
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
        # 1) Required coverage: are all required raw columns present?
        required_coverage = (
            (len(expected_cols) - len(missing_cols)) / len(expected_cols) if expected_cols else 1.0
        )
        # 2) Recognition rate: what fraction of produced columns are known (schema or allowlisted engineered)?
        recognized_cols = [
            c for c in df_cols if (c in schema_cols) or _is_allowlisted_engineered(c)
        ]
        recognition_rate = len(recognized_cols) / len(df_cols) if df_cols else 1.0

        # Final alignment score: both must be strong
        alignment_score = required_coverage * recognition_rate

        allowlisted_engineered = sorted(
            c for c in df_cols if (c not in schema_cols) and _is_allowlisted_engineered(c)
        )

        validation_result = {
            "unknown_columns": unknown_cols,
            "missing_expected_columns": missing_cols,
            "dtype_mismatches": dtype_mismatches,
            "alignment_score": alignment_score,
            "required_coverage": required_coverage,
            "recognition_rate": recognition_rate,
            "total_columns": len(df_cols),
            "schema_columns": len(schema_cols),
            "common_columns": len(common_cols),
            "recognized_columns_count": len(recognized_cols),
            "allowlisted_engineered": allowlisted_engineered,
        }

        # Log validation results
        if unknown_cols:
            logger.warning(
                "Found %d unknown columns not in COLUMN_SCHEMA or allowlist",
                len(unknown_cols),
            )
            if len(unknown_cols) <= 20:
                logger.warning("Unknown columns: %s", unknown_cols)
            else:
                logger.warning("Unknown columns (first 20): %s", unknown_cols[:20])

        if missing_cols:
            logger.info(f"Missing {len(missing_cols)} required ETL schema columns")
            if len(missing_cols) <= 10:
                logger.info(f"Missing columns: {missing_cols}")

        if dtype_mismatches:
            logger.warning(f"Found {len(dtype_mismatches)} dtype mismatches")
            if len(dtype_mismatches) <= 5:
                for col, mismatch in list(dtype_mismatches.items())[:5]:
                    logger.warning(
                        f"  {col}: expected {mismatch['expected']}, got {mismatch['actual']}"
                    )

        logger.info(
            "Schema alignment: %.2f%% (recognized: %d/%d, required coverage: %.2f%%)",
            100.0 * alignment_score,
            len(recognized_cols),
            len(df_cols),
            100.0 * required_coverage,
        )

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

        if validate and self.config.validation.validate_schema_alignment:
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
            >>> print(f"Processing time: {metrics.total_duration:.2f}s")

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
    unified_table: bool = False,
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
    engineer_earnings_analytics: bool = True,  # NEW: Enable earnings analytics by default
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
        feature_preset: Feature engineering preset
            - 'basic': Core ratios, margins, volatility, revenue CAGR (20-30 features)
            - 'momentum': Momentum & technical indicators with RSI, EMA (27 features)
            - 'quality': Accounting quality, distress, composite scores (45+ features)
            - 'standard': Balanced feature set (80-100 features)
            - 'comprehensive': Full advanced feature set (267 features)
            - 'earnings_analytics': Earnings surprises, GAAP vs adjusted (55+ features)
            - 'technical_plus': Technical analysis + valuation timeseries (50+ features)
            - 'dividend_focus': Dividend reliability + capital allocation (30+ features)
            - 'employment_analytics': Employment dynamics + productivity (35+ features)
        feature_categories: Specific feature categories to engineer (optional)
        auto_feature_selection: Enable automated feature selection (default: False)
        importance_threshold: Min importance score to keep feature (default: 0.01)
        correlation_threshold: Max correlation before deduplication (default: 0.95)
        engineer_earnings_analytics: Enable Estimated vs. Actual and GAAP vs. Adjusted
            earnings analytics features (default: True). Only applied when feature_preset
            is 'comprehensive', 'standard', or 'earnings_analytics'.
        config: Optional ETLConfig override
        return_metrics: Whether to return ETLMetrics

    Returns:
        DataFrame with all features, optionally with ETLMetrics

    Feature Coverage by Preset (Phase 9.3 Schema v1.3):
        - basic: 20-30 features (core ratios, margins, volatility, CAGR)
        - momentum: 27 features (momentum, technical indicators, RSI, EMA crossovers)
        - quality: 45+ features (accounting quality, distress, composite scores, analyst quality)
        - standard: 80-100 features (balanced mix: valuation, profitability, growth, sentiment)
        - comprehensive: 267 features (all advanced features including new presets)
        - earnings_analytics: 55+ features (earnings surprises, GAAP vs adjusted, quality flags)
        - technical_plus: 50+ features (technical analysis, valuation timeseries, market sentiment)
        - dividend_focus: 30+ features (dividend reliability, capital allocation, FCF coverage)
        - employment_analytics: 35+ features (employment dynamics, productivity trends)

    Example:
        >>> # Basic usage with comprehensive feature engineering
        >>> df, metrics = etl_with_features(
        ...     source='csv',
        ...     data_dir=Path('data'),
        ...     feature_preset='comprehensive',
        ...     return_metrics=True
        ... )
        >>> print(f"Shape: {df.shape}, Quality: {metrics.quality_score:.3f}")

        >>> # Enhanced earnings analytics workflow
        >>> df, metrics = etl_with_features(
        ...     source='csv',
        ...     data_dir=Path('data'),
        ...     feature_preset='earnings_analytics',
        ...     engineer_earnings_analytics=True,
        ...     return_metrics=True
        ... )
        >>> print(f"Earnings quality features: {metrics.features_added}")

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

    # NEW: Set earnings analytics flag in config
    config.feature_engineering.engineer_earnings_analytics = engineer_earnings_analytics

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
