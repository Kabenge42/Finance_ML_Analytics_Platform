"""ETL configuration dataclasses - extracted from etl.py."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any


@dataclass
class DataExtractionConfig:
    """Configuration for data extraction stage."""

    limit: Optional[int] = None
    normalize_column_names: bool = True
    source_type: Literal["csv", "database", "all_stocks"] = "csv"


@dataclass
class SchemaValidationConfig:
    """Configuration for schema validation."""

    validate_schema: bool = True
    require_target_column: bool = True
    drop_rows_with_missing_critical_fields: bool = False
    validate_quality: bool = True
    validate_pipeline: bool = True
    custom_validators: List[Any] = field(default_factory=list)
    validate_schema_alignment: bool = True
    schema_alignment_threshold: float = 0.85


@dataclass
class DtypeCastingConfig:
    """Configuration for data type casting."""

    apply_dtype_casting: bool = True
    track_diagnostics: bool = True


@dataclass
class SemanticClassificationConfig:
    """Configuration for semantic column classification."""

    enabled: bool = True
    preserve_price_columns: bool = True


@dataclass
class ImputationConfig:
    """Configuration for data imputation."""

    apply_imputation: bool = False
    strategy: str = "6step"
    knn_neighbors: int = 5
    sector_column: str = "sector"
    reference_price_column: str = "last_price"
    impute_categorical_columns: bool = True
    impute_datetime_columns: bool = True

    # NEW: Business-rule imputation flags (v1.19)
    apply_dividend_zero_fill: bool = True
    apply_analyst_rating_zero_fill: bool = True
    apply_financial_statement_zero_fill: bool = True


@dataclass
class SemanticTransformConfig:
    """Configuration for semantic-aware transformations."""

    apply_log_transforms: bool = True
    log_transform_method: Literal["log1p", "signed_log"] = "log1p"  # NEW: "log1p" or "signed_log"
    log_transform_target_columns: Optional[List[str]] = None
    log_transform_market_values: bool = True
    exclude_ratios_from_winsorization: bool = True
    exclude_percentages_from_winsorization: bool = True
    exclude_counts_from_scaling: bool = True


@dataclass
class CurrencyConversionConfig:
    """Configuration for foreign currency conversion."""

    enabled: bool = False
    target_currency: str = "USD"
    currency_column: str = "unit"
    date_column: str = "reference_date"
    columns: Optional[List[str]] = None
    suffix: str = "_usd"
    cache_rates: bool = True
    max_fallback_days: int = 7
    use_business_day_fallback: bool = True


@dataclass
class DataSanitizationConfig:
    """Configuration for data sanitization and winsorization."""

    sanitize_data: bool = True
    apply_winsorization: bool = True
    winsorize_lower_percentile: float = 0.01
    winsorize_upper_percentile: float = 0.99

    # NEW: Business-rule sanitization flags (v1.19)
    apply_business_rule_zero_fills: bool = True


@dataclass
class ScalingConfig:
    """Configuration for feature scaling."""

    enabled: bool = False
    scaler_type: Literal["robust", "standard", "minmax"] = "robust"
    scale_by_sector: bool = True
    target_columns: Optional[List[str]] = None
    exclude_price_columns: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""

    enabled: bool = False
    preset: str = "standard"
    categories: Optional[List[str]] = None
    include_interactions: bool = True
    include_relative: bool = True
    engineer_earnings_analytics: bool = True
    # NEW v1.14 granular flags
    engineer_price_target_dynamics: bool = True
    engineer_fiscal_calendar: bool = True
    engineer_dividend_timing: bool = True
    engineer_eps_trajectory: bool = True
    engineer_cashflow_temporal: bool = True


@dataclass
class FeatureSelectionConfig:
    """Configuration for automated feature selection."""

    enabled: bool = False
    method: Literal["mutual_info", "correlation", "both"] = "both"
    min_importance_threshold: float = 0.01
    max_correlation_threshold: float = 0.95
    categories: Optional[List[str]] = None
    # Columns to preserve during feature selection (ml_workflow_guidelines.md Section 8.2)
    # These columns are kept in output regardless of importance scores
    preserve_columns: Optional[List[str]] = None


@dataclass
class FinancialMetricsConfig:
    """Configuration for financial metrics computation."""

    compute_valuation_metrics: bool = True
    compute_profitability_metrics: bool = True
    compute_growth_metrics: bool = True
    compute_leverage_metrics: bool = True
    compute_target_vs_price_metrics: bool = True
    compute_sector_specific_metrics: bool = True
    handle_sector_specific_metrics: bool = True
    generate_quality_alerts: bool = True
    generate_metrics_dashboard: bool = True
    output_directory: Optional[str] = None


@dataclass
class ETLConfig:
    """Master ETL configuration - combines all stage configs."""

    extraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
    validation: SchemaValidationConfig = field(default_factory=SchemaValidationConfig)
    dtype_casting: DtypeCastingConfig = field(default_factory=DtypeCastingConfig)
    semantic_classification: SemanticClassificationConfig = field(
        default_factory=SemanticClassificationConfig
    )
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    currency_conversion: CurrencyConversionConfig = field(default_factory=CurrencyConversionConfig)
    semantic_transform: SemanticTransformConfig = field(default_factory=SemanticTransformConfig)
    sanitization: DataSanitizationConfig = field(default_factory=DataSanitizationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    financial_metrics: FinancialMetricsConfig = field(default_factory=FinancialMetricsConfig)

    @classmethod
    def for_production(cls) -> "ETLConfig":
        """Factory for production-ready configuration."""
        return cls(
            validation=SchemaValidationConfig(
                validate_schema=True,
                validate_schema_alignment=True,
                schema_alignment_threshold=0.95,
            ),
            sanitization=DataSanitizationConfig(apply_winsorization=True),
            scaling=ScalingConfig(enabled=True),
        )

    @classmethod
    def for_development(cls) -> "ETLConfig":
        """Factory for fast development iteration."""
        return cls(
            extraction=DataExtractionConfig(limit=1000),
            imputation=ImputationConfig(apply_imputation=True, strategy="median_only"),
        )
