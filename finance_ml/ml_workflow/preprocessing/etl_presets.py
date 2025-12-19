"""Factory helpers for standardized ETL configurations (Section 7.5)."""

from __future__ import annotations

from finance_ml.ml_workflow.preprocessing.etl import (
    DataExtractionConfig,
    SchemaValidationConfig,
    SemanticClassificationConfig,
    ImputationConfig,
    SemanticTransformConfig,
    DataSanitizationConfig,
    ScalingConfig,
    FeatureEngineeringConfig,
    ETLConfig,
)


def get_etl_config_comprehensive() -> ETLConfig:
    """Comprehensive ETL configuration aligned with the unified pipeline.

    Emphasizes schema validation, semantic classification, full feature
    engineering, and safety rails (winsorization, scaling).
    """

    extraction = DataExtractionConfig(normalize_column_names=True)
    validation = SchemaValidationConfig(
        validate_schema=True,
        require_target_column=True,
        validate_schema_alignment=True,
        schema_alignment_threshold=0.80,
        validate_pipeline=True,
    )
    semantic = SemanticClassificationConfig(enabled=True, preserve_price_columns=True)
    imputation = ImputationConfig(apply_imputation=True, strategy="6step")
    semantic_transform = SemanticTransformConfig(
        apply_log_transforms=True,
        log_transform_market_values=True,
        exclude_ratios_from_winsorization=True,
        exclude_percentages_from_winsorization=True,
        exclude_counts_from_scaling=True,
    )
    sanitization = DataSanitizationConfig(
        sanitize_data=True,
        apply_winsorization=True,
        winsorize_lower_percentile=0.10,
        winsorize_upper_percentile=0.90,
    )
    scaling = ScalingConfig(enabled=True, scaler_type="robust", scale_by_sector=True)
    feature_engineering = FeatureEngineeringConfig(
        enabled=True,
        preset="comprehensive",
        categories=None,
    )

    return ETLConfig(
        extraction=extraction,
        validation=validation,
        semantic_classification=semantic,
        imputation=imputation,
        semantic_transform=semantic_transform,
        sanitization=sanitization,
        scaling=scaling,
        feature_engineering=feature_engineering,
    )


def get_etl_config_quick() -> ETLConfig:
    """Lightweight ETL configuration optimized for speed and iteration."""

    extraction = DataExtractionConfig(normalize_column_names=True)
    validation = SchemaValidationConfig(
        validate_schema=False,
        require_target_column=False,
        validate_schema_alignment=False,
        schema_alignment_threshold=0.0,
        validate_pipeline=False,
        drop_rows_with_missing_critical_fields=False,
    )
    semantic = SemanticClassificationConfig(enabled=True, preserve_price_columns=True)
    imputation = ImputationConfig(apply_imputation=False)
    semantic_transform = SemanticTransformConfig(
        apply_log_transforms=False,
        log_transform_market_values=False,
        exclude_ratios_from_winsorization=True,
        exclude_percentages_from_winsorization=True,
        exclude_counts_from_scaling=True,
    )
    sanitization = DataSanitizationConfig(
        sanitize_data=False,
        apply_winsorization=False,
        winsorize_lower_percentile=0.10,
        winsorize_upper_percentile=0.90,
    )
    scaling = ScalingConfig(enabled=False, scaler_type="robust", scale_by_sector=False)
    feature_engineering = FeatureEngineeringConfig(
        enabled=False,
        preset="basic",
        categories=None,
    )

    return ETLConfig(
        extraction=extraction,
        validation=validation,
        semantic_classification=semantic,
        imputation=imputation,
        semantic_transform=semantic_transform,
        sanitization=sanitization,
        scaling=scaling,
        feature_engineering=feature_engineering,
    )


__all__ = ["get_etl_config_comprehensive", "get_etl_config_quick"]
