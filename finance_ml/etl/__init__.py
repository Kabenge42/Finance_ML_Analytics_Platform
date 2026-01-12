"""
ETL (Extract, Transform, Load) module for Finance ML Analytics Platform.

This module provides a unified pipeline for processing financial data, including
extraction from various sources, imputation, currency conversion, sanitization,
scaling, and financial metrics computation.
"""

from .config import (
    ETLConfig,
    DataExtractionConfig,
    SchemaValidationConfig,
    DtypeCastingConfig,
    SemanticClassificationConfig,
    ImputationConfig,
    CurrencyConversionConfig,
    SemanticTransformConfig,
    DataSanitizationConfig,
    ScalingConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    FinancialMetricsConfig,
)
from .currency import (
    convert_to_usd,
    CurrencyConverter,
    convert_with_fallback_date,
    CurrencyConversionMetrics,
)
from .metrics import ETLMetrics
from .pipeline import ETLPipeline, run_etl_pipeline, validate_etl_output

__all__ = [
    "ETLConfig",
    "DataExtractionConfig",
    "SchemaValidationConfig",
    "DtypeCastingConfig",
    "SemanticClassificationConfig",
    "ImputationConfig",
    "CurrencyConversionConfig",
    "SemanticTransformConfig",
    "DataSanitizationConfig",
    "ScalingConfig",
    "FeatureEngineeringConfig",
    "FeatureSelectionConfig",
    "FinancialMetricsConfig",
    "ETLPipeline",
    "run_etl_pipeline",
    "validate_etl_output",
    "ETLMetrics",
    "convert_to_usd",
    "CurrencyConverter",
    "convert_with_fallback_date",
    "CurrencyConversionMetrics",
]
