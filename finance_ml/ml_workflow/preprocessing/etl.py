"""
Unified ETL (Extract, Transform, Load) Pipeline for Financial Data.

This module serves as the primary entry point for ETL operations, providing
backward-compatible interfaces and convenience functions while delegating
core orchestration to the modular finance_ml.etl subpackage.

Architecture:
    Orchestration: finance_ml.etl.pipeline.ETLPipeline
    Configuration: finance_ml.etl.config.ETLConfig
    Metrics:       finance_ml.etl.metrics.ETLMetrics
    Stages:        finance_ml.etl.stages.*

Aligned with code_guidelines.md ETL best practices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

# Re-export key classes from modular subpackage
from finance_ml.etl.config import (
    ETLConfig as ModularETLConfig,
)
from finance_ml.etl.metrics import ETLMetrics
from finance_ml.etl.pipeline import ETLPipeline, run_etl_pipeline as run_etl_pipeline_base, validate_etl_output, \
    report_column_changes

logger = logging.getLogger(__name__)

# Re-export ETLMetrics and ETLPipeline for backward compatibility
__all__ = [
    'ETLConfig',
    'ETLMetrics',
    'ETLPipeline',
    'run_etl_pipeline',
    'etl_from_csv',
    'etl_from_database',
    'etl_with_imputation',
    'etl_with_imputation_and_scaling',
    'etl_with_financial_metrics',
    'etl_with_features',
    'validate_etl_output',
    'report_column_changes'
]

class ETLConfig(ModularETLConfig):
    """
    Unified ETL Pipeline Configuration.
    
    This class extends ModularETLConfig with backward-compatible
    property accessors for legacy flat arguments.
    
    For new code, prefer using ModularETLConfig directly from
    finance_ml.etl.config with explicit nested config objects.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if kwargs:
            self._apply_legacy_overrides(kwargs)
    
    def _apply_legacy_overrides(self, legacy_kwargs: Dict[str, Any]) -> None:
        """Map legacy flat arguments into nested configs for backward compatibility."""
        legacy_map = {
            'normalize_columns': ('extraction', 'normalize_column_names'),
            'limit': ('extraction', 'limit'),
            'validate_schema': ('validation', 'validate_schema'),
            'require_target': ('validation', 'require_target_column'),
            'drop_invalid_rows': ('validation', 'drop_rows_with_missing_critical_fields'),
            'validate_quality': ('validation', 'validate_quality'),
            'validate_pipeline': ('validation', 'validate_pipeline'),
            'custom_validators': ('validation', 'custom_validators'),
            'apply_dtype_casting': ('dtype_casting', 'apply_dtype_casting'),
            'track_dtype_diagnostics': ('dtype_casting', 'track_diagnostics'),
            'use_semantic_column_classification': ('semantic_classification', 'enabled'),
            'preserve_price_columns': ('semantic_classification', 'preserve_price_columns'),
            'apply_imputation': ('imputation', 'apply_imputation'),
            'imputation_strategy': ('imputation', 'strategy'),
            'knn_neighbors': ('imputation', 'knn_neighbors'),
            'imputation_sector_column': ('imputation', 'sector_column'),
            'imputation_price_column': ('imputation', 'reference_price_column'),
            'handle_categorical_imputation': ('imputation', 'impute_categorical_columns'),
            'handle_datetime_imputation': ('imputation', 'impute_datetime_columns'),
            'apply_log_transforms': ('semantic_transform', 'apply_log_transforms'),
            'log_transform_method': ('semantic_transform', 'log_transform_method'),
            'log_transform_market_values': ('semantic_transform', 'log_transform_market_values'),
            'log_transform_columns': ('semantic_transform', 'log_transform_target_columns'),
            'exclude_ratios_from_winsorization': ('semantic_transform', 'exclude_ratios_from_winsorization'),
            'exclude_percentages_from_winsorization': ('semantic_transform', 'exclude_percentages_from_winsorization'),
            'exclude_counts_from_scaling': ('semantic_transform', 'exclude_counts_from_scaling'),
            'sanitize_data': ('sanitization', 'sanitize_data'),
            'apply_winsorization': ('sanitization', 'apply_winsorization'),
            'winsorize_lower_percentile': ('sanitization', 'winsorize_lower_percentile'),
            'winsorize_upper_percentile': ('sanitization', 'winsorize_upper_percentile'),
            'apply_scaling': ('scaling', 'enabled'),
            'scaler_type': ('scaling', 'scaler_type'),
            'scale_by_sector': ('scaling', 'scale_by_sector'),
            'scaling_columns': ('scaling', 'target_columns'),
            'exclude_price_columns_from_scaling': ('scaling', 'exclude_price_columns'),
            'apply_feature_engineering': ('feature_engineering', 'enabled'),
            'feature_preset': ('feature_engineering', 'preset'),
            'feature_categories': ('feature_engineering', 'categories'),
            'engineer_earnings_analytics': ('feature_engineering', 'engineer_earnings_analytics'),
            'apply_feature_selection': ('feature_selection', 'enabled'),
            'feature_selection_method': ('feature_selection', 'method'),
            'importance_threshold': ('feature_selection', 'min_importance_threshold'),
            'correlation_threshold': ('feature_selection', 'max_correlation_threshold'),
            'feature_selection_categories': ('feature_selection', 'categories'),
            'compute_valuation_metrics': ('financial_metrics', 'compute_valuation_metrics'),
            'compute_profitability_metrics': ('financial_metrics', 'compute_profitability_metrics'),
            'compute_growth_metrics': ('financial_metrics', 'compute_growth_metrics'),
            'compute_leverage_metrics': ('financial_metrics', 'compute_leverage_metrics'),
            'compute_target_vs_price': ('financial_metrics', 'compute_target_vs_price_metrics'),
            'handle_sector_specific_metrics': ('financial_metrics', 'handle_sector_specific_metrics'),
            'generate_quality_alerts': ('financial_metrics', 'generate_quality_alerts'),
            'generate_metrics_dashboard': ('financial_metrics', 'generate_metrics_dashboard'),
            'output_subdir': ('financial_metrics', 'output_directory'),
        }
        
        for legacy_key, (config_attr, attr_name) in legacy_map.items():
            if legacy_key in legacy_kwargs:
                config_obj = getattr(self, config_attr)
                setattr(config_obj, attr_name, legacy_kwargs[legacy_key])

    # Property accessors for legacy flat arguments
    @property
    def normalize_columns(self) -> bool:
        """Whether to normalize column names."""
        return self.extraction.normalize_column_names

    @normalize_columns.setter
    def normalize_columns(self, value: bool) -> None:
        self.extraction.normalize_column_names = value

    @property
    def limit(self) -> Optional[int]:
        """Limit for data extraction."""
        return self.extraction.limit

    @limit.setter
    def limit(self, value: Optional[int]) -> None:
        self.extraction.limit = value

    @property
    def validate_schema(self) -> bool:
        """Whether to validate schema."""
        return self.validation.validate_schema

    @validate_schema.setter
    def validate_schema(self, value: bool) -> None:
        self.validation.validate_schema = value

def run_etl_pipeline(
    source: Literal['csv', 'db', 'all_stocks'],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    config: Optional[ETLConfig] = None,
    return_metrics: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """Backward-compatible entry point for running the ETL pipeline."""
    return run_etl_pipeline_base(
        source=source,
        data_dir=data_dir,
        db_url=db_url,
        config=config,
        return_metrics=return_metrics,
    )

def etl_from_csv(
    data_dir: Path | str, 
    config: Optional[ETLConfig] = None
) -> pd.DataFrame:
    """Convenience function to run ETL from CSV."""
    return run_etl_pipeline(source='csv', data_dir=data_dir, config=config)

def etl_from_database(
    db_url: str, 
    unified_table: bool = False, 
    config: Optional[ETLConfig] = None
) -> pd.DataFrame:
    """Convenience function to run ETL from database."""
    source = 'all_stocks' if unified_table else 'db'
    return run_etl_pipeline(source=source, db_url=db_url, config=config)

def etl_with_imputation(
    source: Literal['csv', 'db', 'all_stocks'],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    imputation_strategy: str = '6step',
    return_metrics: bool = True
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """ETL with specific imputation strategy."""
    config = ETLConfig(imputation_strategy=imputation_strategy)
    return run_etl_pipeline(source=source, data_dir=data_dir, db_url=db_url, config=config, return_metrics=return_metrics)

def etl_with_imputation_and_scaling(
    source: Literal['csv', 'db', 'all_stocks'],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    imputation_strategy: str = '6step',
    scaler_type: Literal['robust', 'standard', 'minmax'] = 'robust',
    scale_by_sector: bool = True,
    return_metrics: bool = True
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """ETL with imputation and scaling."""
    config = ETLConfig(
        apply_imputation=True,
        imputation_strategy=imputation_strategy,
        apply_scaling=True,
        scaler_type=scaler_type,
        scale_by_sector=scale_by_sector
    )
    return run_etl_pipeline(source=source, data_dir=data_dir, db_url=db_url, config=config, return_metrics=return_metrics)

def etl_with_financial_metrics(
    source: Literal['csv', 'db', 'all_stocks'],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    compute_all_metrics: bool = True,
    output_dir: Optional[Path] = None,
    return_metrics: bool = True
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """ETL with financial metrics computation."""
    config = ETLConfig(
        compute_valuation_metrics=compute_all_metrics,
        compute_profitability_metrics=compute_all_metrics,
        compute_growth_metrics=compute_all_metrics,
        compute_leverage_metrics=compute_all_metrics,
        output_subdir=str(output_dir) if output_dir else None
    )
    return run_etl_pipeline(source=source, data_dir=data_dir, db_url=db_url, config=config, return_metrics=return_metrics)

def etl_with_features(
    source: Literal['csv', 'db', 'all_stocks'],
    data_dir: Optional[Path | str] = None,
    db_url: Optional[str] = None,
    feature_preset: str = 'comprehensive',
    feature_categories: Optional[List[str]] = None,
    auto_feature_selection: bool = False,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    engineer_earnings_analytics: bool = True,
    config: Optional[ETLConfig] = None,
    return_metrics: bool = True
) -> pd.DataFrame | Tuple[pd.DataFrame, ETLMetrics]:
    """ETL with full feature engineering (Phase 9.3)."""
    if config is None:
        config = ETLConfig()
    
    config.apply_feature_engineering = True
    config.feature_preset = feature_preset
    config.feature_categories = feature_categories
    config.apply_feature_selection = auto_feature_selection
    config.importance_threshold = importance_threshold
    config.correlation_threshold = correlation_threshold
    config.engineer_earnings_analytics = engineer_earnings_analytics
    
    return run_etl_pipeline(source=source, data_dir=data_dir, db_url=db_url, config=config, return_metrics=return_metrics)
