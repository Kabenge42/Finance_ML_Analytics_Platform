"""Data Quality validation stage for ETL."""

import logging
from typing import Dict, Any

import pandas as pd

from finance_ml.ml_workflow.preprocessing.data import (
    validate_financial_data_quality,
    perform_early_pipeline_validation,
)

logger = logging.getLogger(__name__)

def run_quality_validation_stage(
    df: pd.DataFrame,
    validate_pipeline: bool = True
) -> Dict[str, Any]:
    """Stage 12: Quality validation."""
    logger.info("Stage 12: Validating data quality")
    
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    quality_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
    
    region = "ALL"
    if "region" in df.columns and len(df["region"].unique()) == 1:
        region = df["region"].iloc[0]
        
    quality_metrics = validate_financial_data_quality(df, region=region)
    quality_metrics["overall_quality_score"] = quality_score
    
    if validate_pipeline:
        pipeline_metrics = perform_early_pipeline_validation(df)
        quality_metrics["pipeline_validation"] = pipeline_metrics
        
    return quality_metrics
