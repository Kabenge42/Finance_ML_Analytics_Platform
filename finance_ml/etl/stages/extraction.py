"""Extraction and Dtype Casting stages for ETL."""

import logging
from typing import Tuple, Dict, Any

import pandas as pd

from finance_ml.ml_workflow.preprocessing.data import normalize_columns
from finance_ml.ml_workflow.preprocessing.dtypes import detect_and_cast_dtypes

logger = logging.getLogger(__name__)

def run_extraction_stage(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Stage 1: Normalize columns."""
    if normalize:
        logger.info("Stage 1: Normalizing columns")
        return normalize_columns(df)
    return df

def run_dtype_casting_stage(df: pd.DataFrame, track_diagnostics: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Stage 1.5: Dtype casting."""
    logger.info("Stage 1.5: Casting data types")
    df_cast, diagnostics = detect_and_cast_dtypes(df)
    # Return empty diagnostics if tracking is disabled
    if not track_diagnostics:
        diagnostics = {}
    return df_cast, diagnostics
