"""Data Sanitization stage for ETL."""

import logging

import pandas as pd

from finance_ml.ml_workflow.preprocessing.data import sanitize_dataframe_with_logging

logger = logging.getLogger(__name__)

def run_sanitization_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Sanitize data."""
    logger.info("Stage 4: Sanitizing data")
    return sanitize_dataframe_with_logging(df)
