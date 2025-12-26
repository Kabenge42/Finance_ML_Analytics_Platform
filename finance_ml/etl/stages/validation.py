"""Validation and Row Dropping stages for ETL."""

import logging
from typing import Tuple, List

import pandas as pd

from finance_ml.ml_workflow.preprocessing.data import validate_schema

logger = logging.getLogger(__name__)

def run_validation_stage(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """Stage 2: Validate schema.
    
    Validates that the DataFrame contains required columns for the ETL pipeline.
    Uses the canonical validate_schema function from preprocessing.data module.
    
    Args:
        df: DataFrame to validate
        require_target: If True, requires price_target or price_target_median column
        
    Returns:
        Tuple of (is_valid, errors) where:
            - is_valid: True if validation passes
            - errors: List of error message strings
            
    Reference:
        code_guidelines.md Section 5.3.1 - ETL-Required Columns
    """
    logger.info("Stage 2: Validating schema")
    # Call validate_schema with correct parameter name (require_target, not require_target_column)
    is_valid, errors = validate_schema(df, require_target=require_target)
    return is_valid, errors


def run_row_dropping_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: Drop invalid rows with missing critical fields.
    
    Drops rows where critical columns (ticker, last_price, sector) are missing.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with invalid rows removed
        
    Reference:
        code_guidelines.md Section 8.6 - ETL Pipeline Best Practices
    """
    logger.info("Stage 3: Dropping rows with missing critical fields")
    critical_cols = ["ticker", "last_price", "sector"]
    existing_critical = [c for c in critical_cols if c in df.columns]
    before = len(df)
    result = df.dropna(subset=existing_critical)
    after = len(result)
    if before > after:
        logger.info(f"Dropped {before - after} rows with missing critical fields")
    return result
