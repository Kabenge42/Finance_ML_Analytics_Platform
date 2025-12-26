"""Feature Scaling stage for ETL."""

import logging

import pandas as pd

from finance_ml.ml_workflow.preprocessing.scaling import scale_features

logger = logging.getLogger(__name__)

def run_scaling_stage(
    df: pd.DataFrame, 
    scaler_type: str = "robust", 
    scale_by_sector: bool = True
) -> pd.DataFrame:
    """Stage 7: Apply feature scaling.
    
    Args:
        df: DataFrame to scale
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        scale_by_sector: Whether to scale separately by sector
        
    Returns:
        DataFrame with scaled features
    """
    logger.info(f"Stage 7: Applying feature scaling (type={scaler_type}, by_sector={scale_by_sector})")
    return scale_features(df, scaler_type=scaler_type, by_sector=scale_by_sector)
