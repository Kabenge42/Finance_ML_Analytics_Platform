"""Feature selection stage for ETL."""

import logging
from typing import Tuple

import pandas as pd

from finance_ml.ml_workflow.features.selection import select_features_auto

logger = logging.getLogger(__name__)

def run_feature_selection_stage(
    df: pd.DataFrame,
    method: str = "both",
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95
) -> Tuple[pd.DataFrame, int, int, int]:
    """Stage 10: Automated feature selection."""
    logger.info("Stage 10: Applying automated feature selection")
    features_before = len(df.columns)
    
    # Need target column for feature selection
    target_col = None
    for col in ["price_target", "price_target_median"]:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        logger.warning("Feature selection skipped: no target column found")
        return df, features_before, features_before, 0
        
    try:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        X_selected = select_features_auto(
            X,
            y,
            importance_threshold=importance_threshold,
            correlation_threshold=correlation_threshold,
            method=method
        )
        
        result = X_selected.copy()
        result[target_col] = y
        
        features_after = len(result.columns) - 1
        features_removed = features_before - features_after - 1
        
        return result, features_before, features_after, features_removed
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        return df, features_before, features_before, 0
