"""Utility functions for feature engineering."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _safe_div(numer: pd.Series | float | int, denom: pd.Series) -> pd.Series:
    """Safely divide two Series (or scalar and Series), replacing inf/NaN with appropriate values."""
    if isinstance(numer, (float, int)):
        numer = pd.Series(numer, index=denom.index)

    result = numer.astype(float) / denom.astype(float).replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def _ensure_float_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Ensure a column exists and is float64 dtype to prevent TypeError on masked assignment."""
    if col_name not in df.columns:
        df[col_name] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df

def engineer_nonlinear_transforms(
    df: pd.DataFrame,
    log_features: Optional[List[str]] = None,
    sqrt_features: Optional[List[str]] = None,
    inverse_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply nonlinear transformations (log, sqrt, inverse) to specified features."""
    result = df.copy()
    
    if log_features:
        for col in log_features:
            if col in result.columns:
                result[f"log_{col}"] = np.log1p(result[col].clip(lower=0))
                
    if sqrt_features:
        for col in sqrt_features:
            if col in result.columns:
                result[f"sqrt_{col}"] = np.sqrt(result[col].clip(lower=0))
                
    if inverse_features:
        for col in inverse_features:
            if col in result.columns:
                result[f"inv_{col}"] = _safe_div(1.0, result[col])
                
    return result

def create_feature_interactions(
    df: pd.DataFrame, 
    features: Optional[List[str]] = None, 
    max_degree: int = 2
) -> pd.DataFrame:
    """Create polynomial interaction features."""
    if features is None:
        return df
        
    result = df.copy()
    valid_features = [f for f in features if f in result.columns]
    
    for i, f1 in enumerate(valid_features):
        for f2 in valid_features[i+1:]:
            result[f"{f1}_x_{f2}"] = result[f1] * result[f2]
            
    return result
