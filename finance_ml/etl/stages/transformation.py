"""Semantic Transformation stages for ETL."""

import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_numeric_feature_cols
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
    MARKET_VALUE_COLUMNS,
    PRICE_COLUMNS,
    RATIO_COLUMNS,
    PERCENTAGE_COLUMNS,
    COUNT_COLUMNS,
)

logger = logging.getLogger(__name__)

def run_semantic_classification_stage(df: pd.DataFrame) -> Dict[str, Any]:
    """Stage 1.6: Semantic column classification."""
    logger.info("Stage 1.6: Applying semantic column classification")
    
    classification_result = classify_columns(df.columns.tolist())
    
    price_cols = [col for col in PRICE_COLUMNS if col in df.columns]
    market_value_cols = [col for col in MARKET_VALUE_COLUMNS if col in df.columns]
    ratio_cols = [col for col in RATIO_COLUMNS if col in df.columns]
    percentage_cols = [col for col in PERCENTAGE_COLUMNS if col in df.columns]
    count_cols = [col for col in COUNT_COLUMNS if col in df.columns]
    
    categorical_cols = list_categorical_cols()
    numeric_feature_cols = list_numeric_feature_cols()
    
    expected_categorical = [col for col in categorical_cols if col in df.columns]
    expected_numeric = [col for col in numeric_feature_cols if col in df.columns]
    
    logger.info(
        f"Column classification: price={len(price_cols)}, "
        f"market_value={len(market_value_cols)}, ratio={len(ratio_cols)}, "
        f"percentage={len(percentage_cols)}, count={len(count_cols)}"
    )
    
    return {
        "price_columns_count": len(price_cols),
        "market_value_columns_count": len(market_value_cols),
        "ratio_columns_count": len(ratio_cols),
        "percentage_columns_count": len(percentage_cols),
        "count_columns_count": len(count_cols),
        "classification_result": classification_result,
        "market_value_cols": market_value_cols
    }

def run_semantic_transformations_stage(
    df: pd.DataFrame,
    apply_log_transforms: bool = True,
    log_transform_market_values: bool = True,
    log_transform_target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int, int]:
    """Stage 5: Semantic-aware transformations (log-transforms)."""
    logger.info("Stage 5: Applying semantic-aware transformations")
    
    if not apply_log_transforms and not log_transform_market_values:
        return df, 0, 0
        
    result = df.copy()
    log_transform_cols = []
    
    if apply_log_transforms or log_transform_market_values:
        log_transform_cols.extend(
            [col for col in MARKET_VALUE_COLUMNS if col in result.columns and col not in PRICE_COLUMNS]
        )
        
    if log_transform_target_columns:
        for col in log_transform_target_columns:
            if col in result.columns and col not in log_transform_cols:
                log_transform_cols.append(col)
                
    transformed_count = 0
    skipped_negative = 0
    
    for col in log_transform_cols:
        if (result[col] < 0).any():
            result[f"log_{col}_applicable"] = result[col] >= 0
            skipped_negative += 1
            continue
            
        log_col_name = f"log_{col}"
        result[log_col_name] = np.log1p(result[col].clip(lower=0))
        transformed_count += 1
        
    return result, transformed_count, skipped_negative
