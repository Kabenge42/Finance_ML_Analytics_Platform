"""Semantic Transformation stages for ETL."""

import logging
from typing import List, Optional, Tuple, Dict, Any, NamedTuple

import numpy as np
import pandas as pd

from finance_ml.core.schema import list_categorical_cols, list_numeric_feature_cols
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
    get_market_value_columns,
    get_price_columns,
    get_ratio_columns,
    get_percentage_columns,
    get_count_columns,
)

logger = logging.getLogger(__name__)


class TransformationDiagnostics(NamedTuple):
    """Diagnostics for semantic transformations."""

    transformed_columns: List[str]
    skipped_negative_columns: List[str]
    skipped_reasons: Dict[str, str]
    transformed_count: int
    skipped_count: int


def _signed_log1p(series: pd.Series) -> pd.Series:
    """Apply signed log transform: sign(x) * log1p(|x|)."""
    return np.sign(series) * np.log1p(np.abs(series))


def run_semantic_classification_stage(df: pd.DataFrame) -> Dict[str, Any]:
    """Stage 1.6: Semantic column classification."""
    logger.info("Stage 1.6: Applying semantic column classification")

    classification_result = classify_columns(df.columns.tolist())

    # Use the lazy-loading functions to avoid constant issues
    price_cols = [col for col in get_price_columns() if col in df.columns]
    market_value_cols = [col for col in get_market_value_columns() if col in df.columns]
    ratio_cols = [col for col in get_ratio_columns() if col in df.columns]
    percentage_cols = [col for col in get_percentage_columns() if col in df.columns]
    count_cols = [col for col in get_count_columns() if col in df.columns]

    categorical_cols = list_categorical_cols()
    numeric_feature_cols = list_numeric_feature_cols()

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
    log_transform_target_columns: Optional[List[str]] = None,
    log_transform_method: str = "log1p",
    classification_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, TransformationDiagnostics]:
    """Stage 5: Semantic-aware transformations (log-transforms)."""
    logger.info(f"Stage 5: Applying semantic-aware transformations (method={log_transform_method})")

    empty_diagnostics = TransformationDiagnostics(
        transformed_columns=[],
        skipped_negative_columns=[],
        skipped_reasons={},
        transformed_count=0,
        skipped_count=0,
    )

    if not apply_log_transforms and not log_transform_market_values:
        return df, empty_diagnostics

    result = df.copy()

    # Identify columns to transform
    log_transform_cols = []

    if apply_log_transforms or log_transform_market_values:
        # Use cached market_value_cols if available, otherwise compute
        if classification_cache and "market_value_cols" in classification_cache:
            log_transform_cols = [
                col
                for col in classification_cache["market_value_cols"]
                if col in result.columns and col not in get_price_columns()
            ]
        else:
            market_value_cols = get_market_value_columns()
            price_cols = get_price_columns()
            log_transform_cols = [
                col for col in market_value_cols if col in result.columns and col not in price_cols
            ]

    if log_transform_target_columns:
        for col in log_transform_target_columns:
            if col in result.columns and col not in log_transform_cols:
                log_transform_cols.append(col)

    if not log_transform_cols:
        return result, empty_diagnostics

    # Batch processing: Identify columns with negative values upfront
    cols_subset = result[log_transform_cols]
    has_negative = (cols_subset < 0).any()

    if log_transform_method == "signed_log":
        # In signed_log mode, all columns are transformable
        transformable_cols = log_transform_cols
        skipped_cols = []
    else:
        # In log1p mode, skip columns with negative values
        transformable_cols = [col for col in log_transform_cols if not has_negative[col]]
        skipped_cols = [col for col in log_transform_cols if has_negative[col]]

    # Vectorized log transform for all transformable columns at once
    transformed_cols_list = []
    for col in transformable_cols:
        log_col_name = f"log_{col}"
        if log_transform_method == "signed_log" and has_negative[col]:
            result[log_col_name] = _signed_log1p(result[col])
        else:
            # Standard log1p for positive values (clipped at 0 just in case of tiny precision issues)
            result[log_col_name] = np.log1p(result[col].clip(lower=0))
        transformed_cols_list.append(col)

    # Mark skipped columns and record reasons
    skipped_reasons = {}
    for col in skipped_cols:
        result[f"log_{col}_applicable"] = result[col] >= 0
        skipped_reasons[col] = "contains_negative_values"

    diagnostics = TransformationDiagnostics(
        transformed_columns=transformed_cols_list,
        skipped_negative_columns=skipped_cols,
        skipped_reasons=skipped_reasons,
        transformed_count=len(transformed_cols_list),
        skipped_count=len(skipped_cols),
    )

    logger.info(f"Transformed {len(transformed_cols_list)} columns, skipped {len(skipped_cols)}")

    return result, diagnostics
