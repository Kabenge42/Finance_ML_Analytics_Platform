"""Schema validation stage for ETL."""

import logging
from typing import Dict, Any, Set

import pandas as pd

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    get_expected_dtype,
    list_required_schema_columns_for_etl,
    PHASE93_FEATURE_CATEGORIES,
)
from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import CONDITIONAL_METRICS

logger = logging.getLogger(__name__)

def run_schema_alignment_validation_stage(df: pd.DataFrame) -> Dict[str, Any]:
    """Stage 11: Validate schema alignment."""
    logger.info("Stage 11: Validating schema alignment")
    
    df_cols = set(df.columns)
    schema_cols = set(COLUMN_SCHEMA.keys())
    
    # Allowlist: Phase 9.3 engineered feature outputs
    engineered_allowlist: Set[str] = set()
    for _cat, feats in PHASE93_FEATURE_CATEGORIES.items():
        engineered_allowlist.update(feats)
        
    def _is_allowlisted_engineered(col: str) -> bool:
        if col in engineered_allowlist: return True
        if col.startswith("event_prob_"): return True
        if col.startswith("log_") and col[4:] in schema_cols: return True
        if col.endswith("_applicable"):
            base = col[:-len("_applicable")]
            if base in schema_cols or base in CONDITIONAL_METRICS: return True
            if base.startswith("log_"): return True
        if col.endswith("_growth") or col.endswith("_yoy"): return True
        if col.startswith("sector_") and "_x_" in col: return True
        if any(x in col for x in ["_sector_percentile", "_sector_zscore", "_vs_sector_median", "_vs_sector_top_quartile"]): return True
        if col.endswith("_squared") or col.endswith("_cubed"): return True
        if col.startswith("fte_"): return True
        if "_momentum_" in col or col.endswith("_momentum"): return True
        if col.endswith("_volatility"): return True
        if col == "_reference_date": return True
        return False

    unknown_columns = []
    recognized_count = 0
    allowlisted_engineered = []
    
    for col in df_cols:
        if col in schema_cols:
            recognized_count += 1
        elif _is_allowlisted_engineered(col):
            allowlisted_engineered.append(col)
        else:
            unknown_columns.append(col)
            
    required_cols = list_required_schema_columns_for_etl()
    missing_expected = [col for col in required_cols if col not in df_cols]
    
    dtype_mismatches = {}
    for col in (df_cols & schema_cols):
        expected = get_expected_dtype(col)
        actual = str(df[col].dtype)
        
        is_match = False
        if expected == "float" and ("float" in actual or "Int" in actual): is_match = True
        elif expected == "int" and ("int" in actual or "Int" in actual): is_match = True
        elif expected == "string" and ("object" in actual or "string" in actual): is_match = True
        elif expected == "category" and ("category" in actual or "object" in actual): is_match = True
        elif expected == "datetime64[ns]" and "datetime" in actual: is_match = True
        elif expected == "bool" and "bool" in actual: is_match = True
        
        if not is_match:
            dtype_mismatches[col] = {"expected": expected, "actual": actual}
            
    total_relevant = len(required_cols)
    matched_required = len([c for c in required_cols if c in df_cols])
    alignment_score = matched_required / total_relevant if total_relevant > 0 else 1.0
    
    return {
        "unknown_columns": sorted(unknown_columns),
        "missing_expected_columns": sorted(missing_expected),
        "dtype_mismatches": dtype_mismatches,
        "alignment_score": alignment_score,
        "recognized_columns_count": recognized_count,
        "allowlisted_engineered": allowlisted_engineered
    }
