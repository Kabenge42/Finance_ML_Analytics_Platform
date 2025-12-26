"""
DEPRECATED: Use finance_ml.core.schema instead.
This module is maintained for backward compatibility.
"""

import warnings

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    get_sql_column_name,
    normalize_column_name,
    generate_sql_schema,
    get_expected_dtype,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
    list_etl_generated_column_patterns,
    list_required_schema_columns_for_etl,
)

warnings.warn(
    "finance_ml.ml_workflow.data.schema is deprecated. "
    "Please import from finance_ml.core.schema instead.",
    DeprecationWarning,
    stacklevel=2
)

def get_column_role(column: str):
    """DEPRECATED: Get role for a column."""
    from finance_ml.core.schema import COLUMN_SCHEMA
    if column in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[column]["role"]
    return None

__all__ = [
    "COLUMN_SCHEMA",
    "PHASE93_FEATURE_CATEGORIES",
    "get_sql_column_name",
    "normalize_column_name",
    "generate_sql_schema",
    "get_expected_dtype",
    "list_numeric_feature_cols",
    "list_categorical_cols",
    "list_date_cols",
    "list_etl_generated_column_patterns",
    "list_required_schema_columns_for_etl",
    "get_column_role",
]
