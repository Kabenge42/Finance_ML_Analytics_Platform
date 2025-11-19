"""
Data loading, schema validation, and datatype detection modules.

This package provides:
- COLUMN_SCHEMA: Authoritative schema registry based on database schema
- Schema validation and datatype detection utilities
- Helper functions for column roles and feature categorization
"""

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_INPUTS,
    get_expected_dtype,
    get_column_role,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
)

__all__ = [
    "COLUMN_SCHEMA",
    "PHASE93_FEATURE_INPUTS",
    "get_expected_dtype",
    "get_column_role",
    "list_numeric_feature_cols",
    "list_categorical_cols",
    "list_date_cols",
]
