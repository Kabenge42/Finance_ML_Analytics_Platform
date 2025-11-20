"""
Datatype detection, validation, and casting utilities.

This module provides schema-aware datatype detection and casting
functionality aligned with code_guidelines.md v1.3+ requirements.
"""

import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    normalize_column_name,
)

logger = logging.getLogger(__name__)


def detect_and_cast_dtypes(
    df: pd.DataFrame, schema: Optional[Dict[str, Dict[str, str]]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Infer, validate, and cast dtypes according to schema.

    This function:
    1. Normalizes column names to match schema keys
    2. Infers dtypes for columns not in schema
    3. Casts columns to target dtypes with error tracking
    4. Reports diagnostics including coercion warnings and unknown columns

    Args:
        df: Input DataFrame with raw dtypes (often object)
        schema: Column schema dict (defaults to COLUMN_SCHEMA)

    Returns:
        Tuple of (casted_df, diagnostics) where diagnostics includes:
            - inferred_dtypes: {col: str} - Original inferred types
            - cast_applied: {col: str} - Target types applied
            - coercion_warnings: {col: int} - Count of values coerced to NaN
            - unknown_columns: list[str] - Columns not in schema
            - missing_expected_columns: list[str] - Schema columns not in df

    Example:
        >>> df = pd.DataFrame({'last_price': ['100.5', 'N/A', '200.0']})
        >>> df_cast, diag = detect_and_cast_dtypes(df)
        >>> df_cast['last_price'].dtype  # float64
        >>> diag['coercion_warnings']['last_price']  # 1 (for 'N/A')
    """
    if schema is None:
        schema = COLUMN_SCHEMA

    # Initialize diagnostics
    diagnostics = {
        "inferred_dtypes": {},
        "cast_applied": {},
        "coercion_warnings": {},
        "unknown_columns": [],
        "missing_expected_columns": [],
    }

    # Make a copy to avoid modifying original
    df_cast = df.copy()

    # Normalize column names in the dataframe for matching
    col_name_mapping = {}
    for col in df_cast.columns:
        normalized = normalize_column_name(col)
        col_name_mapping[col] = normalized

    # Track inferred dtypes before casting
    for col in df_cast.columns:
        diagnostics["inferred_dtypes"][col] = str(df_cast[col].dtype)

    # Identify unknown columns (not in schema)
    for col in df_cast.columns:
        normalized = col_name_mapping[col]
        if normalized not in schema:
            diagnostics["unknown_columns"].append(col)
            logger.warning(f"Column '{col}' (normalized: '{normalized}') not found in schema")

    # Identify missing expected columns (in schema but not in df)
    # Only report columns with role 'feature', 'target', 'id', or 'date'
    important_roles = {"feature", "target", "target_fallback", "id", "date", "categorical"}
    for schema_col, meta in schema.items():
        if meta["role"] in important_roles:
            # Check if any column in df normalizes to this schema column
            if schema_col not in col_name_mapping.values():
                diagnostics["missing_expected_columns"].append(schema_col)

    # Cast columns according to schema
    for col in df_cast.columns:
        normalized = col_name_mapping[col]

        if normalized not in schema:
            # Unknown column - try to infer type intelligently
            df_cast[col] = _infer_and_cast_unknown_column(df_cast[col], col, diagnostics)
            continue

        target_dtype = schema[normalized]["dtype"]

        try:
            if target_dtype in ["float", "int"]:
                df_cast[col], coercion_count = _cast_to_numeric(df_cast[col], target_dtype)
                if coercion_count > 0:
                    diagnostics["coercion_warnings"][col] = coercion_count
                    logger.info(
                        f"Column '{col}': {coercion_count} value(s) coerced to NaN during numeric casting"
                    )

            elif target_dtype == "datetime64[ns]":
                df_cast[col], coercion_count = _cast_to_datetime(df_cast[col])
                if coercion_count > 0:
                    diagnostics["coercion_warnings"][col] = coercion_count
                    logger.info(
                        f"Column '{col}': {coercion_count} value(s) coerced to NaT during datetime casting"
                    )

            elif target_dtype == "category":
                df_cast[col] = df_cast[col].astype("category")

            elif target_dtype == "string":
                df_cast[col] = df_cast[col].astype("string")

            elif target_dtype == "bool":
                df_cast[col] = df_cast[col].astype("bool")

            diagnostics["cast_applied"][col] = target_dtype

        except Exception as e:
            logger.error(f"Failed to cast column '{col}' to {target_dtype}: {e}")
            # Keep original dtype on failure
            continue

    logger.info(
        f"Type casting complete: {len(diagnostics['cast_applied'])} columns cast, "
        f"{len(diagnostics['unknown_columns'])} unknown columns, "
        f"{len(diagnostics['missing_expected_columns'])} expected columns missing"
    )

    return df_cast, diagnostics


def _cast_to_numeric(series: pd.Series, target_dtype: str) -> Tuple[pd.Series, int]:
    """
    Cast a series to numeric dtype with coercion tracking.

    Args:
        series: Input series
        target_dtype: 'float' or 'int'

    Returns:
        Tuple of (casted_series, coercion_count)
    """
    original_valid_count = series.notna().sum()

    # Use pd.to_numeric with errors='coerce' to handle invalid values
    if target_dtype == "int":
        # For int, first convert to float, then to nullable Int64
        numeric_series = pd.to_numeric(series, errors="coerce")
        # Convert to nullable integer type to preserve NaNs
        casted_series = numeric_series.astype("Int64")
    else:
        # For float
        casted_series = pd.to_numeric(series, errors="coerce")

    new_valid_count = casted_series.notna().sum()
    coercion_count = original_valid_count - new_valid_count

    return casted_series, coercion_count


def _cast_to_datetime(series: pd.Series) -> Tuple[pd.Series, int]:
    """
    Cast a series to datetime64[ns] with coercion tracking.

    Args:
        series: Input series

    Returns:
        Tuple of (casted_series, coercion_count)
    """
    original_valid_count = series.notna().sum()

    # Use pd.to_datetime with errors='coerce'
    casted_series = pd.to_datetime(series, errors="coerce")

    new_valid_count = casted_series.notna().sum()
    coercion_count = original_valid_count - new_valid_count

    return casted_series, coercion_count


def _infer_and_cast_unknown_column(
    series: pd.Series, col_name: str, diagnostics: Dict
) -> pd.Series:
    """
    Infer and cast dtype for a column not in schema.

    Uses heuristics:
    - Try numeric first
    - Then datetime
    - Fall back to string/category

    Args:
        series: Input series
        col_name: Column name for logging
        diagnostics: Diagnostics dict to update

    Returns:
        Casted series
    """
    # Try numeric
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() / len(series) > 0.5:  # >50% valid numeric
            logger.info(f"Unknown column '{col_name}' inferred as numeric")
            diagnostics["cast_applied"][col_name] = "float"
            return numeric_series
    except:
        pass

    # Try datetime
    try:
        datetime_series = pd.to_datetime(series, errors="coerce")
        if datetime_series.notna().sum() / len(series) > 0.5:  # >50% valid datetime
            logger.info(f"Unknown column '{col_name}' inferred as datetime")
            diagnostics["cast_applied"][col_name] = "datetime64[ns]"
            return datetime_series
    except:
        pass

    # Check if low cardinality -> category
    nunique = series.nunique()
    if nunique < len(series) * 0.05:  # <5% unique values
        logger.info(f"Unknown column '{col_name}' inferred as category (low cardinality)")
        diagnostics["cast_applied"][col_name] = "category"
        return series.astype("category")

    # Default to string
    logger.info(f"Unknown column '{col_name}' defaulted to string")
    diagnostics["cast_applied"][col_name] = "string"
    return series.astype("string")


def validate_dtypes_against_schema(
    df: pd.DataFrame, schema: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, List[str]]:
    """
    Validate that dataframe dtypes match schema expectations.

    Args:
        df: DataFrame to validate
        schema: Column schema (defaults to COLUMN_SCHEMA)

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    if schema is None:
        schema = COLUMN_SCHEMA

    errors = []
    warnings = []

    # Normalize column names
    col_name_mapping = {col: normalize_column_name(col) for col in df.columns}

    for col in df.columns:
        normalized = col_name_mapping[col]

        if normalized not in schema:
            warnings.append(f"Column '{col}' not in schema")
            continue

        expected_dtype = schema[normalized]["dtype"]
        actual_dtype = str(df[col].dtype)

        # Check type compatibility
        is_compatible = False

        if expected_dtype in ["float", "int"]:
            is_compatible = pd.api.types.is_numeric_dtype(df[col])
        elif expected_dtype == "datetime64[ns]":
            is_compatible = pd.api.types.is_datetime64_any_dtype(df[col])
        elif expected_dtype == "category":
            is_compatible = pd.api.types.is_categorical_dtype(
                df[col]
            ) or pd.api.types.is_object_dtype(df[col])
        elif expected_dtype == "string":
            is_compatible = pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
                df[col]
            )
        elif expected_dtype == "bool":
            is_compatible = pd.api.types.is_bool_dtype(df[col])

        if not is_compatible:
            errors.append(
                f"Column '{col}': expected dtype '{expected_dtype}', got '{actual_dtype}'"
            )

    return {"errors": errors, "warnings": warnings}


def get_dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of dtypes, missing values, and unique counts.

    Args:
        df: Input DataFrame

    Returns:
        Summary DataFrame with columns: column, dtype, missing_count, missing_pct, unique_count
    """
    summary_data = []

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = 100.0 * missing_count / len(df)
        unique_count = df[col].nunique()

        summary_data.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "unique_count": unique_count,
            }
        )

    return pd.DataFrame(summary_data)


def to_jsonable(obj):
    """
    Recursively convert an object to something that json can serialize.

    This function handles NumPy scalar types (np.integer, np.floating, np.bool_)
    that are commonly produced by pandas dtype operations and are not directly
    JSON-serializable. It recursively processes nested containers (dict, list,
    tuple, set) to convert all NumPy scalars to Python native types.

    Args:
        obj: Any Python object (scalar, container, or nested structure)

    Returns:
        JSON-serializable version of the object with all NumPy scalars converted
        to Python native types (int, float, bool). Unknown types are converted
        to string representation as a fallback.

    Example:
        >>> import numpy as np
        >>> diagnostics = {
        ...     'count': np.int64(42),
        ...     'ratio': np.float64(3.14),
        ...     'nested': {'flag': np.bool_(True), 'values': [np.int64(1), np.int64(2)]}
        ... }
        >>> import json
        >>> json.dumps(to_jsonable(diagnostics))  # Works without error
        '{"count": 42, "ratio": 3.14, "nested": {"flag": true, "values": [1, 2]}}'

    Notes:
        - Handles any depth of nesting in containers
        - Preserves JSON-safe types (str, int, float, bool, None) unchanged
        - Converts NumPy integer types to Python int
        - Converts NumPy floating types to Python float
        - Converts NumPy bool_ to Python bool
        - Recursively processes dict, list, tuple, set
        - Falls back to str() for unknown non-serializable types

    See Also:
        - Issue: TypeError: Object of type int64 is not JSON serializable
        - Used primarily for serializing dtype_diagnostics from detect_and_cast_dtypes()
    """
    # NumPy scalar types → Python scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Basic containers - recursively process contents
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # Anything already JSON-serializable (str, int, float, bool, None)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback: string representation for unknown types
    return str(obj)
