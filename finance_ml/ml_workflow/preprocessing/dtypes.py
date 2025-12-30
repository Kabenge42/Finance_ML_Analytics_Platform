"""
Datatype detection, validation, and casting utilities.

This module provides schema-aware datatype detection and casting
functionality aligned with code_guidelines.md v1.3+ requirements.
"""

import logging
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    normalize_column_name,
    list_required_schema_columns_for_etl,
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

    Notes:
        - In the unified ETL pipeline, some columns (e.g. Phase 9.3 engineered
          features) are intentionally not listed in COLUMN_SCHEMA. Those will
          appear under ``diagnostics['unknown_columns']`` but are still cast
          using heuristic inference via ``_infer_and_cast_unknown_column()``.
        - Derived metrics that should be schema-aware (e.g. log-transforms and
          valuation ratios used downstream) are registered in COLUMN_SCHEMA
          so they no longer appear as unknown and can participate in role-based
          validation.
        - Legacy/alias column names have been demoted to ``role: "auxiliary"``
          so they no longer appear in ``missing_expected_columns``. This keeps
          the diagnostics focused on truly required columns.
        - To distinguish hard errors (truly required missing columns) from soft
          warnings (optional features), compare ``missing_expected_columns``
          against ``list_required_schema_columns_for_etl()``.

    Example:
        >>> df = pd.DataFrame({'last_price': ['100.5', 'N/A', '200.0']})
        >>> df_cast, diag = detect_and_cast_dtypes(df)
        >>> df_cast['last_price'].dtype  # float64
        >>> diag['coercion_warnings']['last_price']  # 1 (for 'N/A')

        >>> # Check for truly required missing columns (ETL-specific)
        >>> required = list_required_schema_columns_for_etl()
        >>> critical_missing = [c for c in diag['missing_expected_columns'] if c in required]
    """
    if schema is None:
        schema = COLUMN_SCHEMA

    # Initialize diagnostics
    diagnostics = {
        "inferred_dtypes": {},
        "cast_applied": {},
        "coercion_warnings": {},
        "coercion_details": {},  # Detailed info about coerced values per column
        "unknown_columns": [],
        "missing_expected_columns": [],
    }

    # Make a copy to avoid modifying original
    df_cast = df.copy()

    # Determine reference date for fiscal year alignment
    reference_date: Optional[pd.Timestamp] = None
    if "reference_date" in df_cast.columns:
        ref_series = pd.to_datetime(df_cast["reference_date"], errors="coerce")
        if ref_series.notna().any():
            reference_date = ref_series.dropna().max().normalize()

    if reference_date is None:
        # Default reference date aligns with latest snapshot requirement (21 Dec 2025)
        reference_date = pd.Timestamp("2025-12-21")

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
    important_roles = {
        "feature",
        "target",
        "target_fallback",
        "id",
        "date",
        "categorical",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
        "count",
    }
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
                df_cast[col], coercion_count, coercion_info = _cast_to_numeric(
                    df_cast[col], target_dtype, col_name=col
                )
                if coercion_count > 0:
                    diagnostics["coercion_warnings"][col] = coercion_count
                    diagnostics["coercion_details"][col] = coercion_info
                    # Logging is now handled inside _cast_to_numeric with detailed info

            elif target_dtype == "datetime64[ns]":
                df_cast[col], coercion_count = _cast_to_datetime(df_cast[col])

                if normalized == "reference_date" and df_cast[col].notna().any():
                    # Refresh reference date from cleaned column for downstream fy_end alignment
                    reference_date = df_cast[col].dropna().max().normalize()

                if normalized == "fy_end":
                    # Align fiscal year end to the last calendar day of the month
                    aligned_fy_end = df_cast[col] + pd.offsets.MonthEnd(0)

                    if reference_date is not None:
                        mask = aligned_fy_end.notna() & (aligned_fy_end < reference_date)
                        if mask.any():
                            ref_year = reference_date.year

                            def _snap_to_ref_year(ts: pd.Timestamp) -> pd.Timestamp:
                                return pd.Timestamp(
                                    year=ref_year, month=ts.month, day=1
                                ) + pd.offsets.MonthEnd(0)

                            aligned_fy_end.loc[mask] = aligned_fy_end.loc[mask].apply(
                                _snap_to_ref_year
                            )

                    df_cast[col] = aligned_fy_end

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


def _cast_to_numeric(
    series: pd.Series, target_dtype: str, col_name: str = ""
) -> Tuple[pd.Series, int, Dict[str, Any]]:
    """
    Cast a series to numeric dtype with coercion tracking and detailed diagnostics.

    Provides pre-validation for common non-numeric patterns and fallback handling
    for values that cannot be converted. Logs detailed diagnostics about what
    values were coerced to help identify data quality issues.

    Args:
        series: Input series
        target_dtype: 'float' or 'int'
        col_name: Column name for logging (optional)

    Returns:
        Tuple of (casted_series, coercion_count, coercion_details) where
        coercion_details contains:
            - sample_coerced_values: List of up to 5 example values that were coerced
            - coercion_patterns: Dict of common patterns found (e.g., "N/A", "-", "")
    """
    original_valid_count = series.notna().sum()
    coercion_details: Dict[str, Any] = {
        "sample_coerced_values": [],
        "coercion_patterns": {},
    }

    # Pre-validation: Identify common non-numeric patterns before casting
    # This helps with debugging data quality issues
    if series.dtype == "object" or str(series.dtype) == "string":
        # Define common non-numeric patterns to detect
        common_patterns = [
            "N/A",
            "n/a",
            "NA",
            "na",
            "-",
            "--",
            "",
            " ",
            "null",
            "NULL",
            "None",
        ]

        for pattern in common_patterns:
            if pattern == "":
                # Count empty strings
                count = (series == "").sum() if series.dtype == "object" else 0
            else:
                count = (series == pattern).sum()
            if count > 0:
                coercion_details["coercion_patterns"][pattern] = int(count)

        # Log warning if significant non-numeric patterns found
        total_patterns = sum(coercion_details["coercion_patterns"].values())
        if total_patterns > 0 and col_name:
            logger.debug(
                f"Column '{col_name}': Pre-validation found {total_patterns} values "
                f"matching non-numeric patterns: {coercion_details['coercion_patterns']}"
            )

    # Use pd.to_numeric with errors='coerce' to handle invalid values
    if target_dtype == "int":
        # For int, first convert to float, then to nullable Int64
        numeric_series = pd.to_numeric(series, errors="coerce")
        # Convert to nullable integer type to preserve NaNs
        casted_series = numeric_series.astype("Int64")
    else:
        # For float - use downcast for memory optimization
        casted_series = pd.to_numeric(series, errors="coerce", downcast="float")

    new_valid_count = casted_series.notna().sum()
    coercion_count = int(original_valid_count - new_valid_count)

    # Collect sample coerced values for diagnostics
    if coercion_count > 0:
        # Find indices where original was valid but result is NaN
        was_valid = series.notna()
        now_nan = casted_series.isna()
        coerced_mask = was_valid & now_nan

        # Get sample of coerced values (up to 5)
        coerced_values = series[coerced_mask].head(5).tolist()
        coercion_details["sample_coerced_values"] = [str(v) for v in coerced_values]

        if col_name:
            logger.info(
                f"Column '{col_name}': {coercion_count} value(s) coerced to NaN. "
                f"Sample values: {coercion_details['sample_coerced_values']}"
            )

    return casted_series, coercion_count, coercion_details


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


def get_critical_missing_columns(
    diagnostics: Dict,
    include_extended_financials: bool = False,
) -> List[str]:
    """
    Filter missing_expected_columns to identify truly critical missing columns.

    This helper function compares the ``missing_expected_columns`` from dtype
    diagnostics against the canonical ETL-required columns to distinguish
    hard errors (truly required missing columns) from soft warnings (optional
    features that the ETL can proceed without).

    Args:
        diagnostics: The diagnostics dict returned by ``detect_and_cast_dtypes()``.
                    Must contain 'missing_expected_columns' key.
        include_extended_financials: When True, also check for extended financial
                                     columns (e.g. ebitda_ltm, total_assets_ltm).
                                     See ``list_required_schema_columns_for_etl()``.

    Returns:
        List of column names that are both:
        1. Missing from the DataFrame (in missing_expected_columns)
        2. Required by the ETL pipeline (in list_required_schema_columns_for_etl())

    Example:
        >>> df_cast, diag = detect_and_cast_dtypes(df)
        >>> critical = get_critical_missing_columns(diag)
        >>> if critical:
        ...     raise ValueError(f"Missing required columns: {critical}")

    Notes:
        - If ``critical_missing`` is empty, the ETL can proceed safely even if
          ``missing_expected_columns`` is not empty (those are optional features).
        - Use ``include_extended_financials=True`` for stricter validation in
          production pipelines that rely on core financial metrics.

    See Also:
        - ``detect_and_cast_dtypes()``: Main function that produces diagnostics
        - ``list_required_schema_columns_for_etl()``: Canonical required columns
    """
    missing_expected = diagnostics.get("missing_expected_columns", [])
    required_columns = list_required_schema_columns_for_etl(
        include_extended_financials=include_extended_financials
    )

    # Find intersection: columns that are both missing and required
    critical_missing = [col for col in missing_expected if col in required_columns]

    if critical_missing:
        logger.warning(
            f"Critical columns missing for ETL: {critical_missing}. "
            f"ETL may fail or produce incomplete results."
        )

    return critical_missing
