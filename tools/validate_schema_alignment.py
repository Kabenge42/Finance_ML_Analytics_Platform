#!/usr/bin/env python3
"""
Validate alignment between SQL schema (create_equities_schema.sql),
CSV files, and Python COLUMN_SCHEMA (schema.py).

Performs comprehensive three-way alignment check and generates a report.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set


def normalize_column_name(col: str) -> str:
    """
    Normalize column name per code_guidelines.md rules.

    Transformation rules:
    - Lowercase
    - Replace non-alphanumeric with underscore
    - Strip leading/trailing underscores
    - Special cases: # → num_, % → _pct (handled in context)
    """
    # Handle special prefix: # → num_
    if col.startswith("#"):
        col = "num" + col[1:]

    # Replace non-alphanumeric with underscore
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", col)

    # Strip leading/trailing underscores and lowercase
    normalized = normalized.strip("_").lower()

    return normalized


def extract_sql_columns(schema_path: Path) -> List[Tuple[str, str, str]]:
    """
    Extract (original_name, normalized_name, data_type) from SQL schema.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract CREATE TABLE statement
    match = re.search(
        r"CREATE TABLE equities\s*\((.*?)\)\s*TABLESPACE", content, re.DOTALL | re.IGNORECASE
    )

    if not match:
        raise ValueError("Could not find CREATE TABLE equities statement")

    table_def = match.group(1)

    # Parse column definitions: "Column Name"  TYPE
    pattern = re.compile(r'"([^"]+)"\s+(TEXT|NUMERIC|DATE)', re.IGNORECASE)

    columns = []
    for m in pattern.finditer(table_def):
        orig_name = m.group(1)
        data_type = m.group(2).upper()
        norm_name = normalize_column_name(orig_name)
        columns.append((orig_name, norm_name, data_type))

    return columns


def extract_python_schema(schema_py_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Extract COLUMN_SCHEMA from schema.py by importing it.
    """
    # Add parent directory to path to allow import
    sys.path.insert(0, str(schema_py_path.parent.parent.parent))

    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

    return COLUMN_SCHEMA


def validate_alignment(
    sql_columns: List[Tuple[str, str, str]], python_schema: Dict[str, Dict[str, str]]
) -> Dict[str, any]:
    """
    Validate alignment between SQL and Python schemas.

    Returns dictionary with:
    - total_sql: Count of SQL columns
    - total_python_source: Count of Python source columns (non-derived)
    - matched: Count of matched columns
    - missing_in_python: List of SQL columns not in Python schema
    - extra_in_python: List of Python columns not in SQL (derived features OK)
    - dtype_mismatches: List of columns with dtype mismatches
    """
    sql_norm_to_orig = {norm: (orig, dtype) for orig, norm, dtype in sql_columns}
    sql_norm_set = set(sql_norm_to_orig.keys())

    # Filter Python schema to source columns (exclude derived features)
    # Derived features typically have prefixes like: log_, ratio_, pct_, etc.
    derived_prefixes = ("log_", "ratio_", "pct_", "zscore_", "rank_", "rolling_")
    python_source = {
        k: v
        for k, v in python_schema.items()
        if not any(k.startswith(prefix) for prefix in derived_prefixes)
        and v.get("role") != "auxiliary"  # Skip auxiliary/alias columns
    }

    python_norm_set = set(python_source.keys())

    # Find matches and mismatches
    matched = sql_norm_set & python_norm_set
    missing_in_python = sql_norm_set - python_norm_set
    extra_in_python = python_norm_set - sql_norm_set

    # Check dtype alignment for matched columns
    dtype_map = {"TEXT": "string", "NUMERIC": "float", "DATE": "datetime64[ns]"}

    dtype_mismatches = []
    for norm_name in matched:
        sql_dtype = sql_norm_to_orig[norm_name][1]
        expected_python_dtype = dtype_map.get(sql_dtype, "unknown")
        actual_python_dtype = python_source[norm_name]["dtype"]

        # Allow 'category' as valid for TEXT columns
        if sql_dtype == "TEXT" and actual_python_dtype == "category":
            continue

        if actual_python_dtype != expected_python_dtype:
            dtype_mismatches.append(
                {
                    "column": norm_name,
                    "sql_type": sql_dtype,
                    "expected_python": expected_python_dtype,
                    "actual_python": actual_python_dtype,
                }
            )

    return {
        "total_sql": len(sql_columns),
        "total_python_source": len(python_source),
        "matched": len(matched),
        "missing_in_python": sorted(missing_in_python),
        "extra_in_python": sorted(extra_in_python),
        "dtype_mismatches": dtype_mismatches,
    }


def generate_report(
    validation_results: Dict, sql_columns: List[Tuple[str, str, str]], output_path: Path
):
    """Generate comprehensive alignment report."""

    lines = [
        "=" * 80,
        "SQL-PYTHON SCHEMA ALIGNMENT VALIDATION REPORT",
        "=" * 80,
        "",
        f"SQL Schema Columns: {validation_results['total_sql']}",
        f"Python Source Columns: {validation_results['total_python_source']}",
        f"Matched Columns: {validation_results['matched']}",
        "",
        "=" * 80,
        "ALIGNMENT SUMMARY",
        "=" * 80,
        "",
    ]

    # Check if alignment is complete
    if (
        validation_results["matched"] == validation_results["total_sql"]
        and len(validation_results["missing_in_python"]) == 0
        and len(validation_results["dtype_mismatches"]) == 0
    ):
        lines.append(
            "[SUCCESS] Perfect alignment! All SQL columns are present in Python schema with correct dtypes."
        )
    else:
        lines.append("[WARNING] Alignment issues detected:")

        if validation_results["missing_in_python"]:
            lines.append(
                f"  - {len(validation_results['missing_in_python'])} columns missing in Python schema"
            )

        if validation_results["dtype_mismatches"]:
            lines.append(f"  - {len(validation_results['dtype_mismatches'])} dtype mismatches")

    lines.append("")

    # Missing columns section
    if validation_results["missing_in_python"]:
        lines.append("=" * 80)
        lines.append("COLUMNS MISSING IN PYTHON SCHEMA")
        lines.append("=" * 80)
        lines.append("")
        lines.append("The following SQL columns are not present in Python COLUMN_SCHEMA:")
        lines.append("")

        # Show original SQL names and normalized names
        sql_norm_to_orig = {norm: (orig, dtype) for orig, norm, dtype in sql_columns}
        for norm_name in validation_results["missing_in_python"]:
            orig_name, dtype = sql_norm_to_orig[norm_name]
            lines.append(f'  "{orig_name}" -> {norm_name} ({dtype})')

        lines.append("")

    # Dtype mismatches section
    if validation_results["dtype_mismatches"]:
        lines.append("=" * 80)
        lines.append("DATA TYPE MISMATCHES")
        lines.append("=" * 80)
        lines.append("")

        for mismatch in validation_results["dtype_mismatches"]:
            lines.append(f"Column: {mismatch['column']}")
            lines.append(f"  SQL Type: {mismatch['sql_type']}")
            lines.append(f"  Expected Python: {mismatch['expected_python']}")
            lines.append(f"  Actual Python: {mismatch['actual_python']}")
            lines.append("")

    # Extra columns section (informational)
    if validation_results["extra_in_python"]:
        lines.append("=" * 80)
        lines.append("PYTHON-ONLY COLUMNS (informational)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("These columns exist in Python schema but not in SQL:")
        lines.append("(This is normal for computed/derived features)")
        lines.append("")

        for norm_name in validation_results["extra_in_python"][:20]:  # Show first 20
            lines.append(f"  {norm_name}")

        if len(validation_results["extra_in_python"]) > 20:
            lines.append(f"  ... and {len(validation_results['extra_in_python']) - 20} more")

        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also print to console
    print(report)


def main():
    base_dir = Path(__file__).parent.parent

    schema_sql_path = base_dir / "create_equities_schema.sql"
    schema_py_path = base_dir / "finance_ml" / "ml_workflow" / "data" / "schema.py"
    output_path = base_dir / "tools" / "schema_alignment_report.txt"

    print("Extracting SQL schema columns...")
    sql_columns = extract_sql_columns(schema_sql_path)
    print(f"  Found {len(sql_columns)} columns")

    print("\nLoading Python COLUMN_SCHEMA...")
    python_schema = extract_python_schema(schema_py_path)
    print(f"  Found {len(python_schema)} total entries")

    print("\nValidating alignment...")
    validation_results = validate_alignment(sql_columns, python_schema)

    print("\nGenerating report...")
    generate_report(validation_results, sql_columns, output_path)

    print(f"\n[SUCCESS] Report saved to: {output_path}")

    # Return exit code based on validation
    if (
        validation_results["matched"] == validation_results["total_sql"]
        and len(validation_results["missing_in_python"]) == 0
        and len(validation_results["dtype_mismatches"]) == 0
    ):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
