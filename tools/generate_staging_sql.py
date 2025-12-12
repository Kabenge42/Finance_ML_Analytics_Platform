#!/usr/bin/env python3
"""
Generate SQL staging table column definitions from CSV headers.

This script reads CSV headers and generates SQL CREATE TABLE statements
with all columns as TEXT type for safe initial loading, matching the
exact CSV column order.

Usage:
    python tools/generate_staging_sql.py
    python tools/generate_staging_sql.py --output staging_columns.sql
    python tools/generate_staging_sql.py --csv data/screening_us.csv
"""

import argparse
import csv
from pathlib import Path


def read_csv_header(csv_path: Path) -> list[str]:
    """Read the header row from a CSV file."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def generate_staging_table_sql(columns: list[str], table_name: str = "staging_raw") -> str:
    """
    Generate CREATE TABLE SQL with all TEXT columns.

    Args:
        columns: List of column names from CSV header
        table_name: Name for the staging table

    Returns:
        SQL CREATE TABLE statement
    """
    lines = [f"CREATE TEMP TABLE IF NOT EXISTS {table_name}"]
    lines.append("(")

    col_lines = []
    for col in columns:
        # Escape column name with double quotes for SQL
        col_lines.append(f'    "{col}" TEXT')

    lines.append(",\n".join(col_lines))
    lines.append(");")

    return "\n".join(lines)


def generate_insert_column_list(columns: list[str]) -> str:
    """
    Generate the column list for INSERT INTO equities SELECT ... statements.

    Args:
        columns: List of column names from CSV header

    Returns:
        Formatted column list for SELECT statement
    """
    lines = []
    for i, col in enumerate(columns):
        suffix = "," if i < len(columns) - 1 else ""
        lines.append(f'       "{col}"{suffix}')

    return "\n".join(lines)


def compare_with_schema(csv_columns: list[str], schema_path: Path) -> dict:
    """
    Compare CSV columns with SQL schema columns.

    Args:
        csv_columns: List of column names from CSV
        schema_path: Path to create_equities_schema.sql

    Returns:
        Dictionary with comparison results
    """
    # Extract column names from schema SQL
    schema_columns = []
    with schema_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('"') and (
                line.endswith("TEXT,")
                or line.endswith("TEXT")
                or line.endswith("NUMERIC,")
                or line.endswith("NUMERIC")
                or line.endswith("DATE,")
                or line.endswith("DATE")
            ):
                # Extract column name between quotes
                col_name = line.split('"')[1]
                schema_columns.append(col_name)

    csv_set = set(csv_columns)
    schema_set = set(schema_columns)

    return {
        "csv_count": len(csv_columns),
        "schema_count": len(schema_columns),
        "in_csv_not_schema": sorted(csv_set - schema_set),
        "in_schema_not_csv": sorted(schema_set - csv_set),
        "common": sorted(csv_set & schema_set),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL staging table definitions from CSV headers"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/screening_us.csv"),
        help="Path to CSV file (default: data/screening_us.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for SQL (default: print to stdout)",
    )
    parser.add_argument(
        "--table-name",
        default="screening_raw",
        help="Name for staging table (default: screening_raw)",
    )
    parser.add_argument(
        "--compare-schema",
        type=Path,
        help="Path to create_equities_schema.sql for comparison",
    )
    parser.add_argument(
        "--insert-list",
        action="store_true",
        help="Generate INSERT column list instead of CREATE TABLE",
    )

    args = parser.parse_args()

    # Read CSV header
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return 1

    columns = read_csv_header(args.csv)
    print(f"Read {len(columns)} columns from {args.csv}")

    # Compare with schema if requested
    if args.compare_schema:
        if args.compare_schema.exists():
            comparison = compare_with_schema(columns, args.compare_schema)
            print(f"\n=== Schema Comparison ===")
            print(f"CSV columns: {comparison['csv_count']}")
            print(f"Schema columns: {comparison['schema_count']}")
            print(f"\nIn CSV but not in schema ({len(comparison['in_csv_not_schema'])}):")
            for col in comparison["in_csv_not_schema"]:
                print(f"  + {col}")
            print(f"\nIn schema but not in CSV ({len(comparison['in_schema_not_csv'])}):")
            for col in comparison["in_schema_not_csv"]:
                print(f"  - {col}")
        else:
            print(f"Warning: Schema file not found: {args.compare_schema}")

    # Generate SQL
    if args.insert_list:
        sql = generate_insert_column_list(columns)
    else:
        sql = generate_staging_table_sql(columns, args.table_name)

    # Output
    if args.output:
        args.output.write_text(sql, encoding="utf-8")
        print(f"\nSQL written to {args.output}")
    else:
        print(f"\n=== Generated SQL ===\n")
        print(sql)

    return 0


if __name__ == "__main__":
    exit(main())
