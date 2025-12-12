#!/usr/bin/env python3
"""
generate_schema_columns.py
Auto-generate SQL column definitions for create_equities_schema.sql from CSV headers.

This script reads CSV headers from the data/ directory and generates:
1. SQL column definitions for CREATE TABLE statements
2. Column lists for INSERT statements (import_equities_data.sql)
3. Python COLUMN_SCHEMA entries for schema.py

Usage:
    python tools/generate_schema_columns.py
    python tools/generate_schema_columns.py --filter "employee"
    python tools/generate_schema_columns.py --output sql
    python tools/generate_schema_columns.py --output python
    python tools/generate_schema_columns.py --output insert
"""

import csv
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

DATA_DIR = Path("data")
FILES = [
    "screening_us.csv",
    "screening_eu.csv",
    "screening_apac.csv",
    "screening_rotw.csv",
]

# Known column types based on patterns
DATE_PATTERNS = [
    r"date",
    r"last updated",
    r"next earnings$",
    r"report date",
]

TEXT_PATTERNS = [
    r"^ticker$",
    r"^isin$",
    r"^name$",
    r"^description$",
    r"^exchange$",
    r"^unit$",
    r"^sector$",
    r"^industry$",
    r"^style class$",
    r"^size class$",
    r"^region$",
    r"^country$",
    r"^trading country$",
    r"^flag$",
    r"status\)$",
    r"frequency\)$",
    r"currency\)$",
]

# Semantic categories for comments
PRICE_PATTERNS = [
    r"^last price$",
    r"^price target",
    r"^price \(",
    r"^ema \(",
    r"52w high",
    r"52w low",
    r"dividend.*amount",
]

MARKET_VALUE_PATTERNS = [
    r"^market cap",
    r"^enterprise value",
    r"^total revenues",
    r"^ebitda",
    r"^ebit(?!da)",
    r"^net income",
    r"^total assets",
    r"^total debt",
    r"^total equity",
    r"^tbv ",
    r"^cfo ",
    r"^cfi ",
    r"^cff ",
    r"^fcf ",
    r"^gross profit",
    r"^operating income",
]

RATIO_PATTERNS = [
    r"^p/e ",
    r"^p/b ",
    r"^p/tbv",
    r"^ev/",
    r"^return on",
    r"^current ratio",
    r"^asset turnover",
    r"altman z-score",
]

PERCENTAGE_PATTERNS = [
    r"margin %",
    r"return.*%",
    r"yield",
    r"volatility",
    r"beta",
    r"cagr",
    r"price chg",
    r"1-day %",
    r"total return",
]

COUNT_PATTERNS = [
    r"# .* ratings",
    r"analyst rating",
    r"price target - #",
    r"employees",
    r"dividend streak",
    r"shrs out",
]


def read_header(path: Path) -> List[str]:
    """Read CSV header row."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def normalize_column_name(col: str) -> str:
    """Convert CSV column name to Python-style normalized name."""
    # Replace special chars with underscores
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    # Strip leading/trailing underscores
    normalized = normalized.strip("_")
    # Convert to lowercase
    return normalized.lower()


def get_sql_type(col: str) -> str:
    """Determine SQL type for a column based on patterns."""
    col_lower = col.lower()

    for pattern in DATE_PATTERNS:
        if re.search(pattern, col_lower):
            return "DATE"

    for pattern in TEXT_PATTERNS:
        if re.search(pattern, col_lower):
            return "TEXT"

    return "NUMERIC"


def get_semantic_category(col: str) -> str:
    """Determine semantic category for a column."""
    col_lower = col.lower()

    for pattern in PRICE_PATTERNS:
        if re.search(pattern, col_lower):
            return "PRICE"

    for pattern in MARKET_VALUE_PATTERNS:
        if re.search(pattern, col_lower):
            return "MARKET_VALUE"

    for pattern in RATIO_PATTERNS:
        if re.search(pattern, col_lower):
            return "RATIO"

    for pattern in PERCENTAGE_PATTERNS:
        if re.search(pattern, col_lower):
            return "PERCENTAGE"

    for pattern in COUNT_PATTERNS:
        if re.search(pattern, col_lower):
            return "COUNT"

    return "OTHER"


def generate_sql_column(col: str, max_len: int = 50) -> str:
    """Generate SQL column definition line."""
    sql_type = get_sql_type(col)
    category = get_semantic_category(col)

    # Pad column name for alignment
    quoted_name = f'"{col}"'
    padded = quoted_name.ljust(max_len)

    # Add category comment for NUMERIC columns
    if sql_type == "NUMERIC":
        return f"    {padded} {sql_type}, -- {category}: {col}"
    else:
        return f"    {padded} {sql_type},"


def generate_insert_column(col: str) -> str:
    """Generate column name for INSERT statement."""
    return f'       "{col}",'


def generate_python_schema_entry(col: str) -> str:
    """Generate Python COLUMN_SCHEMA entry."""
    normalized = normalize_column_name(col)
    sql_type = get_sql_type(col)

    # Map SQL type to Python dtype
    if sql_type == "DATE":
        dtype = "datetime"
    elif sql_type == "TEXT":
        dtype = "str"
    else:
        # Check if it's likely an integer (counts)
        category = get_semantic_category(col)
        if category == "COUNT" and "employees" in col.lower():
            dtype = "int"
        else:
            dtype = "float"

    return f'    "{normalized}": {{"dtype": "{dtype}", "role": "feature"}},'


def get_all_columns() -> Tuple[List[str], Dict[str, Set[str]]]:
    """Get union of all columns and track which files have each."""
    all_columns = []
    seen = set()
    file_columns = {}

    for fn in FILES:
        p = DATA_DIR / fn
        if not p.exists():
            print(f"Warning: {p} not found")
            continue

        headers = read_header(p)
        file_columns[fn] = set(headers)

        for col in headers:
            if col not in seen:
                all_columns.append(col)
                seen.add(col)

    return all_columns, file_columns


def filter_columns(columns: List[str], filter_term: str) -> List[str]:
    """Filter columns by keyword (case-insensitive)."""
    filter_lower = filter_term.lower()
    return [c for c in columns if filter_lower in c.lower()]


def main():
    parser = argparse.ArgumentParser(description="Generate SQL column definitions from CSV headers")
    parser.add_argument(
        "--filter", "-f", help="Filter columns containing this keyword (case-insensitive)"
    )
    parser.add_argument(
        "--output",
        "-o",
        choices=["sql", "insert", "python", "all"],
        default="all",
        help="Output format: sql (CREATE TABLE), insert (INSERT), python (COLUMN_SCHEMA), or all",
    )
    parser.add_argument("--data-dir", "-d", default="data", help="Directory containing CSV files")

    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)

    all_columns, file_columns = get_all_columns()

    if args.filter:
        columns = filter_columns(all_columns, args.filter)
        print(f"Filtered columns containing '{args.filter}': {len(columns)} matches\n")
    else:
        columns = all_columns
        print(f"Total unique columns: {len(columns)}\n")

    if not columns:
        print("No columns found matching filter.")
        return

    # Find max column name length for alignment
    max_len = max(len(f'"{c}"') for c in columns) + 2

    if args.output in ("sql", "all"):
        print("=" * 70)
        print("SQL CREATE TABLE column definitions:")
        print("=" * 70)
        for col in columns:
            print(generate_sql_column(col, max_len))
        print()

    if args.output in ("insert", "all"):
        print("=" * 70)
        print("SQL INSERT statement column list:")
        print("=" * 70)
        for col in columns:
            print(generate_insert_column(col))
        print()

    if args.output in ("python", "all"):
        print("=" * 70)
        print("Python COLUMN_SCHEMA entries:")
        print("=" * 70)
        for col in columns:
            print(generate_python_schema_entry(col))
        print()

    # Show which files have each column
    if args.filter:
        print("=" * 70)
        print("Column presence by file:")
        print("=" * 70)
        for col in columns:
            files_with = [fn for fn, cols in file_columns.items() if col in cols]
            print(f"  {col}")
            for fn in files_with:
                print(f"    ✓ {fn}")


if __name__ == "__main__":
    main()
