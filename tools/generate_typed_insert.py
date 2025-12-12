#!/usr/bin/env python3
"""
Generate type-casted INSERT statements for import_equities_data.sql

This script parses create_equities_schema.sql to extract column names and data types,
then generates properly casted SELECT statements for the INSERT operations.

Addresses the critical type mismatch error:
  ERROR: column "Market Cap" is of type numeric but expression is of type text
"""

import re
from pathlib import Path
from typing import List, Tuple


def parse_schema_columns(schema_path: Path) -> List[Tuple[str, str]]:
    """
    Parse create_equities_schema.sql to extract (column_name, data_type) tuples.

    Returns:
        List of (column_name, data_type) where data_type is 'TEXT', 'NUMERIC', or 'DATE'
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the CREATE TABLE statement
    create_table_match = re.search(
        r"CREATE TABLE equities\s*\((.*?)\)\s*TABLESPACE", content, re.DOTALL | re.IGNORECASE
    )

    if not create_table_match:
        raise ValueError("Could not find CREATE TABLE equities statement")

    table_def = create_table_match.group(1)

    # Parse column definitions
    # Pattern: "Column Name"  TYPE, -- optional comment
    column_pattern = re.compile(r'"([^"]+)"\s+(TEXT|NUMERIC|DATE)', re.IGNORECASE)

    columns = []
    for match in column_pattern.finditer(table_def):
        col_name = match.group(1)
        col_type = match.group(2).upper()
        columns.append((col_name, col_type))

    return columns


def generate_casted_select(columns: List[Tuple[str, str]], indent: str = "       ") -> str:
    """
    Generate the SELECT clause with proper type casting.

    Args:
        columns: List of (column_name, data_type) tuples
        indent: Indentation string for formatting

    Returns:
        Multi-line SELECT clause with proper casting
    """
    select_lines = []

    for i, (col_name, col_type) in enumerate(columns):
        # Determine the appropriate cast expression
        if col_type == "TEXT":
            # No casting needed for TEXT columns
            expr = f'"{col_name}"'
        elif col_type == "DATE":
            # Cast with NULLIF for DATE columns
            expr = f"NULLIF(\"{col_name}\", '')::DATE"
        elif col_type == "NUMERIC":
            # Cast with NULLIF for NUMERIC columns
            expr = f"NULLIF(\"{col_name}\", '')::NUMERIC"
        else:
            raise ValueError(f"Unknown column type: {col_type}")

        # Add comma except for last column
        if i < len(columns) - 1:
            select_lines.append(f"{indent}{expr},")
        else:
            select_lines.append(f"{indent}{expr}")

    return "\n".join(select_lines)


def generate_column_list(columns: List[Tuple[str, str]], indent: str = "    ") -> str:
    """
    Generate the column list for INSERT INTO statement.

    Args:
        columns: List of (column_name, data_type) tuples
        indent: Indentation string for formatting

    Returns:
        Multi-line column list
    """
    col_lines = []

    for i, (col_name, _) in enumerate(columns):
        if i < len(columns) - 1:
            col_lines.append(f'{indent}"{col_name}",')
        else:
            col_lines.append(f'{indent}"{col_name}"')

    return "\n".join(col_lines)


def generate_insert_statement(columns: List[Tuple[str, str]], staging_table: str) -> str:
    """
    Generate complete INSERT statement with type casting.

    Args:
        columns: List of (column_name, data_type) tuples
        staging_table: Name of the staging table (e.g., 'screening_us')

    Returns:
        Complete INSERT INTO ... SELECT ... statement
    """
    col_list = generate_column_list(columns)
    select_clause = generate_casted_select(columns)

    insert_stmt = f"""INSERT INTO equities
(
{col_list}
)
SELECT
{select_clause}
FROM {staging_table}
ON CONFLICT DO NOTHING;"""

    return insert_stmt


def main():
    """Generate all 4 regional INSERT statements with proper type casting."""

    # Paths
    base_dir = Path(__file__).parent.parent
    schema_path = base_dir / "create_equities_schema.sql"
    output_path = base_dir / "tools" / "typed_insert_statements.sql"

    print(f"Reading schema from: {schema_path}")

    # Parse schema
    columns = parse_schema_columns(schema_path)
    print(f"Parsed {len(columns)} columns from schema")

    # Count column types
    type_counts = {"TEXT": 0, "NUMERIC": 0, "DATE": 0}
    for _, col_type in columns:
        type_counts[col_type] += 1

    print(f"Column type breakdown:")
    print(f"  TEXT: {type_counts['TEXT']}")
    print(f"  NUMERIC: {type_counts['NUMERIC']}")
    print(f"  DATE: {type_counts['DATE']}")

    # Generate INSERT statements for all 4 regions
    regions = ["screening_us", "screening_eu", "screening_apac", "screening_rotw"]

    output_lines = [
        "-- Generated INSERT statements with proper type casting",
        '-- This fixes the ERROR: column "Market Cap" is of type numeric but expression is of type text',
        "",
        f"-- Schema contains {len(columns)} columns:",
        f"--   {type_counts['TEXT']} TEXT columns (no cast needed)",
        f"--   {type_counts['DATE']} DATE columns (cast with NULLIF)",
        f"--   {type_counts['NUMERIC']} NUMERIC columns (cast with NULLIF)",
        "",
    ]

    for region in regions:
        output_lines.append(f"-- {region.upper()} REGION")
        output_lines.append("-" * 80)
        output_lines.append(generate_insert_statement(columns, region))
        output_lines.append("")
        output_lines.append("")

    # Write output
    output_content = "\n".join(output_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\nGenerated typed INSERT statements: {output_path}")
    print(f"Total size: {len(output_content):,} characters")

    # Also print summary statistics
    print("\nSample casted expressions:")
    for col_name, col_type in columns[:5]:
        if col_type == "TEXT":
            print(f'  "{col_name}" (TEXT - no cast)')
        elif col_type == "DATE":
            print(f"  NULLIF(\"{col_name}\", '')::DATE")
        elif col_type == "NUMERIC":
            print(f"  NULLIF(\"{col_name}\", '')::NUMERIC")


if __name__ == "__main__":
    main()
