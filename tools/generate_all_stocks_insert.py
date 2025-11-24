"""
Generate SQL INSERT statement for all_stocks_raw.sql with proper type casting.

This script reads the CREATE TABLE statement from all_stocks_raw.sql and generates:
1. INSERT INTO all_stocks_raw column list
2. Four SELECT statements (US, EU, APAC, ROTW) with proper type casting
3. Handles DATE, NUMERIC, TEXT, and INTEGER types appropriately

Type casting rules:
- DATE: to_date(NULLIF("Column",''), 'YYYY-MM-DD')
- NUMERIC: NULLIF("Column",'')::numeric (for TEXT sources) or "Column"::numeric
- TEXT: "Column"::text
- INTEGER: "Column"::integer
- Region: Hardcoded to 'US', 'EU', 'APAC', 'ROTW' for each table
"""

import re
from pathlib import Path


def extract_columns_from_sql(sql_file_path):
    """Extract column definitions from CREATE TABLE statement."""
    sql_content = Path(sql_file_path).read_text(encoding="utf-8")

    # Find CREATE TABLE all_stocks_raw section
    create_pattern = r"CREATE TABLE all_stocks_raw\s*\((.*?)\s*CONSTRAINT"
    match = re.search(create_pattern, sql_content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError("Could not find CREATE TABLE all_stocks_raw statement")

    table_def = match.group(1)

    # Extract column definitions: "Column Name" TYPE
    # Pattern: quoted name, followed by whitespace, followed by type (TEXT, NUMERIC, DATE, INTEGER)
    column_pattern = r'"([^"]+)"\s+(TEXT|NUMERIC|DATE|INTEGER)'
    columns = re.findall(column_pattern, table_def, re.IGNORECASE)

    return [(name, dtype.upper()) for name, dtype in columns]


def generate_cast_expression(column_name, dtype, region_value=None):
    """Generate proper type casting expression for a column."""
    quoted_col = f'"{column_name}"'

    # Special handling for Region column
    if column_name == "Region":
        return f"'{region_value}'::text"

    # DATE columns need to_date with NULLIF for empty strings
    if dtype == "DATE":
        return f"to_date(NULLIF({quoted_col},''), 'YYYY-MM-DD')"

    # NUMERIC columns need NULLIF for empty strings (TEXT sources)
    elif dtype == "NUMERIC":
        return f"NULLIF({quoted_col},'')::numeric"

    # TEXT columns - direct cast
    elif dtype == "TEXT":
        return f"{quoted_col}::text"

    # INTEGER columns
    elif dtype == "INTEGER":
        return f"NULLIF({quoted_col},'')::integer"

    else:
        # Default: direct reference
        return quoted_col


def generate_insert_sql(columns, output_file=None):
    """Generate complete INSERT INTO statement with UNION ALL."""

    # Generate INSERT column list
    insert_columns = ",\n  ".join([f'"{col}"' for col, _ in columns])

    # Generate SELECT for each region
    regions = [
        ("US", "postgres.public.screening_us"),
        ("EU", "postgres.public.screening_eu"),
        ("APAC", "postgres.public.screening_apac"),
        ("ROTW", "postgres.public.screening_rotw"),
    ]

    select_statements = []

    for region_name, table_name in regions:
        # Generate casted column list for this region
        casted_columns = []
        for col_name, col_type in columns:
            cast_expr = generate_cast_expression(col_name, col_type, region_name)
            casted_columns.append(cast_expr)

        # Format SELECT statement with 4 columns per line for readability
        select_cols = []
        for i in range(0, len(casted_columns), 4):
            line_cols = casted_columns[i : i + 4]
            select_cols.append(", ".join(line_cols))

        select_body = ",\n  ".join(select_cols)

        select_stmt = f"SELECT\n  {select_body}\nFROM {table_name}"
        select_statements.append(select_stmt)

    # Combine with UNION ALL
    union_sql = "\nUNION ALL\n".join(select_statements)

    # Complete INSERT statement
    full_sql = f"""-- Insert data from regional tables using UNION ALL with explicit type casting
-- This ensures type compatibility across regional tables with different schemas
INSERT INTO all_stocks_raw (
  {insert_columns}
)
{union_sql};"""

    if output_file:
        Path(output_file).write_text(full_sql, encoding="utf-8")
        print(f"Generated SQL written to: {output_file}")

    return full_sql


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Correct path to use 'all_stocks' directory instead of 'all_stocks_raw'
    sql_file = project_root / "all_stocks" / "all_stocks_raw.sql"
    output_file = project_root / "all_stocks" / "insert_statement_generated.sql"

    print(f"Reading column definitions from: {sql_file}")
    columns = extract_columns_from_sql(sql_file)
    print(f"Found {len(columns)} columns")

    print("\nGenerating INSERT statement with type casting...")
    sql = generate_insert_sql(columns, output_file)

    print(f"\nGenerated SQL has {len(sql.splitlines())} lines")
    print("\nFirst 20 lines of generated SQL:")
    print("\n".join(sql.splitlines()[:20]))

    print("\n" + "=" * 80)
    print("IMPORTANT: Review the generated SQL and replace lines 415-428 in all_stocks_raw.sql")
    print("=" * 80)


if __name__ == "__main__":
    main()
