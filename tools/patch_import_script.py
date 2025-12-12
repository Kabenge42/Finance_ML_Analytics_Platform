#!/usr/bin/env python3
"""
Patch import_equities_data.sql to add proper type casting to all INSERT statements.

This script reads the generated typed_insert_statements.sql and replaces the
4 regional INSERT statements in import_equities_data.sql.
"""

import re
from pathlib import Path


def extract_insert_statements(typed_sql_path: Path) -> dict:
    """Extract the 4 regional INSERT statements from typed_insert_statements.sql"""
    with open(typed_sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by region markers
    regions = {}

    # Pattern to match each INSERT statement
    pattern = r"-- (SCREENING_\w+) REGION\s*-+\s*(INSERT INTO equities.*?ON CONFLICT DO NOTHING;)"

    for match in re.finditer(pattern, content, re.DOTALL):
        region_name = match.group(1).lower()
        insert_stmt = match.group(2)
        regions[region_name] = insert_stmt

    return regions


def patch_import_script(import_sql_path: Path, typed_inserts: dict, output_path: Path):
    """
    Replace the 4 INSERT statements in import_equities_data.sql with typed versions.
    """
    with open(import_sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    # For each region, find and replace the INSERT statement
    for region, typed_insert in typed_inserts.items():
        # Pattern to match existing INSERT statement for this region
        # The pattern looks for:
        # - "Insert into main equities table" comment
        # - INSERT INTO equities
        # - SELECT ... FROM region_table
        # - ON CONFLICT DO NOTHING;

        table_name = region  # e.g., 'screening_us'

        # Find the INSERT statement for this region
        pattern = (
            r"-- Insert into main equities table\s*"
            r"-- Columns are selected.*?\s*"
            r"INSERT INTO equities\s+"
            r"SELECT.*?"
            rf"FROM {table_name}\s+"
            r"ON CONFLICT DO NOTHING;"
        )

        matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))

        if not matches:
            print(f"  WARNING: Could not find INSERT statement for {table_name}")
            continue

        if len(matches) > 1:
            print(
                f"  WARNING: Found {len(matches)} INSERT statements for {table_name}, replacing first one"
            )

        match = matches[0]
        old_insert = match.group(0)

        # Create the new INSERT with proper comments
        new_insert = f"""-- Insert into main equities table
-- Columns are selected in the order defined in equities table schema
-- DATE and NUMERIC columns are cast from TEXT with NULLIF to handle empty strings
{typed_insert}"""

        # Replace
        content = content.replace(old_insert, new_insert, 1)
        print(f"  [OK] Replaced INSERT statement for {table_name}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    base_dir = Path(__file__).parent.parent

    typed_sql_path = base_dir / "tools" / "typed_insert_statements.sql"
    import_sql_path = base_dir / "import_equities_data.sql"
    output_path = base_dir / "import_equities_data_fixed.sql"

    print("Extracting typed INSERT statements...")
    typed_inserts = extract_insert_statements(typed_sql_path)
    print(f"  Found {len(typed_inserts)} regional INSERT statements")

    print("\nPatching import_equities_data.sql...")
    patch_import_script(import_sql_path, typed_inserts, output_path)

    print(f"\n[SUCCESS] Created fixed import script: {output_path}")
    print("\nTo use the fixed script:")
    print(f"  1. Review the changes: diff import_equities_data.sql import_equities_data_fixed.sql")
    print(f"  2. Backup original: copy import_equities_data.sql import_equities_data_backup.sql")
    print(f"  3. Replace original: copy import_equities_data_fixed.sql import_equities_data.sql")
    print(
        f"  4. Run import: psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql"
    )


if __name__ == "__main__":
    main()
