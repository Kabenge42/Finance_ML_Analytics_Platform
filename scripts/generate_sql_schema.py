#!/usr/bin/env python
"""
Generate SQL schema files from COLUMN_SCHEMA.

This ensures SQL and Python schemas are always synchronized.
Run: python scripts/generate_sql_schema.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))

from finance_ml.core.schema import COLUMN_SCHEMA, generate_sql_schema

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    # Generate CREATE TABLE
    schema_sql = generate_sql_schema()
    schema_path = PROJECT_ROOT / "create_equities_schema.sql"
    schema_path.write_text(schema_sql)
    print(f"✓ Generated: {schema_path}")

    # Generate IMPORT statement
    import_sql = generate_import_sql()
    import_path = PROJECT_ROOT / "import_equities_data.sql"
    import_path.write_text(import_sql)
    print(f"✓ Generated: {import_path}")

def generate_import_sql() -> str:
    """Generate COPY FROM CSV statement with column mapping."""
    columns = [meta.get("sql_name") or col for col, meta in COLUMN_SCHEMA.items()]
    return f"""
-- Auto-generated from COLUMN_SCHEMA
COPY equities ({', '.join(f'"{c}"' for c in columns)})
FROM '/path/to/data.csv'
WITH (FORMAT csv, HEADER true, NULL '');
"""

if __name__ == "__main__":
    main()
