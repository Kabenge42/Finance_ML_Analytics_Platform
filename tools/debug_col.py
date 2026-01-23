import os
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

from finance_ml.core.schema import COLUMN_SCHEMA

sql_name = "Gain (Loss) On Sale Of Assets (LTM)"
role = None
for col, meta in COLUMN_SCHEMA.items():
    if meta.get("sql_name") == sql_name:
        role = meta.get("role")
        print(f"Found: {col}, Role: {role}")

if not role:
    print("Not found exactly. Checking normalized.")
    from finance_ml.core.schema import normalize_column_name

    normalized = normalize_column_name(sql_name)
    print(f"Normalized: {normalized}")
    if normalized in COLUMN_SCHEMA:
        print(f"Role in SCHEMA for normalized: {COLUMN_SCHEMA[normalized].get('role')}")
