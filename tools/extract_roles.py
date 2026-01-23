import os
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

from finance_ml.core.schema import COLUMN_SCHEMA

roles_to_zero_fill = ["financial_statement", "balance_sheet", "cash_flow", "count", "non_recurring"]

for col, meta in COLUMN_SCHEMA.items():
    if meta.get("role") in roles_to_zero_fill:
        print(
            f"Column: {col}, Role: {meta.get('role')}, DType: {meta.get('dtype')}, SQL: {meta.get('sql_name')}"
        )
