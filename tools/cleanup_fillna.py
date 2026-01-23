import os
import re
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name

roles_to_zero_fill = ["financial_statement", "balance_sheet", "cash_flow", "count", "non_recurring"]


def should_remove_fillna(col_name):
    # Try exact match
    if col_name in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[col_name].get("role") in roles_to_zero_fill
    # Try normalized match
    norm_name = normalize_column_name(col_name)
    if norm_name in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[norm_name].get("role") in roles_to_zero_fill
    return False


file_path = "finance_ml/features/advanced/quality.py"
with open(file_path, "r") as f:
    content = f.read()

# Pattern for df["col"].fillna(0)
# Group 1: df["col"]
# Group 2: col
pattern = r'(df\["([^"]+)"\])\.fillna\(0\)'


def replacer(match):
    full_expr = match.group(0)
    df_col = match.group(1)
    col_name = match.group(2)

    if should_remove_fillna(col_name):
        print(f"Removing .fillna(0) for {col_name} in {file_path}")
        return df_col
    return full_expr


new_content = re.sub(pattern, replacer, content)

with open(file_path, "w") as f:
    f.write(new_content)
