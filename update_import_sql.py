import os
import re
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

from finance_ml.core.schema import COLUMN_SCHEMA

roles_to_zero_fill = ["financial_statement", "balance_sheet", "cash_flow", "count", "non_recurring"]

# Map SQL name to Role
sql_name_to_role = {}
for col, meta in COLUMN_SCHEMA.items():
    sql_name = meta.get("sql_name")
    if sql_name:
        sql_name_to_role[sql_name] = meta.get("role")

with open("import_equities_data.sql", "r") as f:
    content = f.read()


def replacer(match):
    full_call = match.group(0)
    sql_col = match.group(1)
    role = sql_name_to_role.get(sql_col)

    if role in roles_to_zero_fill:
        return f'COALESCE(safe_to_numeric("{sql_col}"), 0)'
    return full_call


# Pattern to match: safe_to_numeric("Column Name")
pattern = r'safe_to_numeric\("([^"]+)"\)'
new_content = re.sub(pattern, replacer, content)

# Also handle Dividend Streak and others that might be handled differently if any
# Wait, safe_to_numeric already has quotes in it in some places or not?
# Let's check some examples from the file content I saw earlier.
# 1341:       safe_to_numeric("Dividend Record (Amount)")                            AS "Dividend Record (Amount)",
# 1343:       safe_to_numeric("Dividend Streak")                                     AS "Dividend Streak",

with open("import_equities_data.sql", "w") as f:
    f.write(new_content)
