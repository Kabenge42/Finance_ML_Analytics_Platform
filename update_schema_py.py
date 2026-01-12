import os
import re
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

roles_to_zero_fill = ["financial_statement", "balance_sheet", "cash_flow", "count", "non_recurring"]

with open("finance_ml/core/schema.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_column_schema = False
current_column = None
current_meta = {}

# This is a bit complex to do with simple regex because it's a nested dict.
# I'll use a state-machine-like approach or just regex on blocks.

content = "".join(lines)

# Pattern to find entries in COLUMN_SCHEMA
# "column_name": { ... }
pattern = r'(\s*"([^"]+)"\s*:\s*\{[^\}]+\})'


def replacer(match):
    entry = match.group(1)

    # Extract role
    role_match = re.search(r'"role"\s*:\s*"([^"]+)"', entry)
    if role_match:
        role = role_match.group(1)
        if role in roles_to_zero_fill:
            # Update dtype
            entry = entry.replace('"Float64"', '"float"')
            entry = entry.replace('"Int64"', '"int"')
    return entry


new_content = re.sub(pattern, replacer, content)

with open("finance_ml/core/schema.py", "w") as f:
    f.write(new_content)
