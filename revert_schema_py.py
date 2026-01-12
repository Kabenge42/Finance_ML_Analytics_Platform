import os
import re
import sys

# Add the project root to sys.path to import finance_ml
sys.path.append(os.getcwd())

with open("finance_ml/core/schema.py", "r") as f:
    content = f.read()

# Pattern to find entries in COLUMN_SCHEMA
pattern = r'(\s*"([^"]+)"\s*:\s*\{[^\}]+\})'


def replacer(match):
    entry = match.group(1)
    # Revert float -> Float64, int -> Int64
    entry = entry.replace('"float"', '"Float64"')
    entry = entry.replace('"int"', '"Int64"')
    return entry


new_content = re.sub(pattern, replacer, content)

with open("finance_ml/core/schema.py", "w") as f:
    f.write(new_content)
