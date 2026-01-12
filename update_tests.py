import re

file_path = "tests/test_etl_unified_pipeline.py"
with open(file_path, "r") as f:
    content = f.read()

# Pattern to find dict entries in pd.DataFrame({ ... })
# and add unit and reference_date
# We look for the closing brace of the dict.

# This is a bit risky with regex.
# Let's try to match "num_analysts": ... and add after it.

pattern = r'("num_analysts"\s*:\s*[^,]+,)'
replacement = r'\1\n                "unit": ["USD"] * n,\n                "reference_date": [pd.Timestamp("2025-01-01")] * n,'

new_content = re.sub(pattern, replacement, content)

# Also handle cases where num_analysts is the last item (no comma)
pattern2 = r'("num_analysts"\s*:\s*[^,\n\s\}]+)(\s*\})'
replacement2 = r'\1,\n                "unit": ["USD"] * n,\n                "reference_date": [pd.Timestamp("2025-01-01")] * n\2'

new_content = re.sub(pattern2, replacement2, new_content)

with open(file_path, "w") as f:
    f.write(new_content)
