"""Show the exact source of the imports from finance_ml.eval."""

import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][4]
source = "".join(cell.get("source", []))

# Find the finance_ml.eval import section
lines = source.split("\n")
in_eval_import = False
import_lines = []

for i, line in enumerate(lines):
    if "from finance_ml.eval import" in line:
        in_eval_import = True
    if in_eval_import:
        import_lines.append((i, line))
        if ")" in line and not line.strip().startswith("#"):
            # Check if this closes the import
            # Count parens to see if we're done
            paren_count = sum(1 for c in line if c == "(") - sum(1 for c in line if c == ")")
            if paren_count <= 0:
                break

print("finance_ml.eval import section:")
print("=" * 80)
for line_num, line in import_lines:
    print(f"{line_num:3d}: {line}")
print("=" * 80)
print(f"Total lines in import: {len(import_lines)}")
