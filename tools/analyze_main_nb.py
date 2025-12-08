"""Analyze ml_finance_model_main.ipynb structure."""

import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

code_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]
print(f"Code cells: {len(code_cells)}")

# Show first 20 code cells with their content preview
print("\n=== First 20 Code Cells ===")
for idx, (i, cell) in enumerate(code_cells[:20]):
    source = "".join(cell["source"])
    preview = source[:100].replace("\n", " ")
    print(f"Cell {i}: {preview}...")

# Look for common issues
print("\n=== Issue Detection ===")

# Find cells with broad exception handling
broad_except = []
# Find cells with unused imports hints
unused_imports = []
# Find function definitions without type hints
no_type_hints = []

for i, cell in code_cells:
    source = "".join(cell["source"])

    if "except Exception" in source or "except:" in source:
        broad_except.append(i)

    if "def " in source and "(" in source:
        # Check if function has type hints
        lines = source.split("\n")
        for line in lines:
            if line.strip().startswith("def ") and "->" not in line:
                no_type_hints.append(i)
                break

print(f"Cells with broad exception handling: {broad_except}")
print(f"Cells with functions missing return type hints: {no_type_hints[:20]}...")
