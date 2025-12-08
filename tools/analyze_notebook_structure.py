"""Analyze notebook structure for refactoring."""

import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

code_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]
markdown_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "markdown"]

print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(markdown_cells)}")

print("\n=== First 30 Code Cells Overview ===")
for idx, (cell_idx, cell) in enumerate(code_cells[:30]):
    source = "".join(cell["source"])
    first_line = source.split("\n")[0][:80] if source else "(empty)"
    print(f"Cell {cell_idx}: {first_line}")

print("\n=== Looking for issues in code cells ===")
issues = {
    "broad_exception": [],
    "missing_type_hints": [],
    "unused_imports_candidates": [],
    "deprecated_4step": [],
}

for cell_idx, cell in code_cells:
    source = "".join(cell["source"])

    # Check for broad exception
    if "except Exception" in source or "except:" in source:
        issues["broad_exception"].append(cell_idx)

    # Check for deprecated 4-step imputation
    if "4step" in source.lower() or "four_step" in source.lower():
        issues["deprecated_4step"].append(cell_idx)

    # Check for functions without type hints (simple heuristic)
    if "def " in source:
        lines = source.split("\n")
        for line in lines:
            if line.strip().startswith("def ") and "->" not in line:
                issues["missing_type_hints"].append(cell_idx)
                break

print(f"\nBroad exception cells: {issues['broad_exception']}")
print(f"Missing type hints cells: {issues['missing_type_hints']}")
print(f"Deprecated 4step cells: {issues['deprecated_4step']}")
