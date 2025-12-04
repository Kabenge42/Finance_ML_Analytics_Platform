"""Analyze notebook for Phase 9.5 and 9.6 sections."""

import json

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]
print(f"Total cells: {len(cells)}\n")

# Find cells mentioning 9.5, 9.6, regression, evaluation
print("=" * 80)
print("CELLS MENTIONING PHASE 9.5, 9.6, REGRESSION, OR EVALUATION")
print("=" * 80)

keywords = [
    "9.5",
    "9.6",
    "regression",
    "Regression",
    "evaluation",
    "Evaluation",
    "Phase 9.5",
    "Phase 9.6",
]

for i, cell in enumerate(cells):
    source = "".join(cell["source"])
    for kw in keywords:
        if kw in source:
            first_line = cell["source"][0][:100] if cell["source"] else "(empty)"
            print(f"\nCell {i} ({cell['cell_type']}):")
            print(f"  First line: {first_line}")
            break

# Also print markdown headers to understand structure
print("\n" + "=" * 80)
print("MARKDOWN HEADERS (## or ###)")
print("=" * 80)

for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        lines = source.split("\n")
        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                print(f"Cell {i}: {line[:80]}")
                break
