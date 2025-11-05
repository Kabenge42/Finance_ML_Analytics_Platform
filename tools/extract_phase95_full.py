#!/usr/bin/env python3
"""Extract Phase 9.5 cell from notebook."""
import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cells around line 4032 (approximately cell index based on cumulative lines)
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "Phase 9.5: Advanced Regression" in source and len(source) > 20000:
        print(f"Found Phase 9.5 cell at index {i}")
        with open("phase95_full_code.txt", "w", encoding="utf-8") as out:
            out.write(source)
        print(f"Extracted {len(source)} characters to phase95_full_code.txt")
        break
