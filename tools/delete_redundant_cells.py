#!/usr/bin/env python3
"""
Delete redundant Phase 9.2 cells (32-35) from notebook.

According to PHASE92_RESTRUCTURING_SUMMARY.md:
- Cell 32: Redundant markdown header
- Cell 33: Old eda_summary() call (replaced by Cell 25)
- Cell 34: Removed correlation heatmap (replaced by Cell 23)
- Cell 35: Old sector_distribution_summary() call (replaced by Cell 24)
"""
import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
print(f"Loading {notebook_path}...")
with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

total_cells_before = len(nb["cells"])
print(f"Total cells before: {total_cells_before}")

# Show what we're deleting
print("\nCells to delete (32-35):")
for i in range(32, 36):
    cell = nb["cells"][i]
    source = cell.get("source", [])
    first_line = source[0] if source else "(empty)"
    if isinstance(first_line, str):
        first_line = first_line[:80].replace("\n", " ")
    print(f"  Cell {i}: {cell['cell_type']:8s} | {first_line}")

# Delete cells 32-35 (4 cells)
# Remove in reverse order to maintain indices
for i in reversed(range(32, 36)):
    del nb["cells"][i]

total_cells_after = len(nb["cells"])
print(f"\nTotal cells after: {total_cells_after}")
print(f"Cells removed: {total_cells_before - total_cells_after}")

# Save updated notebook
print(f"\nSaving updated notebook to {notebook_path}...")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook updated successfully!")
print("\nPhase 9.2 structure (cells 20-25):")
for i in range(20, min(26, len(nb["cells"]))):
    cell = nb["cells"][i]
    source = cell.get("source", [])
    first_line = source[0] if source else "(empty)"
    if isinstance(first_line, str):
        first_line = first_line[:80].replace("\n", " ")
    print(f"  Cell {i}: {cell['cell_type']:8s} | {first_line}")
