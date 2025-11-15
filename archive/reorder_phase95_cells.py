"""
Reorder Phase 9.5 cells to ensure correct logical flow:
Cell 139 (Phase 9.5 header) → Cell 145 (Phase 9.5 main) → Cells 140-141 (Phase 9.5.1) → Cells 142-144 (Phase 9.6.1) → Cells 149-150 (Phase 9.6)

Current order: 139, 140, 141, 142, 143, 144, 145, ..., 149, 150
Target order: 139, 145, 140, 141, 142, 143, 144, ..., 149, 150
"""

import json
import shutil
from datetime import datetime

# Backup
backup_filename = (
    f"ml_finance_model_main_backup.ipynb.backup_reorder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
shutil.copy("ml_finance_model_main_backup.ipynb", backup_filename)
print(f"✓ Backup created: {backup_filename}")

# Load notebook
with open("ml_finance_model_main_backup.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook["cells"]
print(f"\n📊 Current notebook: {len(cells)} cells")

# Identify cells to reorder
print("\n🔍 Current cell order (139-150):")
for i in range(139, min(151, len(cells))):
    source = "".join(cells[i].get("source", []))
    cell_type = cells[i].get("cell_type", "unknown")
    preview = source[:70].replace("\n", " ")
    print(f"  Cell {i} ({cell_type}): {preview}")

# Extract Cell 145 (Phase 9.5 main implementation)
cell_145 = cells[145]
cell_145_preview = "".join(cell_145.get("source", []))[:100]
print(f"\n📦 Extracting Cell 145 for reordering:")
print(f"   {cell_145_preview}...")

# Remove Cell 145 from its current position
del cells[145]
print("✓ Cell 145 removed from position 145")

# Insert Cell 145 after Cell 139 (which becomes position 140 after insertion)
cells.insert(140, cell_145)
print("✓ Cell 145 inserted at position 140 (right after Phase 9.5 header)")

# Verify new order
print("\n✅ New cell order (139-150):")
for i in range(139, min(151, len(cells))):
    source = "".join(cells[i].get("source", []))
    cell_type = cells[i].get("cell_type", "unknown")

    # Identify cell type
    if "Phase 9.5 —" in source and cell_type == "markdown" and "Classification Features" in source:
        label = "Phase 9.5 Header"
    elif "PHASE 9.5 — SECTOR-OPTIMIZED REGRESSION" in source and cell_type == "code":
        label = "Phase 9.5 MAIN (NEW)"
    elif "Phase 9.5.1" in source and cell_type == "markdown":
        label = "Phase 9.5.1 Header"
    elif "MODEL OPTIMIZATION ENHANCEMENTS" in source and cell_type == "code":
        label = "Phase 9.5.1 Code"
    elif "Phase 9.6.1" in source and cell_type == "markdown":
        label = "Phase 9.6.1 Header"
    elif "ENHANCED ERROR ANALYSIS" in source and cell_type == "code":
        label = "Phase 9.6.1 Code"
    elif "Phase 9.6 —" in source and cell_type == "markdown":
        label = "Phase 9.6 Header"
    elif "PHASE 9.6 — MODEL EVALUATION" in source and cell_type == "code":
        label = "Phase 9.6 Code"
    else:
        label = "Other"

    preview = source[:60].replace("\n", " ")
    print(f"  Cell {i} ({cell_type}) [{label}]: {preview}")

# Update notebook metadata
if "metadata" not in notebook:
    notebook["metadata"] = {}

if "restructured" not in notebook["metadata"]:
    notebook["metadata"]["restructured"] = {}

notebook["metadata"]["restructured"]["reordered"] = {
    "date": datetime.now().isoformat(),
    "change": "Moved Phase 9.5 main implementation (was Cell 145) to position 140 (after Phase 9.5 header)",
    "correct_order": "9.5 header → 9.5 main → 9.5.1 → 9.6.1 → 9.6",
}

# Save
with open("ml_finance_model_main_backup.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook reordered successfully!")
print(f"📊 Total cells: {len(cells)}")
print(f"\n📁 Files:")
print(f"  - Backup: {backup_filename}")
print(f"  - Updated: ml_finance_model_main_backup.ipynb")
print(f"\n✓ Correct logical flow achieved:")
print(f"   Cell 139: Phase 9.5 — Header")
print(f"   Cell 140: Phase 9.5 — MAIN IMPLEMENTATION (8 workflow steps)")
print(f"   Cell 141: Phase 9.5.1 — Optimization Enhancements Header")
print(f"   Cell 142: Phase 9.5.1 — Code")
print(f"   Cell 143: Phase 9.6.1 — Enhanced Error Analysis Header")
print(f"   Cell 144: Phase 9.6.1 — Code")
print(f"   Cell 145: Summary")
print(f"   Cell 149: Phase 9.6 — Model Evaluation Header")
print(f"   Cell 150: Phase 9.6 — Code")
