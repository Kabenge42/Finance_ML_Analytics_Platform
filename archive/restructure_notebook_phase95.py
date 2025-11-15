"""
Restructure ml_finance_model_main_backup.ipynb to fix Phase 9.5 and 9.6 sections.

Changes:
1. Replace Cell 147 with comprehensive Phase 9.5 implementation
2. Remove redundant comment cells (143, 146, 148)
3. Ensure correct cell order: 9.5 main → 9.5.1 → 9.6 → 9.6.1 → summary
"""

import json
import shutil
from datetime import datetime

# Backup original notebook
backup_filename = (
    f"ml_finance_model_main_backup.ipynb.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
shutil.copy("ml_finance_model_main_backup.ipynb", backup_filename)
print(f"✓ Backup created: {backup_filename}")

# Load notebook
with open("ml_finance_model_main_backup.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook["cells"]
print(f"\n📊 Original notebook: {len(cells)} cells")

# Read new Phase 9.5 implementation
with open("phase_95_complete_implementation.py", "r", encoding="utf-8") as f:
    new_phase95_code = f.read()

# Create new cell with complete Phase 9.5 implementation
new_cell_147 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_phase95_code.split("\n"),
}

# Replace Cell 147 with new implementation
print("\n🔧 Replacing Cell 147 with comprehensive Phase 9.5 implementation...")
cells[147] = new_cell_147
print("✓ Cell 147 replaced")

# Remove redundant cells (143, 146, 148)
# Note: Remove in reverse order to maintain correct indices
print("\n🗑 Removing redundant comment cells...")
cells_to_remove = [148, 146, 143]  # Reverse order

for cell_idx in cells_to_remove:
    if cell_idx < len(cells):
        cell_type = cells[cell_idx].get("cell_type", "unknown")
        source_preview = "".join(cells[cell_idx].get("source", []))[:50]
        print(f"  Removing Cell {cell_idx} ({cell_type}): {source_preview}...")
        del cells[cell_idx]

print(f"✓ Removed {len(cells_to_remove)} redundant cells")

# Update cell order verification
print("\n📋 Verifying Phase 9.5 and 9.6 structure...")

# Find Phase 9.5/9.6 related cells
phase_cells = []
for i, cell in enumerate(cells):
    source = "".join(cell.get("source", []))
    if any(marker in source for marker in ["Phase 9.5", "Phase 9.6", "PHASE 9.5", "PHASE 9.6"]):
        cell_type = cell.get("cell_type", "unknown")
        preview = source[:80].replace("\n", " ")
        phase_cells.append((i, cell_type, preview))

print("\nPhase 9.5/9.6 cells after restructuring:")
for idx, ctype, preview in phase_cells[:20]:  # Show first 20
    print(f"  Cell {idx} ({ctype}): {preview}")

# Update notebook metadata
if "metadata" not in notebook:
    notebook["metadata"] = {}

notebook["metadata"]["restructured"] = {
    "date": datetime.now().isoformat(),
    "changes": [
        "Replaced Cell 147 with comprehensive Phase 9.5 implementation (8 workflow steps)",
        "Removed redundant comment cells (143, 146, 148)",
        "Integrated TDD enhancements (Huber loss, enhanced metadata)",
        "Added proper workflow: interaction features, model comparison, stacking, quantile regression, sector regression",
    ],
}

# Save restructured notebook
with open("ml_finance_model_main_backup.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook restructured successfully!")
print(f"📊 New notebook: {len(cells)} cells (was {len(cells) + len(cells_to_remove)})")
print(f"\n📁 Files:")
print(f"  - Original backed up to: {backup_filename}")
print(f"  - Updated notebook: ml_finance_model_main_backup.ipynb")
print(f"\n🎯 Next steps:")
print(f"  1. Open ml_finance_model_main_backup.ipynb in Jupyter")
print(f"  2. Run Phase 9.4 cells to create all_stocks_phase94")
print(f"  3. Run Phase 9.5 (now has complete 8-step implementation)")
print(f"  4. Run Phase 9.5.1 (TDD optimization enhancements)")
print(f"  5. Run Phase 9.6 (model evaluation)")
print(f"  6. Run Phase 9.6.1 (enhanced error analysis)")
