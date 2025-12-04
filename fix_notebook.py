"""Extract embedded notebook from raw cell and save as proper notebook."""

import json
import shutil
from datetime import datetime

# Backup original
backup_name = f'ml_finance_model_main2_0.ipynb.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy("ml_finance_model_main2_0.ipynb", backup_name)
print(f"✓ Created backup: {backup_name}")

# Load current notebook
with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Get the embedded notebook from first raw cell
first_cell = nb["cells"][0]
embedded_json = "".join(first_cell.get("source", []))
embedded_nb = json.loads(embedded_json)

print(f"\nExtracting embedded notebook:")
print(f"  Original cells: {len(nb['cells'])} (including 1 raw cell)")
print(f"  Embedded cells: {len(embedded_nb['cells'])} (all proper cells)")

# Verify embedded notebook structure
assert "cells" in embedded_nb, "Missing cells in embedded notebook"
assert "nbformat" in embedded_nb, "Missing nbformat"
assert embedded_nb["nbformat"] == 4, "Invalid nbformat"

# Count cell types in embedded
cell_types = {}
for c in embedded_nb["cells"]:
    ct = c["cell_type"]
    cell_types[ct] = cell_types.get(ct, 0) + 1
print(f"  Cell types: {cell_types}")

# Save embedded notebook as the new file
with open("ml_finance_model_main2_0.ipynb", "w", encoding="utf-8") as f:
    json.dump(embedded_nb, f, indent=1)

print(f"\n✓ Saved fixed notebook: ml_finance_model_main2_0.ipynb")
print(f"  Total cells: {len(embedded_nb['cells'])}")

# Verify the saved file
with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    verify_nb = json.load(f)
print(f"  Verified: {len(verify_nb['cells'])} cells loaded successfully")

# Check for raw cells
raw_cells = sum(1 for c in verify_nb["cells"] if c["cell_type"] == "raw")
print(f"  Raw cells remaining: {raw_cells}")
if raw_cells == 0:
    print("\n✓ SUCCESS: Notebook converted to proper Jupyter format (no raw cells)")
