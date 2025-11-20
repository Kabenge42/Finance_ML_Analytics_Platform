"""
Fix the final two hardcoded values:
1. Cell 51: 'last_price' → TARGET_COL_FALLBACK
2. Cell 81: max_sector_weight=0.25 → max_sector_weight=MAX_SECTOR_WEIGHT
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

changes = []
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

# Fix Cell 51: 'last_price' in preserve_cols
cell51 = code_cells[51]
source51 = "".join(cell51["source"])
if "preserve_cols = ['market_cap', 'last_price', 'price_target'" in source51:
    source51 = source51.replace(
        "preserve_cols = ['market_cap', 'last_price', 'price_target', 'enterprise_value']",
        "preserve_cols = ['market_cap', TARGET_COL_FALLBACK, 'price_target', 'enterprise_value']",
    )
    cell51["source"] = [line + "\n" for line in source51.split("\n")[:-1]] + [
        source51.split("\n")[-1]
    ]
    changes.append("Cell 51: Replaced 'last_price' with TARGET_COL_FALLBACK")

# Fix Cell 81: max_sector_weight=0.25
cell81 = code_cells[81]
source81 = "".join(cell81["source"])
if "max_sector_weight=0.25" in source81:
    source81 = source81.replace("max_sector_weight=0.25", "max_sector_weight=MAX_SECTOR_WEIGHT")
    cell81["source"] = [line + "\n" for line in source81.split("\n")[:-1]] + [
        source81.split("\n")[-1]
    ]
    changes.append("Cell 81: Replaced max_sector_weight=0.25 with MAX_SECTOR_WEIGHT")

# Save
if changes:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"✓ Fixed final hardcoded values")
    for change in changes:
        print(f"  - {change}")
else:
    print("No changes needed")
