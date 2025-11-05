"""Quick diagnostic to check notebook structure after edits."""

import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Notebook structure check:")
print(f"Total cells: {len(nb['cells'])}")

# Find cells with finance_ml.eval imports
import_cells = []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "from finance_ml.eval import" in source:
            import_cells.append((i, cell))

print(f"\nFound {len(import_cells)} cells with 'from finance_ml.eval import'")

for idx, (cell_num, cell) in enumerate(import_cells[:2]):  # Show first 2
    source = "".join(cell.get("source", []))
    lines = source.split("\n")
    print(f"\nCell {cell_num} (import cell {idx+1}):")
    print(f"  Type: {cell['cell_type']}")
    print(f"  Lines: {len(lines)}")
    print(f"  First 5 lines:")
    for line in lines[:5]:
        print(f"    {line}")

    # Check for specific functions
    funcs_to_check = [
        "assign_valuation_category",
        "calculate_sector_zscores",
        "simple_eda",
        "calculate_mispricing_score",
    ]
    found_funcs = [f for f in funcs_to_check if f in source]
    print(f"  Contains {len(found_funcs)} of checked functions: {found_funcs}")

print("\n✓ Notebook JSON is valid and readable")
