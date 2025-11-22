"""
Find and update the benchmarking section in ml_finance_model_main.ipynb
"""

import json
from pathlib import Path

# Load notebook
nb_path = Path("ml_finance_model_main.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cells with phase93_category_cells_backup reference
print("Searching for benchmarking cell...")
found_cells = []
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "phase93_category_cells_backup" in source:
        found_cells.append(i)
        print(f"\nFound at cell {i}:")
        print(f"  Type: {cell.get('cell_type')}")
        print(f"  Preview: {source[:300]}...")

if found_cells:
    print(f"\n✓ Found {len(found_cells)} cell(s) with phase93_category_cells_backup reference")
    print(f"  Cell indices: {found_cells}")

    # Load corrected benchmarking code
    corrected_code_path = Path("corrected_benchmarking_cell.py")
    with open(corrected_code_path, "r", encoding="utf-8") as f:
        corrected_code = f.read()

    # Remove the docstring from corrected code
    lines = corrected_code.split("\n")
    # Skip first 4 lines (docstring)
    corrected_code_clean = "\n".join(lines[4:])

    print(f"\n📝 Replacement code prepared ({len(corrected_code_clean)} chars)")
    print(f"\nWould replace cell {found_cells[0]} with corrected benchmarking code")
    print(f"This will:")
    print(f"  - Use actual metrics from preprocessed_stocks_metadata.json")
    print(f"  - Report accurate coverage (7 valuation metrics available)")
    print(f"  - Skip empty categories in visualizations")
    print(f"  - Add note about feature engineering requirement")

else:
    print("\n⚠️  No cells found with phase93_category_cells_backup reference")
    print("    Searching for alternative patterns...")

    # Search for Phase 9.3 related cells
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if "Phase 9.3" in source and "Benchmark" in source:
            print(f"\n  Found Phase 9.3 + Benchmark at cell {i}")
            print(f"    Preview: {source[:200]}...")
