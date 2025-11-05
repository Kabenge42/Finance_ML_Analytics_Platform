"""Verify Section 9.1.8 integration in notebook"""

import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}\n")

# Check cells around insertion point
for i in range(39, 43):
    cell = cells[i]
    cell_type = cell["cell_type"]
    source = cell["source"]

    # Get preview
    if isinstance(source, list):
        preview = "".join(source)[:120]
    else:
        preview = source[:120]

    print(f"Cell {i} ({cell_type}):")
    print(f"  {preview}...")
    print()

# Verify Section 9.1.8 presence
section_918_found = False
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "9.1.8" in source and "Enhanced 4-Step Imputation" in source:
            section_918_found = True
            print(f"✓ Found Section 9.1.8 at cell index {i}")
            break

if not section_918_found:
    print("✗ Section 9.1.8 NOT found in notebook")
else:
    print("✓ Notebook integration verified successfully!")
