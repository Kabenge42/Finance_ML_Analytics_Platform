"""Analyze the notebook structure to understand the raw cell issue."""

import json

with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print(f"\nCell types distribution:")
cell_types = {}
for c in nb["cells"]:
    ct = c["cell_type"]
    cell_types[ct] = cell_types.get(ct, 0) + 1
for ct, count in cell_types.items():
    print(f"  {ct}: {count}")

print(f"\nFirst 10 cells:")
for i, c in enumerate(nb["cells"][:10]):
    src_preview = "".join(c.get("source", []))[:80].replace("\n", "\\n")
    print(f"  [{i}] {c['cell_type']}: {src_preview}...")

# Check if first cell is raw and contains notebook JSON
first_cell = nb["cells"][0]
if first_cell["cell_type"] == "raw":
    src = "".join(first_cell.get("source", []))
    print(f"\nFirst cell is RAW with {len(src)} characters")
    print(f"First 200 chars: {src[:200]}")

    # Check if it looks like notebook JSON
    if src.strip().startswith("{") and '"cells"' in src[:100]:
        print("\n*** WARNING: First raw cell appears to contain notebook JSON! ***")
