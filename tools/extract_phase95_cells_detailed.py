import json
import sys

with open("ml_finance_model_main_backup.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

cells = data.get("cells", [])

# Extract cells 139-155
for i in range(139, min(156, len(cells))):
    cell = cells[i]
    cell_type = cell.get("cell_type", "unknown")
    source = "".join(cell.get("source", []))

    print(f"\n{'='*80}")
    print(f"Cell {i} ({cell_type})")
    print(f"{'='*80}")
    print(source[:1000])  # First 1000 chars
    if len(source) > 1000:
        print(f"\n... (truncated, total length: {len(source)} chars)")
    print()
