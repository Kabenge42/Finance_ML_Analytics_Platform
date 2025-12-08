"""Show specific cells from notebook."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]

# Show specific cells: 72 (imports), 79 (model comparison), 96-100 (Phase 9.6)
target_cells = [72, 74, 77, 79, 96, 97, 98, 99, 100, 124, 125, 126]

for i in target_cells:
    if i < len(cells):
        cell = cells[i]
        source = "".join(cell["source"])
        source = source.replace("\u274c", "[X]").replace("\u2713", "[OK]").replace("\u2714", "[OK]")
        source = source.replace("\u2717", "[X]").replace("\u2705", "[OK]").replace("\u26a0", "[!]")
        source = source.replace("\U0001f4ca", "[CHART]").replace("\U0001f4c8", "[GRAPH]")
        source = source.replace("\U0001f4c2", "[FOLDER]").replace("\U0001f50d", "[SEARCH]")
        print(f"\n{'='*80}")
        print(f"CELL {i} ({cell['cell_type']})")
        print(f"{'='*80}")
        print(source[:1500])
        if len(source) > 1500:
            print(f"\n... ({len(source)} total chars)")
