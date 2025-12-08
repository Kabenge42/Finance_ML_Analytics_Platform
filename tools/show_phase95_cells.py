"""Show Phase 9.5 cells from notebook."""

import json
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]

# Show cells 71-95 (Phase 9.5)
print("=" * 80)
print("PHASE 9.5 CELLS (71-95)")
print("=" * 80)

for i in range(71, min(96, len(cells))):
    cell = cells[i]
    source = "".join(cell["source"])
    # Replace problematic unicode chars
    source = source.replace("\u274c", "[X]").replace("\u2713", "[OK]").replace("\u2714", "[OK]")
    print(f"\n{'='*40}")
    print(f"CELL {i} ({cell['cell_type']})")
    print(f"{'='*40}")
    # Show first 400 chars
    print(source[:400])
    if len(source) > 400:
        print(f"... ({len(source)} total chars)")
