"""Verify new cells in notebook."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]

print("=" * 80)
print(f"NOTEBOOK VERIFICATION - Total cells: {len(cells)}")
print("=" * 80)

# Check Phase 9.5 cells (around 78-100)
print("\nPhase 9.5 Section (cells 78-100):")
print("-" * 60)
for i in range(78, min(100, len(cells))):
    source = "".join(cells[i]["source"])
    first_line = source.split("\n")[0][:70] if source else "(empty)"
    # Clean unicode
    first_line = first_line.replace("\u274c", "[X]").replace("\u2713", "[OK]")
    first_line = first_line.replace("\u26a0", "[!]").replace("\U0001f4ca", "[CHART]")
    first_line = first_line.replace("\U0001f680", "[ROCKET]").replace("\U0001f332", "[TREE]")
    first_line = first_line.replace("\U0001f9e0", "[BRAIN]").replace("\U0001f5f3", "[BALLOT]")
    print(f"Cell {i:3d} ({cells[i]['cell_type'][:4]}): {first_line}")

# Check Phase 9.6 cells (around 106-120)
print("\nPhase 9.6 Section (cells 106-125):")
print("-" * 60)
for i in range(106, min(125, len(cells))):
    source = "".join(cells[i]["source"])
    first_line = source.split("\n")[0][:70] if source else "(empty)"
    first_line = first_line.replace("\u274c", "[X]").replace("\u2713", "[OK]")
    first_line = first_line.replace("\u26a0", "[!]").replace("\U0001f4ca", "[CHART]")
    print(f"Cell {i:3d} ({cells[i]['cell_type'][:4]}): {first_line}")

# Verify new cells exist
print("\n" + "=" * 80)
print("VERIFICATION CHECKS")
print("=" * 80)

checks = [
    ("Linear Models", "6.3.1 Linear Models"),
    ("Tree Models", "6.3.2 Tree-Based"),
    ("Gradient Boosting", "6.3.3 Gradient Boosting"),
    ("Neural Network", "6.3.4 Neural Network"),
    ("Ensemble Summary", "6.3.5 Ensemble Methods"),
    ("Feature Importance Sector", "7.1 Feature Importance"),
    ("Performance by Region", "7.2 Model Performance"),
]

for name, pattern in checks:
    found = False
    for cell in cells:
        source = "".join(cell["source"])
        if pattern in source:
            found = True
            break
    status = "[OK]" if found else "[MISSING]"
    print(f"  {status} {name}")

print("\n[OK] Verification complete")
