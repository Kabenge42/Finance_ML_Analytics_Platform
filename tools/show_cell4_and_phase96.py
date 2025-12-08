"""Show cell 4 imports and Phase 9.6 cells."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]


def clean_source(source):
    source = source.replace("\u274c", "[X]").replace("\u2713", "[OK]").replace("\u2714", "[OK]")
    source = source.replace("\u2717", "[X]").replace("\u2705", "[OK]").replace("\u26a0", "[!]")
    source = source.replace("\U0001f4ca", "[CHART]").replace("\U0001f4c8", "[GRAPH]")
    source = source.replace("\U0001f4c2", "[FOLDER]").replace("\U0001f50d", "[SEARCH]")
    source = source.replace("\U0001f4dd", "[NOTE]").replace("\U0001f4a1", "[IDEA]")
    return source


# Show cell 4 - just regression imports section
print("=" * 80)
print("CELL 4 - REGRESSION IMPORTS SECTION")
print("=" * 80)
source = "".join(cells[4]["source"])
# Find regression imports
lines = source.split("\n")
in_regression = False
for line in lines:
    if "regression" in line.lower():
        in_regression = True
    if in_regression:
        print(line)
        if line.strip() == ")" or (
            line.strip()
            and not line.startswith(" ")
            and not line.startswith("from")
            and not line.startswith("#")
        ):
            if "from finance_ml" not in line and "import" not in line:
                in_regression = False

# Show Phase 9.6 cells (96-126)
print("\n" + "=" * 80)
print("PHASE 9.6 CELLS (96-130) - HEADERS ONLY")
print("=" * 80)

for i in range(96, min(131, len(cells))):
    cell = cells[i]
    source = "".join(cell["source"])
    source = clean_source(source)
    first_line = source.split("\n")[0] if source else "(empty)"
    print(f"Cell {i} ({cell['cell_type'][:4]}): {first_line[:80]}")
