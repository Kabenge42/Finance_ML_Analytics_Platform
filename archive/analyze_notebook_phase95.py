import json
from pathlib import Path

# Load the notebook
notebook_path = Path(
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main_backup.ipynb"
)
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")
print("=" * 80)
print("PHASE 9.5 SECTIONS FOUND:")
print("=" * 80)

phase95_cells = []
for i, cell in enumerate(nb["cells"]):
    source = cell.get("source", [])
    if isinstance(source, list):
        source_text = "".join(source)
    else:
        source_text = source

    if "Phase 9.5" in source_text or "PHASE 9.5" in source_text:
        # Extract first line as title
        lines = source_text.split("\n")
        title = lines[0][:100] if lines else ""
        print(f"\nCell {i}: {cell.get('cell_type', 'unknown')}")
        print(f"  Title: {title}")
        phase95_cells.append((i, title, source_text[:300]))

print(f"\n\nTotal Phase 9.5 cells found: {len(phase95_cells)}")

# Check for specific subsections
print("\n" + "=" * 80)
print("CHECKING FOR SPECIFIC SUBSECTIONS:")
print("=" * 80)

subsections = [
    "9.5.1",
    "9.5.2",
    "9.5.3",
    "9.5.4",
    "9.5.5",
    "9.5.6",
    "classification meta-features",
    "interaction features",
    "multiple regression regression",
    "sector-specific",
    "quantile regression",
    "model persistence",
]

for subsection in subsections:
    found = False
    for i, cell in enumerate(nb["cells"]):
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        if subsection.lower() in source_text.lower() and "phase 9.5" in source_text.lower():
            found = True
            print(f"✓ Found '{subsection}' in cell {i}")
            break

    if not found:
        print(f"✗ MISSING: '{subsection}'")
