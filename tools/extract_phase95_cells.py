import json
from pathlib import Path

# Load the notebook
notebook_path = Path(
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main_backup.ipynb"
)
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract cells 138-146
print("=" * 80)
print("PHASE 9.5 CELLS CONTENT (138-146)")
print("=" * 80)

for i in range(138, 147):
    if i < len(nb["cells"]):
        cell = nb["cells"][i]
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        print(f"\n{'='*80}")
        print(f"CELL {i} ({cell.get('cell_type', 'unknown')})")
        print(f"{'='*80}")
        print(source_text[:1500])  # First 1500 chars
        if len(source_text) > 1500:
            print(f"\n... (truncated, total length: {len(source_text)} chars)")
