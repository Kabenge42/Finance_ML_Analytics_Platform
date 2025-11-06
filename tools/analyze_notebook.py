"""
Quick script to analyze notebook structure
"""
import json
from pathlib import Path

notebook_path = Path("ml_finance_model_main_backup.ipynb")

if not notebook_path.exists():
    print(f"ERROR: {notebook_path} not found")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f"Total cells: {len(cells)}")
print(f"\nPhase 9 sections found:")

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown' and cell.get('source'):
        source = ''.join(cell['source'])
        if 'Phase 9' in source:
            # Get first line only
            first_line = source.split('\n')[0]
            print(f"  Cell {i}: {first_line[:100]}")

print(f"\nCells around index 111 (check for duplicates):")
for i in range(max(0, 109), min(len(cells), 115)):
    cell = cells[i]
    source = ''.join(cell.get('source', ['empty']))[:80]
    print(f"  Cell {i} ({cell['cell_type']}): {source}")
