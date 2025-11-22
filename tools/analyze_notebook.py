"""Quick script to analyze notebook structure."""
import json

with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f"Total cells: {len(cells)}\n")

# Find cells with Phase or Section markers
print("Cells with 'Phase' or 'Section' markers:")
print("=" * 80)
for i, cell in enumerate(cells):
    source = cell.get('source', [])
    if isinstance(source, list):
        source = ''.join(source)
    
    if 'Phase' in source or 'Section' in source:
        # Get first line
        first_line = source.split('\n')[0][:100]
        print(f"Cell {i} ({cell.get('cell_type', '?')}): {first_line}")

print("\n" + "=" * 80)
print("\nLast 10 cells:")
for i in range(max(0, len(cells)-10), len(cells)):
    cell = cells[i]
    source = cell.get('source', [])
    if isinstance(source, list):
        source = ''.join(source)
    first_line = source.split('\n')[0][:80] if source else "(empty)"
    print(f"Cell {i} ({cell.get('cell_type', '?')}): {first_line}")
