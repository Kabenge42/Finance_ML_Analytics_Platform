"""Analyze Phase 9.5 implementation in notebook."""
import json

# Load notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f"Total cells in notebook: {len(cells)}")
print()

# Find Phase 9.5 cells
phase95_indices = []
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if 'Phase 9.5' in source:
        phase95_indices.append(i)
        cell_type = cell.get('cell_type', 'unknown')
        preview = source[:100].replace('\n', ' ')
        print(f"Cell {i} ({cell_type}): {preview}...")

print(f"\nFound {len(phase95_indices)} cells mentioning Phase 9.5")
print(f"Phase 9.5 cell indices: {phase95_indices[:10]}")

# Save Phase 9.5 cells to files for detailed analysis
print("\n" + "="*80)
print("SAVING PHASE 9.5 CELLS TO FILES")
print("="*80)

saved_files = []
for idx in [140, 142]:  # Main Phase 9.5 and 9.5.1 cells
    if idx in phase95_indices:
        cell = cells[idx]
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            filename = f'phase95_cell{idx}.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(source)
            saved_files.append(filename)
            print(f"[OK] Saved Cell {idx} to {filename} ({len(source)} chars)")

print(f"\n[OK] Total saved: {len(saved_files)} files")
print("Files saved for analysis of current implementation")
