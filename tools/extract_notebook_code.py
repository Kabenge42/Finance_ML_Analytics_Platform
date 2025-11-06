import json
import sys

# Read the notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook['cells'])}")

# Find cells containing "train_and_compare_models"
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'train_and_compare_models' in source and 'def train_and_compare_models' in source:
            # Write to file to avoid encoding issues
            with open('cell_140_code.txt', 'w', encoding='utf-8') as out:
                out.write(f"=== Cell {i}: train_and_compare_models function ===\n\n")
                out.write(source)
            print(f"Found function in cell {i}, saved to cell_140_code.txt")
            break
