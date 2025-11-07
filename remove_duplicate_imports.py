"""
Remove duplicate imports from ml_finance_model_main_v10.ipynb.
Keeps only the main import cell (cell 4) and removes all other import statements.
"""
import json
from pathlib import Path
import shutil
from datetime import datetime

def remove_duplicate_imports(notebook_path):
    """Remove duplicate imports from notebook cells after the main import cell."""
    
    # Backup the notebook first
    backup_path = notebook_path.parent / f"{notebook_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    shutil.copy(notebook_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Patterns to remove (imports already in main import cell)
    import_patterns = [
        'from finance_ml import',
        'from finance_ml.',
        'import finance_ml',
        'from pathlib import Path',
        'import logging',
        'import traceback',
        'import numpy as np',
        'import pandas as pd',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'import plotly.express as px',
        'from sklearn.preprocessing import',
        'from dataclasses import dataclass',
        'from typing import',
        'from urllib.parse import',
        'from urllib.request import',
    ]
    
    removed_count = 0
    cells_modified = 0
    
    # Process each cell (skip first 5 cells which contain setup and main imports)
    for idx, cell in enumerate(nb['cells']):
        if idx < 5 or cell['cell_type'] != 'code':
            continue
        
        original_source = cell['source'][:]
        new_source = []
        lines_removed = 0
        
        for line in cell['source']:
            # Check if line matches any import pattern
            line_stripped = line.strip()
            is_import = False
            
            if line_stripped.startswith(('import ', 'from ')):
                # Check against patterns
                for pattern in import_patterns:
                    if pattern in line_stripped:
                        is_import = True
                        removed_count += 1
                        lines_removed += 1
                        break
            
            # Keep line if it's not a duplicate import
            if not is_import:
                new_source.append(line)
        
        # Update cell if we removed any lines
        if lines_removed > 0:
            cell['source'] = new_source
            cells_modified += 1
            if lines_removed > 1:
                print(f"  Cell {idx}: removed {lines_removed} import lines")
    
    # Save modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Removed {removed_count} duplicate import lines from {cells_modified} cells")
    print(f"✓ Updated notebook: {notebook_path}")
    print(f"✓ Backup saved: {backup_path}")
    
    return removed_count, cells_modified

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main_v10.ipynb')
    if notebook_path.exists():
        print("=" * 80)
        print("REMOVING DUPLICATE IMPORTS FROM NOTEBOOK")
        print("=" * 80)
        removed, modified = remove_duplicate_imports(notebook_path)
        print("\n" + "=" * 80)
        print(f"SUMMARY: Removed {removed} imports from {modified} cells")
        print("=" * 80)
    else:
        print(f"Error: Notebook not found: {notebook_path}")
