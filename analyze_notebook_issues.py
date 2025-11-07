"""
Analyze ml_finance_model_main_v10.ipynb for import issues and dead code.
"""
import json
from collections import defaultdict
from pathlib import Path

def analyze_notebook(notebook_path):
    """Analyze notebook for issues."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Track imports by cell
    imports_by_cell = []
    all_imports = defaultdict(list)
    
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            lines = cell['source']
            
            cell_imports = []
            for line_num, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped.startswith(('import ', 'from ')):
                    cell_imports.append((line_num, line_stripped))
                    
                    # Extract module name
                    if line_stripped.startswith('from '):
                        module = line_stripped.split()[1]
                    else:
                        module = line_stripped.split()[1].split('.')[0]
                    
                    all_imports[line_stripped].append(idx)
            
            if cell_imports:
                imports_by_cell.append((idx, cell_imports))
    
    # Find duplicates
    print("=" * 80)
    print("DUPLICATE IMPORTS ANALYSIS")
    print("=" * 80)
    
    duplicates = {k: v for k, v in all_imports.items() if len(v) > 1}
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate import statements:\n")
        for imp, cells in sorted(duplicates.items()):
            print(f"  Cells {cells}: {imp[:80]}")
    else:
        print("\n✓ No duplicate imports found")
    
    # Show all imports from finance_ml
    print("\n" + "=" * 80)
    print("FINANCE_ML IMPORTS BY MODULE")
    print("=" * 80)
    
    fm_imports = defaultdict(set)
    for imp in all_imports.keys():
        if 'finance_ml' in imp and 'from finance_ml' in imp:
            if ' import ' in imp:
                parts = imp.split(' import ')
                module = parts[0].replace('from ', '').strip()
                items = parts[1].strip()
                fm_imports[module].add(items)
    
    for module in sorted(fm_imports.keys()):
        print(f"\n{module}:")
        for items in sorted(fm_imports[module]):
            print(f"  - {items[:70]}")
    
    # Check for imports in middle of notebook
    print("\n" + "=" * 80)
    print("IMPORTS AFTER CELL 20 (should be consolidated at top)")
    print("=" * 80)
    
    late_imports = [(idx, imps) for idx, imps in imports_by_cell if idx > 20]
    if late_imports:
        print(f"\nFound {len(late_imports)} cells with imports after cell 20:\n")
        for idx, imps in late_imports:
            print(f"\n  Cell {idx}:")
            for line_num, imp in imps[:3]:  # Show first 3
                print(f"    {imp[:70]}")
    else:
        print("\n✓ All imports are in the first 20 cells")
    
    return nb, imports_by_cell, duplicates

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main_v10.ipynb')
    if notebook_path.exists():
        nb, imports, dups = analyze_notebook(notebook_path)
        print(f"\n\nTotal cells: {len(nb['cells'])}")
        print(f"Code cells with imports: {len(imports)}")
        print(f"Duplicate import statements: {len(dups)}")
    else:
        print(f"Notebook not found: {notebook_path}")
