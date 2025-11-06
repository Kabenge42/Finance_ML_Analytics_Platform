#!/usr/bin/env python3
"""
Fix the NonNegativeRegressionWrapper import pattern in the notebook
to resolve PyCharm inspection warning.
"""

import json
import sys
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def fix_notebook():
    """Fix the import pattern in ml_finance_model_main_v10.ipynb"""
    
    notebook_path = Path("ml_finance_model_main_v10.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        return False
    
    # Read the notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return False
    
    # The problematic pattern
    old_pattern = '''        try:
            from finance_ml.advanced_models import NonNegativeRegressionWrapper
            models = {name: NonNegativeRegressionWrapper(model) for name, model in models.items()}
        except ImportError:
            print("⚠ Warning: NonNegativeRegressionWrapper not available, predictions may be negative")'''
    
    # The fixed pattern - separate the import from the usage
    new_pattern = '''        try:
            from finance_ml.advanced_models import NonNegativeRegressionWrapper
        except ImportError:
            NonNegativeRegressionWrapper = None
            print("⚠ Warning: NonNegativeRegressionWrapper not available, predictions may be negative")
        
        if NonNegativeRegressionWrapper is not None:
            models = {name: NonNegativeRegressionWrapper(model) for name, model in models.items()}'''
    
    modified = False
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # Join source lines if it's a list
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source
            
            # Check if this cell contains the pattern
            if 'NonNegativeRegressionWrapper' in source_text and 'except ImportError:' in source_text:
                # Replace the pattern
                new_source_text = source_text.replace(old_pattern, new_pattern)
                
                if new_source_text != source_text:
                    # Convert back to list format if needed
                    if isinstance(source, list):
                        cell['source'] = new_source_text.splitlines(keepends=True)
                    else:
                        cell['source'] = new_source_text
                    
                    modified = True
                    print("[OK] Fixed NonNegativeRegressionWrapper import pattern")
    
    if not modified:
        print("[WARNING] Pattern not found or already fixed")
        return False
    
    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup_nonneg_fix')
    try:
        import shutil
        shutil.copy2(notebook_path, backup_path)
        print(f"[OK] Created backup: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Write the modified notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[OK] Updated {notebook_path}")
        return True
    except Exception as e:
        print(f"Error writing notebook: {e}")
        return False

if __name__ == "__main__":
    success = fix_notebook()
    sys.exit(0 if success else 1)
