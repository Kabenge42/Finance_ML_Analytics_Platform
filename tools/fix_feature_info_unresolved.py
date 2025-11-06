"""
Fix unresolved reference to 'feature_info' in ml_finance_model_main_v10.ipynb

The issue: prepare_regression_data returns feature_info (a dict), but the notebook
unpacks it as 'feature_cols'. Later code tries to access feature_info['numeric_features']
which causes an unresolved reference error.

Solution: Replace 'feature_cols' with 'feature_info' in the unpacking statement.
"""

import json
import sys
from pathlib import Path

def fix_notebook_feature_info(notebook_path: str) -> None:
    """Fix the feature_info unpacking issue in the notebook."""
    
    path = Path(notebook_path)
    if not path.exists():
        print(f"Error: Notebook not found at {path}")
        sys.exit(1)
    
    print(f"Reading notebook: {path}")
    
    # Read the notebook
    with open(path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    # Iterate through cells
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        if not source:
            continue
        
        # Convert to string for easier processing
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Check if this cell contains the problematic pattern
        if 'X_train, X_test, y_train, y_test, feature_cols = prepare_regression_data(' in source_str:
            print(f"\nFound problematic unpacking in cell {cell_idx}")
            print(f"Original line: X_train, X_test, y_train, y_test, feature_cols = prepare_regression_data(...)")
            
            # Replace feature_cols with feature_info
            new_source_str = source_str.replace(
                'X_train, X_test, y_train, y_test, feature_cols = prepare_regression_data(',
                'X_train, X_test, y_train, y_test, feature_info = prepare_regression_data('
            )
            
            # Convert back to list format
            cell['source'] = new_source_str.split('\n')
            # Add newlines back (except last line)
            cell['source'] = [line + '\n' if idx < len(cell['source']) - 1 else line 
                             for idx, line in enumerate(cell['source'])]
            
            changes_made += 1
            print(f"Fixed: X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(...)")
    
    if changes_made == 0:
        print("\n⚠ No changes needed - pattern not found or already fixed")
        return
    
    # Create backup
    backup_path = path.with_suffix(path.suffix + '.backup_feature_info_fix')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    # Write fixed notebook
    print(f"Writing fixed notebook: {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n✓ Successfully fixed {changes_made} occurrence(s)")
    print(f"✓ Backup saved to: {backup_path}")
    print("\nThe fix changes:")
    print("  FROM: X_train, X_test, y_train, y_test, feature_cols = prepare_regression_data(...)")
    print("  TO:   X_train, X_test, y_train, y_test, feature_info = prepare_regression_data(...)")
    print("\nThis resolves the 'Unresolved reference: feature_info' error.")

if __name__ == '__main__':
    notebook_path = 'ml_finance_model_main_v10.ipynb'
    fix_notebook_feature_info(notebook_path)
