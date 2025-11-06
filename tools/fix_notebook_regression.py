import json
import re
import sys

# Set stdout encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Read the notebook
with open('ml_finance_model_main_backup.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix the problematic cell
fixed = False
for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        
        # Check if this cell contains the problematic code
        if 'X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_regression_data(' in source:
            print("Found problematic cell!")
            print("Original code length:", len(source), "characters")
            print("\n" + "="*60 + "\n")
            
            # Fix 1: Add feature_info to unpacking (5 values instead of 4)
            source = re.sub(
                r'X_train_reg,\s*X_test_reg,\s*y_train_reg,\s*y_test_reg\s*=\s*prepare_regression_data\(',
                'X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_info_reg = prepare_regression_data(',
                source
            )
            
            # Fix 2: Remove default parameter arguments
            # Remove test_size=0.2 (default is 0.2)
            source = re.sub(r',\s*test_size=0\.2', '', source)
            
            # Remove random_state=42 (default is 42)
            source = re.sub(r',\s*random_state=42', '', source)
            
            # Update the cell source
            cell['source'] = source.split('\n')
            if not cell['source'][-1].endswith('\n'):
                cell['source'][-1] += '\n'
            
            print("Fixed code:")
            print(source)
            fixed = True
            break

if fixed:
    # Save the fixed notebook
    with open('ml_finance_model_main_backup.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("\n" + "="*60)
    print("✓ Notebook fixed successfully!")
else:
    print("⚠ Could not find the problematic code in the notebook")
