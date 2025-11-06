#!/usr/bin/env python3
"""Script to update code in ml_finance_model_main_v10.ipynb"""

import json
import sys

def find_and_replace_code(notebook_path):
    """Find and replace the inefficient loop code with optimized version"""
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Code to find (old implementation)
        old_code = [
            "    all_stocks_imputed_reset = all_stocks_imputed.reset_index(drop=True)\n",
            "\n",
            "    # Update only the numeric feature columns with imputed data\n",
            "    for col in feature_info['numeric_features']:\n",
            "        if col in X_combined.columns:\n",
            "            all_stocks_imputed_reset[col] = X_combined[col]\n"
        ]
        
        # New optimized code (Option 3)
        new_code = [
            "    # Reset index with proper validation\n",
            "    all_stocks_imputed_reset = all_stocks_imputed.reset_index(drop=True)\n",
            "\n",
            "    # Validate shape compatibility\n",
            "    if len(all_stocks_imputed_reset) != len(X_combined):\n",
            "        raise ValueError(\n",
            "            f\"Shape mismatch: all_stocks_imputed has {len(all_stocks_imputed_reset)} rows \"\n",
            "            f\"but X_combined has {len(X_combined)} rows\"\n",
            "        )\n",
            "\n",
            "    # Filter to valid numeric features\n",
            "    valid_numeric_features = [\n",
            "        col for col in feature_info['numeric_features']\n",
            "        if col in X_combined.columns and col in all_stocks_imputed_reset.columns\n",
            "    ]\n",
            "\n",
            "    if not valid_numeric_features:\n",
            "        raise ValueError(\"No overlapping numeric features found between DataFrames\")\n",
            "\n",
            "    # Efficient bulk assignment using .values to avoid index alignment issues\n",
            "    X_combined[valid_numeric_features] = all_stocks_imputed_reset[valid_numeric_features].values\n"
        ]
        
        # Search through all cells
        modified = False
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                
                # Check if this cell contains the old code
                source_str = ''.join(source)
                old_code_str = ''.join(old_code)
                
                if old_code_str in source_str:
                    print(f"Found matching code in cell")
                    
                    # Find the exact position in the source
                    for i in range(len(source) - len(old_code) + 1):
                        if source[i:i+len(old_code)] == old_code:
                            # Replace the old code with new code
                            cell['source'] = source[:i] + new_code + source[i+len(old_code):]
                            modified = True
                            print(f"Replaced code at position {i}")
                            break
        
        if not modified:
            print("Could not find exact matching code. Searching for approximate match...")
            
            # Try a more flexible search
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = cell['source']
                    source_str = ''.join(source)
                    
                    if 'all_stocks_imputed_reset = all_stocks_imputed.reset_index(drop=True)' in source_str and \
                       "for col in feature_info['numeric_features']:" in source_str and \
                       'all_stocks_imputed_reset[col] = X_combined[col]' in source_str:
                        
                        print("Found approximate match - replacing entire relevant section")
                        
                        # Find indices
                        start_idx = None
                        end_idx = None
                        
                        for i, line in enumerate(source):
                            if 'all_stocks_imputed_reset = all_stocks_imputed.reset_index(drop=True)' in line:
                                start_idx = i
                            if start_idx is not None and 'all_stocks_imputed_reset[col] = X_combined[col]' in line:
                                end_idx = i + 1
                                break
                        
                        if start_idx is not None and end_idx is not None:
                            # Calculate leading whitespace from first line
                            leading_space = len(source[start_idx]) - len(source[start_idx].lstrip())
                            
                            # Replace the section
                            cell['source'] = source[:start_idx] + new_code + source[end_idx:]
                            modified = True
                            print(f"Replaced code from line {start_idx} to {end_idx}")
                            break
        
        if modified:
            # Create backup
            backup_path = notebook_path + '.backup_code_optimization'
            with open(backup_path, 'w', encoding='utf-8') as f:
                with open(notebook_path, 'r', encoding='utf-8') as orig:
                    f.write(orig.read())
            print(f"Created backup: {backup_path}")
            
            # Write updated notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            
            print(f"Successfully updated {notebook_path}")
            return True
        else:
            print("ERROR: Could not find the code to replace")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    notebook_path = 'ml_finance_model_main_v10.ipynb'
    success = find_and_replace_code(notebook_path)
    sys.exit(0 if success else 1)
