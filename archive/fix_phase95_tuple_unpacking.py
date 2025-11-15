#!/usr/bin/env python3
"""
Fix Phase 9.5 tuple unpacking errors in ml_finance_model_main_v10.ipynb

This script fixes three tuple unpacking mismatches between the notebook code
and the finance_ml.advanced_models package functions:
1. train_stacking_regressor returns (model, results_dict) not dict
2. train_quantile_regressor returns (models_list, results_dict) not dict
3. train_sector_specific_models returns (models_dict, results_dict) not dict
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def backup_notebook(notebook_path: Path) -> Path:
    """Create timestamped backup of notebook."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.parent / f"{notebook_path.stem}_backup_{timestamp}{notebook_path.suffix}"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def fix_train_ensemble_models(code_lines: list) -> list:
    """Fix tuple unpacking in train_ensemble_models function."""
    fixed_lines = []
    i = 0
    
    while i < len(code_lines):
        line = code_lines[i]
        
        # Fix 1: train_stacking_regressor tuple unpacking
        if 'stacking_result = train_stacking_regressor(' in line:
            # Find the closing parenthesis (may span multiple lines)
            call_lines = [line]
            j = i + 1
            while j < len(code_lines) and ')' not in ''.join(call_lines):
                call_lines.append(code_lines[j])
                j += 1
            
            # Replace the call with tuple unpacking
            full_call = ''.join(call_lines)
            fixed_call = full_call.replace(
                'stacking_result = train_stacking_regressor(',
                'stacking_model, stacking_results = train_stacking_regressor('
            )
            fixed_lines.extend(fixed_call.split('\n'))
            i = j
            continue
        
        # Fix 1b: Update None check for stacking
        if 'if stacking_result is None:' in line:
            fixed_lines.append(line.replace('stacking_result', 'stacking_model'))
            i += 1
            continue
        
        # Fix 1c: Remove dict access for stacking_model
        if 'stacking_model = stacking_result[' in line:
            # Skip this line entirely, we already have stacking_model from tuple unpacking
            i += 1
            continue
        
        # Fix 1d: Update stacking_result dict accesses
        if "stacking_result.get('train_score'" in line:
            fixed_lines.append(line.replace("stacking_result.get('train_score'", "stacking_results.get('train_score'"))
            i += 1
            continue
        
        if "stacking_result.get('cv_score'" in line:
            fixed_lines.append(line.replace("stacking_result.get('cv_score'", "stacking_results.get('cv_score'"))
            i += 1
            continue
        
        if "'train_score': stacking_result.get('train_score'" in line:
            fixed_lines.append(line.replace("stacking_result.get('train_score'", "stacking_results.get('train_score'"))
            i += 1
            continue
        
        if "'cv_score': stacking_result.get('cv_score'" in line:
            fixed_lines.append(line.replace("stacking_result.get('cv_score'", "stacking_results.get('cv_score'"))
            i += 1
            continue
        
        # Add line if no match
        fixed_lines.append(line)
        i += 1
    
    return fixed_lines


def fix_train_quantile_models(code_lines: list) -> list:
    """Fix tuple unpacking in train_quantile_models function."""
    fixed_lines = []
    i = 0
    
    while i < len(code_lines):
        line = code_lines[i]
        
        # Fix 2: train_quantile_regressor tuple unpacking
        if 'quantile_result = train_quantile_regressor(' in line:
            # Find the closing parenthesis
            call_lines = [line]
            j = i + 1
            while j < len(code_lines) and ')' not in ''.join(call_lines):
                call_lines.append(code_lines[j])
                j += 1
            
            # Replace the call with tuple unpacking
            full_call = ''.join(call_lines)
            fixed_call = full_call.replace(
                'quantile_result = train_quantile_regressor(',
                'quantile_models, quantile_results = train_quantile_regressor('
            )
            fixed_lines.extend(fixed_call.split('\n'))
            i = j
            continue
        
        # Fix 2b: Update None check for quantile
        if 'if quantile_result is None:' in line:
            fixed_lines.append(line.replace('quantile_result', 'quantile_models'))
            i += 1
            continue
        
        # Fix 2c: Remove dict access for quantile_models
        if 'quantile_models = quantile_result[' in line:
            # Skip this line, we already have quantile_models from tuple unpacking
            i += 1
            continue
        
        # Add line if no match
        fixed_lines.append(line)
        i += 1
    
    return fixed_lines


def fix_train_sector_models(code_lines: list) -> list:
    """Fix tuple unpacking in train_sector_models function."""
    fixed_lines = []
    i = 0

    while i < len(code_lines):
        line = code_lines[i]

        # Fix 3: train_sector_specific_models tuple unpacking
        if 'sector_models_result = train_sector_specific_models(' in line:
            # Find the closing parenthesis
            call_lines = [line]
            j = i + 1
            while j < len(code_lines) and ')' not in ''.join(call_lines):
                call_lines.append(code_lines[j])
                j += 1

            # Replace the call with tuple unpacking
            full_call = ''.join(call_lines)
            fixed_call = full_call.replace(
                'sector_models_result = train_sector_specific_models(',
                'sector_models, sector_results = train_sector_specific_models('
            )
            fixed_lines.extend(fixed_call.split('\n'))
            i = j
            continue

        # Fix 3b: Update None check for sector regression
        if 'if sector_models_result is None:' in line:
            fixed_lines.append(line.replace('sector_models_result', 'sector_models'))
            i += 1
            continue

        # Fix 3c: Remove dict access for sector_models
        if 'sector_models = sector_models_result[' in line:
            # Skip this line, we already have sector_models from tuple unpacking
            i += 1
            continue

        # Fix 3d: Update sector_metrics access
        if 'sector_metrics = sector_models_result[' in line:
            fixed_lines.append(line.replace(
                "sector_metrics = sector_models_result['metrics']",
                "sector_metrics = sector_results['sector_metrics']"
            ))
            i += 1
            continue

        # Add line if no match
        fixed_lines.append(line)
        i += 1

    return fixed_lines


def fix_cell_140(notebook_path: Path) -> bool:
    """Fix Cell 140 tuple unpacking errors."""
    print(f"\n[INFO] Reading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find Cell 140 (index 139 in 0-based indexing)
    if len(notebook['cells']) <= 140:
        print(f"[ERROR] Notebook has only {len(notebook['cells'])} cells")
        return False
    
    cell_140 = notebook['cells'][140]
    
    if cell_140['cell_type'] != 'code':
        print(f"[ERROR] Cell 140 is not a code cell")
        return False
    
    print(f"[OK] Found Cell 140 (code cell with {len(cell_140['source'])} lines)")
    
    # Get source code lines
    original_lines = cell_140['source']
    
    # Apply fixes in sequence
    print("\n[INFO] Applying fixes...")
    print("  Fix 1: train_ensemble_models tuple unpacking")
    fixed_lines = fix_train_ensemble_models(original_lines)
    
    print("  Fix 2: train_quantile_models tuple unpacking")
    fixed_lines = fix_train_quantile_models(fixed_lines)
    
    print("  Fix 3: train_sector_models tuple unpacking")
    fixed_lines = fix_train_sector_models(fixed_lines)
    
    # Update cell source
    cell_140['source'] = fixed_lines
    
    # Save updated notebook
    print(f"\n[INFO] Saving updated notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"[OK] Notebook updated successfully")
    
    # Report changes
    lines_added = len(fixed_lines) - len(original_lines)
    print(f"\n[INFO] Changes summary:")
    print(f"  Original lines: {len(original_lines)}")
    print(f"  Updated lines: {len(fixed_lines)}")
    print(f"  Net change: {lines_added:+d} lines")
    
    return True


def main():
    """Main execution."""
    print("=" * 80)
    print("PHASE 9.5 TUPLE UNPACKING FIX")
    print("=" * 80)
    
    notebook_path = Path("ml_finance_model_main_v10.ipynb")
    
    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        return 1
    
    # Create backup
    backup_path = backup_notebook(notebook_path)
    
    # Apply fixes
    success = fix_cell_140(notebook_path)
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] PHASE 9.5 FIX COMPLETE")
        print("=" * 80)
        print(f"\n[INFO] Files:")
        print(f"  Original backup: {backup_path}")
        print(f"  Updated notebook: {notebook_path}")
        print("\n[INFO] Next steps:")
        print("  1. Restart Jupyter kernel")
        print("  2. Run cells sequentially from Phase 9.1")
        print("  3. Verify Phase 9.5 completes without errors")
        return 0
    else:
        print("\n[ERROR] Fix failed. Notebook unchanged.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
