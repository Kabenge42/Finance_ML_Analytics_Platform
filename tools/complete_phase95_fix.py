"""
Complete Phase 9.5 Fix - Addresses BOTH Issues:
1. KeyError: 'Model' - DataFrame conversion fix
2. Missing checkpoint("regression_complete") at end of Phase 9.5

Based on: docs/summaries/PHASE95_FIXES_IMPLEMENTATION_SUMMARY.md
"""
import json
import shutil
from datetime import datetime

print("=" * 80)
print("COMPLETE PHASE 9.5 FIX")
print("=" * 80)

# Create backup
backup_name = f'ml_finance_model_main_v9_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
shutil.copy('ml_finance_model_main_v10.ipynb', backup_name)
print(f"\n[1] Created backup: {backup_name}")

# Read notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

total_cells = len(notebook['cells'])
print(f"[2] Loaded notebook: {total_cells} cells")

changes_made = []

# ============================================================================
# FIX 1: DataFrame Conversion in train_and_compare_models
# ============================================================================
print("\n[3] Applying Fix 1: DataFrame conversion...")
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_and_compare_models' in source:
            print(f"    Found train_and_compare_models in cell {i}")
            
            # Check if already fixed
            if 'pd.DataFrame.from_dict(comparison_results, orient=\'index\')' in source:
                print("    [SKIP] Already contains correct DataFrame conversion")
            else:
                # Apply the fix
                old_pattern = 'pd.DataFrame([comparison_results])'
                if old_pattern in source:
                    # Replace the entire problematic section
                    old_code = """    # Convert dict to DataFrame if needed (compare_regressors returns dict)
    if isinstance(comparison_results, dict):
        comparison_results = pd.DataFrame([comparison_results])

    print("\\n📈 Model Comparison Results:")
    print(comparison_results.to_string(index=False))

    best_model = comparison_results.iloc[0]['Model']
    best_mae = comparison_results.iloc[0]['MAE']
    best_r2 = comparison_results.iloc[0]['R2']"""
                    
                    new_code = """    # Convert dict to DataFrame (compare_regressors returns dict)
    if isinstance(comparison_results, dict):
        # Convert from dict format to DataFrame with Model as a column
        comparison_results = pd.DataFrame.from_dict(comparison_results, orient='index')
        comparison_results = comparison_results.reset_index().rename(columns={'index': 'Model'})
        # Sort by R2 score descending
        comparison_results = comparison_results.sort_values('r2', ascending=False)
        # Rename columns to be more readable
        comparison_results = comparison_results.rename(columns={
            'mae': 'MAE',
            'rmse': 'RMSE', 
            'r2': 'R2',
            'train_r2': 'Train_R2',
            'train_time': 'Train_Time'
        })

    print("\\n📈 Model Comparison Results:")
    print(comparison_results.to_string(index=False))

    best_model = comparison_results.iloc[0]['Model']
    best_mae = comparison_results.iloc[0]['MAE']
    best_r2 = comparison_results.iloc[0]['R2']"""
                    
                    source = source.replace(old_code, new_code)
                    cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
                    changes_made.append(f"Cell {i}: Fixed DataFrame conversion")
                    print(f"    [OK] Applied DataFrame conversion fix")
                else:
                    print(f"    [WARNING] Could not find exact pattern to replace")
            break

# ============================================================================
# FIX 2: Add checkpoint("regression_complete") at end of Phase 9.5
# ============================================================================
print("\n[4] Applying Fix 2: Add regression_complete checkpoint...")

# Find the Phase 9.5 cell (the one with train_and_compare_models execution)
phase95_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'print_section_header("PHASE 9.5' in source and 'out_models_dir = Path("outputs")' in source:
            phase95_cell_idx = i
            print(f"    Found Phase 9.5 execution cell at index {i}")
            break

if phase95_cell_idx is not None:
    # Check if checkpoint already exists at the end
    source = ''.join(notebook['cells'][phase95_cell_idx]['source'])
    if 'checkpoint("regression_complete"' in source:
        print("    [SKIP] Checkpoint already exists in Phase 9.5 cell")
    else:
        # Add checkpoint at the end of the try block, before the except
        # Find the location to insert (after print_phase_summary, before except)
        lines = source.split('\n')
        
        # Find where to insert (before the "except Exception as e:" line)
        insert_idx = None
        for idx, line in enumerate(lines):
            if 'except Exception as e:' in line and 'Phase 9.5 failed' in lines[idx+1] if idx+1 < len(lines) else False:
                insert_idx = idx
                break
        
        if insert_idx:
            # Get the indentation of the except block
            indent = '    '  # Standard 4-space indent
            
            checkpoint_code = [
                '',
                indent + '# Set checkpoint for Phase 9.5.1 and downstream phases',
                indent + 'checkpoint("regression_complete", requires=["classification_complete"])',
                indent + 'print("\\n✓ Checkpoint: regression_complete")',
                ''
            ]
            
            # Insert before except block
            lines = lines[:insert_idx] + checkpoint_code + lines[insert_idx:]
            
            # Update cell source
            source = '\n'.join(lines)
            notebook['cells'][phase95_cell_idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
            changes_made.append(f"Cell {phase95_cell_idx}: Added regression_complete checkpoint")
            print(f"    [OK] Added checkpoint call at line {insert_idx}")
        else:
            print("    [WARNING] Could not find insertion point for checkpoint")
else:
    print("    [WARNING] Could not find Phase 9.5 execution cell")

# ============================================================================
# Save Modified Notebook
# ============================================================================
print(f"\n[5] Saving modified notebook...")
with open('ml_finance_model_main_v10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"    [OK] Notebook saved")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("FIX SUMMARY")
print("=" * 80)

if len(changes_made) > 0:
    print("\nChanges applied:")
    for change in changes_made:
        print(f"  - {change}")
else:
    print("\n[INFO] No changes needed - fixes already applied")

print("\n" + "=" * 80)
print("CRITICAL: RESTART THE JUPYTER/PYCHARM KERNEL")
print("=" * 80)
print("""
The notebook file has been updated, but your Jupyter/PyCharm kernel is still
running OLD CACHED CODE from previous cell executions.

TO APPLY THE FIX:

1. In PyCharm:
   - Click "Interrupt Kernel" (stop icon)
   - Click "Restart Kernel" 
   - Or: Run → Restart Kernel

2. In Jupyter:
   - Kernel → Restart Kernel
   - Confirm the restart

3. Then run cells sequentially:
   - Start from Cell 1 (imports)
   - Run through Phase 9.1, 9.2, 9.3, 9.4
   - Run Phase 9.5 (Cell 140) - should now complete
   - Continue to Phase 9.5.1, 9.6, 9.7, 9.8

EXPECTED OUTPUT after restart:
  - Phase 9.5 Step 4: Model comparison with proper table
  - "✓ Best model: Ridge (MAE=1.15e-07, R²=1.0000)"
  - "✓ Checkpoint: regression_complete"
  - Phase 9.5.1 executes without checkpoint error
""")

print("\n" + "=" * 80)
print(f"Backup created: {backup_name}")
print("=" * 80)
