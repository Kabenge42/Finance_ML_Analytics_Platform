"""
Fix parameter mismatches in ml_finance_model_main.ipynb.

This script parses the notebook JSON, finds cells with incorrect function calls,
and fixes the parameter names to match the actual function signatures.
"""

import json
import re
from pathlib import Path

def fix_notebook_parameters(notebook_path: Path):
    """Fix all 7 parameter mismatches in the notebook."""
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixes_applied = []
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell.get('source', [])
        if not source:
            continue
        
        # Join source lines for easier searching
        source_text = ''.join(source)
        modified = False
        
        # Fix #1: safety_rails_sensitivity_app - df_raw → data_df
        if 'safety_rails_sensitivity_app(' in source_text and 'df_raw=' in source_text:
            new_source = []
            for line in source:
                if 'df_raw=' in line:
                    new_line = line.replace('df_raw=', 'data_df=')
                    new_source.append(new_line)
                    modified = True
                    fixes_applied.append(f"Cell {cell_idx}: Fixed safety_rails_sensitivity_app - df_raw → data_df")
                else:
                    new_source.append(line)
            if modified:
                cell['source'] = new_source
                continue
        
        # Fix #2: estimate_sector_bias - remove extra column parameters
        if 'estimate_sector_bias(' in source_text:
            # Check if it has the wrong parameters
            if any(param in source_text for param in ['y_true_col=', 'y_pred_col=', 'y_pred_calibrated_col=', 'sector_col=']):
                new_source = []
                skip_line = False
                for i, line in enumerate(source):
                    # Skip lines with incorrect parameters
                    if any(param in line for param in ['y_true_col=', 'y_pred_col=', 'y_pred_calibrated_col=', 'sector_col=']):
                        # Check if line ends with comma, if so skip it
                        skip_line = True
                        modified = True
                        continue
                    else:
                        new_source.append(line)
                
                if modified:
                    cell['source'] = new_source
                    fixes_applied.append(f"Cell {cell_idx}: Fixed estimate_sector_bias - removed extra column parameters")
                    continue
        
        # Fix #3: plot_metrics_by_sector_time - metrics_history → predictions_df, snapshot_date_col → date_col
        if 'plot_metrics_by_sector_time(' in source_text:
            new_source = []
            for line in source:
                new_line = line
                if 'metrics_history=' in line:
                    new_line = new_line.replace('metrics_history=', 'predictions_df=')
                    modified = True
                if 'snapshot_date_col=' in line:
                    new_line = new_line.replace('snapshot_date_col=', 'date_col=')
                    modified = True
                new_source.append(new_line)
            
            if modified:
                cell['source'] = new_source
                fixes_applied.append(f"Cell {cell_idx}: Fixed plot_metrics_by_sector_time parameters")
                continue
        
        # Fix #4: create_sector_bias_dashboard - remove bias_dict parameter
        if 'create_sector_bias_dashboard(' in source_text and 'bias_dict=' in source_text:
            new_source = []
            for line in source:
                # Skip lines with bias_dict parameter
                if 'bias_dict=' in line:
                    modified = True
                    continue
                else:
                    new_source.append(line)
            
            if modified:
                cell['source'] = new_source
                fixes_applied.append(f"Cell {cell_idx}: Fixed create_sector_bias_dashboard - removed bias_dict")
                continue
        
        # Fix #5: compute_stacking_contributions - remove y_true parameter
        if 'compute_stacking_contributions(' in source_text and 'y_true=' in source_text:
            new_source = []
            for line in source:
                # Skip lines with y_true parameter
                if 'y_true=' in line and 'compute_stacking_contributions' in ''.join(source):
                    modified = True
                    continue
                else:
                    new_source.append(line)
            
            if modified:
                cell['source'] = new_source
                fixes_applied.append(f"Cell {cell_idx}: Fixed compute_stacking_contributions - removed y_true")
                continue
        
        # Fix #6: meta_error_maps - error_col/sector_col → remove them
        if 'meta_error_maps(' in source_text:
            if 'error_col=' in source_text or 'sector_col=' in source_text:
                new_source = []
                for line in source:
                    # Skip lines with error_col or sector_col
                    if 'error_col=' in line or ('sector_col=' in line and 'meta_error_maps' in ''.join(source)):
                        modified = True
                        continue
                    else:
                        new_source.append(line)
                
                if modified:
                    cell['source'] = new_source
                    fixes_applied.append(f"Cell {cell_idx}: Fixed meta_error_maps - removed error_col/sector_col")
                    continue
        
        # Fix #7: build_lineage_json - restructure to use model_info dict
        if 'build_lineage_json(' in source_text:
            # Check if it's using old structure (datasets=, features=, etc.)
            if any(param in source_text for param in ['datasets=', 'features=', 'models=', 'artifacts=', 'metrics=']):
                # Need to restructure this call - it's more complex
                # Look for the pattern and reconstruct
                new_source = []
                in_lineage_call = False
                lineage_params = {}
                
                for line in source:
                    if 'build_lineage_json(' in line:
                        in_lineage_call = True
                        new_source.append(line)
                        # Start collecting parameters
                        continue
                    
                    if in_lineage_call:
                        # Check for closing parenthesis
                        if ')' in line and '=' not in line:
                            # End of call - insert model_info dict
                            new_source.insert(-1, "    model_info = {\n")
                            for key, val in lineage_params.items():
                                new_source.insert(-1, f"        '{key}': {val},\n")
                            new_source.insert(-1, "    }\n")
                            new_source.insert(-1, "    lineage = build_lineage_json(\n")
                            new_source.insert(-1, "        model_info=model_info,\n")
                            new_source.insert(-1, "        output_dir=governance_dir,\n")
                            new_source.insert(-1, "        model_version=MODEL_VERSION\n")
                            new_source.insert(-1, "    )\n")
                            in_lineage_call = False
                            modified = True
                            break
                        
                        # Extract parameter
                        for param in ['datasets', 'features', 'models', 'artifacts', 'metrics']:
                            if f'{param}=' in line:
                                # Extract value
                                match = re.search(f'{param}=(.+?)(?:,|$)', line.strip())
                                if match:
                                    lineage_params[param] = match.group(1).strip().rstrip(',')
                                break
                        else:
                            # Keep other lines (output_dir, model_version)
                            if 'output_dir=' not in line and 'model_version=' not in line:
                                new_source.append(line)
                    else:
                        new_source.append(line)
                
                if modified:
                    cell['source'] = new_source
                    fixes_applied.append(f"Cell {cell_idx}: Fixed build_lineage_json - restructured to use model_info dict")
                    continue
    
    # Save modified notebook
    if fixes_applied:
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Created backup: {backup_path}")
        
        # Save fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"\nFixed {len(fixes_applied)} issues:")
        for fix in fixes_applied:
            print(f"  ✓ {fix}")
    else:
        print("No fixes needed or no matching patterns found.")
    
    return len(fixes_applied)

if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_main.ipynb")
    num_fixes = fix_notebook_parameters(notebook_path)
    print(f"\nTotal fixes applied: {num_fixes}")
