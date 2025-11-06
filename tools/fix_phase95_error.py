"""
Fix Phase 9.5 KeyError: 'Model' issue in ml_finance_model_main_v10.ipynb

The compare_regressors function returns a dict like:
{
    "Ridge": {"mae": ..., "rmse": ..., "r2": ..., "train_r2": ..., "train_time": ...},
    "Lasso": {...},
    ...
}

But train_and_compare_models tries to convert it incorrectly and access ['Model'].
We need to properly structure the DataFrame.
"""
import json

# Read notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with train_and_compare_models function
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_and_compare_models' in source:
            print(f"Found train_and_compare_models in cell {i}")
            
            # Find and replace the problematic section
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
            
            if old_code in source:
                source = source.replace(old_code, new_code)
                # Update cell source (split back into lines)
                cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
                with open('fix_result.txt', 'w', encoding='utf-8') as out:
                    out.write("SUCCESS: Fixed the DataFrame conversion issue\n")
                print("Fixed the DataFrame conversion issue")
            else:
                with open('fix_result.txt', 'w', encoding='utf-8') as out:
                    out.write("WARNING: Could not find exact match for old code\n")
                    out.write("Searching for alternative pattern...\n")
                print("Could not find exact match for old code")
                print("Searching for alternative pattern...")
                
                # Try to find and fix the specific problematic line
                lines = source.split('\n')
                fixed = False
                for j, line in enumerate(lines):
                    if 'pd.DataFrame([comparison_results])' in line:
                        # Replace this line and add proper conversion
                        indent = len(line) - len(line.lstrip())
                        replacement = [
                            ' ' * indent + '# Convert from dict format to DataFrame with Model as a column',
                            ' ' * indent + 'comparison_results = pd.DataFrame.from_dict(comparison_results, orient=\'index\')',
                            ' ' * indent + 'comparison_results = comparison_results.reset_index().rename(columns={\'index\': \'Model\'})',
                            ' ' * indent + '# Sort by R2 score descending',
                            ' ' * indent + 'comparison_results = comparison_results.sort_values(\'r2\', ascending=False)',
                            ' ' * indent + '# Rename columns to be more readable',
                            ' ' * indent + 'comparison_results = comparison_results.rename(columns={',
                            ' ' * indent + '    \'mae\': \'MAE\',',
                            ' ' * indent + '    \'rmse\': \'RMSE\',',
                            ' ' * indent + '    \'r2\': \'R2\',',
                            ' ' * indent + '    \'train_r2\': \'Train_R2\',',
                            ' ' * indent + '    \'train_time\': \'Train_Time\'',
                            ' ' * indent + '})'
                        ]
                        lines[j] = '\n'.join(replacement)
                        fixed = True
                        break
                
                if fixed:
                    source = '\n'.join(lines)
                    cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
                    with open('fix_result.txt', 'a', encoding='utf-8') as out:
                        out.write("SUCCESS: Fixed using alternative pattern\n")
                    print("Fixed using alternative pattern")
                else:
                    with open('fix_result.txt', 'a', encoding='utf-8') as out:
                        out.write("ERROR: Could not automatically fix. Manual intervention needed.\n")
                    print("Could not automatically fix. Manual intervention needed.")
            
            break

# Save fixed notebook
with open('ml_finance_model_main_v10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

with open('fix_result.txt', 'a', encoding='utf-8') as out:
    out.write("\nSUCCESS: Notebook updated and saved\n")
    out.write("Please re-run the Phase 9.5 cell to test the fix\n")
print("\nNotebook updated and saved")
print("Please re-run the Phase 9.5 cell to test the fix")
