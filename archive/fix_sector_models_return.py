import json

# Read the notebook
with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the cell with the error
fixed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'return sector_models_result' in source:
            # Replace the incorrect return statement
            new_source = source.replace('return sector_models_result', 'return sector_models')
            # Update the cell source (keeping it as a list of lines)
            cell['source'] = new_source.split('\n')
            # Add newline back to each line except the last
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                            for i, line in enumerate(cell['source'])]
            fixed = True
            print("[FIXED] Changed 'return sector_models_result' to 'return sector_models'")
            break

if not fixed:
    print("[ERROR] Could not find the problematic return statement")
    exit(1)

# Save the fixed notebook
with open('ml_finance_model_main_v10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[SUCCESS] Notebook saved successfully")
print("\nFixed the unresolved reference error:")
print("  - Changed 'return sector_models_result' to 'return sector_models'")
print("  - The function now correctly returns the 'sector_models' variable")
