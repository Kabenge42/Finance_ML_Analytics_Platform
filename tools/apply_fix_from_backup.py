import json

# Read the backup file with the corrected cell
with open("phase93_category_cells_backup.json", "r", encoding="utf-8") as f:
    backup_cells = json.load(f)

# Read the main notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find the cell with syntax error in the notebook
corrected_cell = backup_cells[1]  # Second cell in backup has the corrected visualization code
target_cell_found = False

for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "Category Performance Heatmaps" in source and "category_mapping.items()" in source:
            # Replace with corrected version from backup
            notebook["cells"][i]["source"] = corrected_cell["source"]
            target_cell_found = True
            print(f"✓ Found and fixed cell at index {i}")
            break

if not target_cell_found:
    print("❌ Target cell not found in notebook")
    exit(1)

# Save the corrected notebook
with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✓ Successfully applied fix from backup to notebook")
print("\nFixed syntax errors:")
print("  1. Indented 'available_in_category' assignment inside the for loop")
print("  2. Properly indented the if/continue block")
print("  3. Indented all code blocks within the for loop")
print("  4. Fixed indentation throughout the cell")
