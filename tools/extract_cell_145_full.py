import json
from pathlib import Path

# Load the notebook
notebook_path = Path(
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main_backup.ipynb"
)
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract cell 145 fully
cell = nb["cells"][145]
source = cell.get("source", [])
if isinstance(source, list):
    source_text = "".join(source)
else:
    source_text = source

print(f"CELL 145 FULL CONTENT ({len(source_text)} chars)")
print("=" * 80)

# Save to file for analysis
output_path = Path("cell_145_full.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(source_text)

print(f"✓ Full content saved to: {output_path}")
print(f"\nFirst 2000 chars:")
print(source_text[:2000])

# Analyze for key sections
print("\n" + "=" * 80)
print("SECTION ANALYSIS")
print("=" * 80)

sections = [
    ("9.5.2", "prepare_regression_data", "Prepare regression data"),
    ("9.5.3", "create_classification_interactions", "Create interaction features"),
    ("9.5.4", "compare_regressors", "Train and compare multiple regression models"),
    ("9.5.5", "train_sector_specific_models", "Sector-specific model optimization"),
    ("9.5.6", "train_quantile_regressor", "Quantile regression"),
    ("9.5.7", "save_model", "Model persistence"),
]

for section_num, function_name, description in sections:
    if section_num in source_text:
        print(f"✓ Found section marker: {section_num}")
    else:
        print(f"✗ Missing section marker: {section_num}")

    if function_name in source_text:
        print(f"  ✓ Found function call: {function_name}()")
    else:
        print(f"  ✗ Missing function call: {function_name}()")
    print()
