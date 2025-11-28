"""
Apply fix to Cell 62 in ml_finance_model_main.ipynb
Filters out sector interaction features when passing feature_cols to train_and_evaluate_regression_by_sector
"""

import json
import shutil
import sys
import io
from pathlib import Path
from datetime import datetime

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Backup the notebook first
notebook_path = Path("ml_finance_model_main.ipynb")
backup_path = (
    Path("backups")
    / f'ml_finance_model_main_fix62_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
)
backup_path.parent.mkdir(exist_ok=True)
shutil.copy2(notebook_path, backup_path)
print(f"[OK] Backed up notebook to: {backup_path}")

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and fix Cell 62
cells = nb["cells"]
target_cell_idx = None

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    # Look for the specific pattern in the problematic cell
    if (
        "train_and_evaluate_regression_by_sector" in source
        and "feature_cols=list(X_train.columns)" in source
        and "Failed to export enhanced predictions" in source
    ):
        target_cell_idx = i
        break

if target_cell_idx is None:
    print("[ERROR] Could not find target cell with the exact pattern")
    print("   Searching for cells with train_and_evaluate_regression_by_sector...")

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "train_and_evaluate_regression_by_sector" in source:
            print(f"   Found in cell {i}")
    exit(1)

print(f"[OK] Found target cell at index: {target_cell_idx}")

# Get the cell source
cell = cells[target_cell_idx]
source_lines = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]

# Find and replace the problematic line
modified = False
new_source = []

for line in source_lines:
    if "feature_cols=list(X_train.columns)" in line:
        print(f"[OK] Found problematic line:")
        print(f"  BEFORE: {line.strip()}")

        # Replace with filtered version
        # Preserve indentation
        indent = len(line) - len(line.lstrip())
        new_line = (
            " " * indent
            + "feature_cols=[c for c in X_train.columns if '__x__' not in c and c in all_stocks_enhanced.columns],\n"
        )

        print(f"  AFTER:  {new_line.strip()}")
        new_source.append(new_line)
        modified = True
    else:
        new_source.append(line)

if not modified:
    print("[ERROR] Could not find the exact line to replace")
    print("   Cell content:")
    print("".join(source_lines[:20]))
    exit(1)

# Update the cell
cells[target_cell_idx]["source"] = new_source

# Save the modified notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[SUCCESS] Applied fix to Cell {target_cell_idx}")
print(f"[OK] Modified notebook saved to: {notebook_path}")
print("\nFix details:")
print("  - Filtered out sector interaction features (containing '__x__')")
print("  - Only passes base features that exist in all_stocks_enhanced")
print("  - Sector-specific features will be regenerated per sector internally")
