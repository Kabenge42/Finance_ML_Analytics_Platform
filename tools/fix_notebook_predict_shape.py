"""Fix notebook cell to add shape correction for y_pred_test."""

import json
import sys

notebook_path = "../ml_finance_model_main.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell with predict calls
target_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if (
            "y_pred_test = cls_model.predict" in source
            and "y_proba_test = cls_model.predict_proba" in source
        ):
            target_cell_idx = i
            print(f"Found target cell at index {i}")
            print(f"Execution count: {cell.get('execution_count')}")
            break

if target_cell_idx is None:
    print("ERROR: Could not find target cell")
    sys.exit(1)

# Get the cell source
cell = nb["cells"][target_cell_idx]
source_lines = cell["source"]

# Find the line after y_proba_test assignment
insert_idx = None
for i, line in enumerate(source_lines):
    if "y_proba_test = cls_model.predict_proba" in line:
        # Insert after this line (and after the closing of the else block)
        insert_idx = i + 1
        break

if insert_idx is None:
    print("ERROR: Could not find insertion point")
    sys.exit(1)

# Check if shape correction already exists
if any("Ensure y_pred_test is 1D" in line for line in source_lines):
    print("Shape correction already exists in notebook")
    sys.exit(0)

# Create shape correction code
shape_correction = [
    "\n",
    "# Ensure y_pred_test is 1D (defensive fix for shape mismatches)\n",
    "import numpy as np\n",
    "if hasattr(y_pred_test, 'ndim') and y_pred_test.ndim == 2:\n",
    "    if y_pred_test.shape[1] == 1:\n",
    "        y_pred_test = y_pred_test.ravel()\n",
    '        print(f"  ⚠️  Converted y_pred_test from shape {y_pred_test.shape} to 1D")\n',
    "    else:\n",
    "        # Model returned probabilities instead of labels - use argmax\n",
    "        y_pred_test = np.argmax(y_pred_test, axis=1)\n",
    '        print(f"  ⚠️  y_pred_test was 2D (probabilities). Converted to 1D labels using argmax.")\n',
]

# Insert the shape correction code
source_lines[insert_idx:insert_idx] = shape_correction

# Update the cell
nb["cells"][target_cell_idx]["source"] = source_lines

# Save the notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Successfully added shape correction code to cell {target_cell_idx}")
print(f"  Inserted {len(shape_correction)} lines after line {insert_idx}")
