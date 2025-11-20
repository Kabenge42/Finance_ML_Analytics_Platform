"""
Add clarifying comments to correlation matrix construction in Cells 83-84
to distinguish from train/test split ratios.
"""

import json
from pathlib import Path

notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
changes = []

# Fix Cell 83
cell83 = code_cells[83]
source83 = "".join(cell83["source"])
if "corr_matrix = np.eye(n_stocks) * 0.8 + np.random.rand(n_stocks, n_stocks) * 0.2" in source83:
    # Add comment before the line
    old_line = "    corr_matrix = np.eye(n_stocks) * 0.8 + np.random.rand(n_stocks, n_stocks) * 0.2"
    new_line = "    # Correlation matrix: 0.8 diagonal weight + 0.2 random (not train/test split)\n    corr_matrix = np.eye(n_stocks) * 0.8 + np.random.rand(n_stocks, n_stocks) * 0.2"
    source83 = source83.replace(old_line, new_line)
    cell83["source"] = [line + "\n" for line in source83.split("\n")[:-1]] + [
        source83.split("\n")[-1]
    ]
    changes.append("Cell 83: Added comment clarifying 0.8 is matrix weight")

# Fix Cell 84
cell84 = code_cells[84]
source84 = "".join(cell84["source"])
if (
    "corr_matrix = np.eye(len(opt_universe)) * 0.8 + np.random.rand(len(opt_universe), len(opt_universe)) * 0.2"
    in source84
):
    old_line = "    corr_matrix = np.eye(len(opt_universe)) * 0.8 + np.random.rand(len(opt_universe), len(opt_universe)) * 0.2"
    new_line = "    # Correlation matrix: 0.8 diagonal weight + 0.2 random (not train/test split)\n    corr_matrix = np.eye(len(opt_universe)) * 0.8 + np.random.rand(len(opt_universe), len(opt_universe)) * 0.2"
    source84 = source84.replace(old_line, new_line)
    cell84["source"] = [line + "\n" for line in source84.split("\n")[:-1]] + [
        source84.split("\n")[-1]
    ]
    changes.append("Cell 84: Added comment clarifying 0.8 is matrix weight")

if changes:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"✓ Added clarifying comments to correlation matrix construction")
    for change in changes:
        print(f"  - {change}")
else:
    print("No changes needed")
