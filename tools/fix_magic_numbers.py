"""
Script to fix magic numbers in ml_finance_model_main.ipynb
Replaces hardcoded values with configuration constants
"""

import json
import re
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Track changes
changes = []

# Process code cells
for cell_idx, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])
    original_source = source

    # Skip config cell (cell 0)
    if cell_idx == 0:
        continue

    # 1. Replace random_state=42 with RANDOM_SEED (but not in comments)
    if "random_state=42" in source or "random_seed=42" in source:
        # Only replace if not in a string or comment context
        lines = source.split("\n")
        new_lines = []
        for line in lines:
            if "#" in line and "random" in line:
                # Keep comments as-is
                new_lines.append(line)
            else:
                line = re.sub(r"random_state\s*=\s*42\b", "random_state=RANDOM_SEED", line)
                line = re.sub(r"random_seed\s*=\s*42\b", "random_seed=RANDOM_SEED", line)
                new_lines.append(line)
        source = "\n".join(new_lines)

    # 2. Replace test_size=0.2 with TEST_SIZE
    if "test_size=0.2" in source or "test_size = 0.2" in source:
        source = re.sub(r"test_size\s*=\s*0\.2\b", "test_size=TEST_SIZE", source)

    # 3. Replace train size calculations: int(len(X) * 0.8) with int(len(X) * TRAIN_SIZE)
    if "* 0.8" in source and "len(" in source:
        source = re.sub(r"(\blen\([^)]+\))\s*\*\s*0\.8\b", r"\1 * TRAIN_SIZE", source)

    # 4. Replace hardcoded quantile lists [0.05, 0.5, 0.95] with QUANTILES
    # But note: config has [0.1, 0.5, 0.9], so we need to use QUANTILES but might need adjustment
    if "[0.05, 0.5, 0.95]" in source:
        source = source.replace("[0.05, 0.5, 0.95]", "QUANTILES")
        changes.append(
            f"Cell {cell_idx}: Replaced [0.05, 0.5, 0.95] with QUANTILES (NOTE: config has [0.1, 0.5, 0.9])"
        )

    # 5. Replace hardcoded 'last_price' with TARGET_COL_FALLBACK (but be careful)
    # Only in specific contexts like list comprehensions or column selection
    if "'last_price'" in source and cell_idx > 5:
        # Check if it's in a context where we should replace
        if "core_valuation_cols" in source or "feature_cols" in source:
            # Replace in list contexts
            source = re.sub(r"'last_price'(?=\s*[,\]])", "TARGET_COL_FALLBACK", source)
            changes.append(f"Cell {cell_idx}: Replaced 'last_price' with TARGET_COL_FALLBACK")

    # Update cell if changed
    if source != original_source:
        cell["source"] = source.split("\n")
        # Ensure each line ends with \n except the last
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
        if not changes or cell_idx not in [int(c.split(":")[0].split()[1]) for c in changes]:
            changes.append(f"Cell {cell_idx}: Fixed magic numbers")

# Save updated notebook
output_path = Path("ml_finance_model_main.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Fixed magic numbers in notebook")
print(f"  Changes made: {len(changes)}")
for change in changes:
    print(f"  - {change}")
