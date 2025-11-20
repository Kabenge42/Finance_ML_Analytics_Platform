"""
Script to fix remaining TDD compliance issues:
1. Replace hardcoded 'last_price' in cell 51 with TARGET_COL_FALLBACK
2. Replace hardcoded 0.25 in cell 81 with MAX_SECTOR_WEIGHT
3. Propagate all_stocks_scaled to downstream cells that still use all_stocks
"""

import json
import re
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

changes = []

# Process code cells
for cell_idx, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])
    original_source = source

    # Skip config cell
    if cell_idx == 0:
        continue

    # 1. Fix hardcoded 'last_price' in list context (cell 51 according to tests)
    # Look for patterns like: ['last_price', 'price_target', ...]
    if cell_idx >= 45 and cell_idx <= 55:  # Around cell 51
        if "'last_price'" in source and (
            "core_valuation_cols" in source or "feature_cols" in source or "in [" in source
        ):
            # Replace 'last_price' with TARGET_COL_FALLBACK in list contexts
            source = re.sub(
                r"(\[.*?)'last_price'(.*?\])", r"\1TARGET_COL_FALLBACK\2", source, flags=re.DOTALL
            )
            if source != original_source:
                changes.append(f"Cell {cell_idx}: Replaced 'last_price' with TARGET_COL_FALLBACK")

    # 2. Fix hardcoded 0.25 for MAX_SECTOR_WEIGHT (cell 81 according to tests)
    # Look for max_sector_weight=0.25 or similar patterns
    if cell_idx >= 75 and cell_idx <= 85:  # Around cell 81
        if "max_sector_weight" in source.lower() or "sector_weight" in source.lower():
            # Replace = 0.25 with = MAX_SECTOR_WEIGHT
            source = re.sub(r"=\s*0\.25\b", "=MAX_SECTOR_WEIGHT", source)
            if source != original_source:
                changes.append(f"Cell {cell_idx}: Replaced 0.25 with MAX_SECTOR_WEIGHT")

    # 3. Propagate all_stocks_scaled to downstream cells (after preprocessing)
    # Skip if this is a preprocessing cell (cells 0-20 typically)
    if cell_idx > 20:
        # Replace standalone all_stocks references with all_stocks_scaled
        # But be careful not to replace stage names we just created

        # Pattern 1: all_stocks[ or all_stocks. (column access)
        if re.search(r"\ball_stocks\[", source) or re.search(r"\ball_stocks\.", source):
            # Don't replace if it's already a stage name
            if not any(
                stage in source
                for stage in [
                    "all_stocks_raw",
                    "all_stocks_normalized",
                    "all_stocks_typed",
                    "all_stocks_winsorized",
                    "all_stocks_imputed",
                    "all_stocks_scaled",
                ]
            ):
                # Replace all_stocks with all_stocks_scaled
                source = re.sub(r"\ball_stocks\[", "all_stocks_scaled[", source)
                source = re.sub(r"\ball_stocks\.", "all_stocks_scaled.", source)
                if source != original_source:
                    changes.append(f"Cell {cell_idx}: Propagated all_stocks_scaled")

        # Pattern 2: (all_stocks) or (all_stocks,
        if re.search(r"\(all_stocks[,\)]", source):
            if not any(
                stage in source
                for stage in [
                    "all_stocks_raw",
                    "all_stocks_normalized",
                    "all_stocks_typed",
                    "all_stocks_winsorized",
                    "all_stocks_imputed",
                    "all_stocks_scaled",
                ]
            ):
                source = re.sub(r"\(all_stocks([,\)])", r"(all_stocks_scaled\1", source)
                if source != original_source and cell_idx not in [
                    c.split(":")[0].split()[-1] for c in changes
                ]:
                    changes.append(
                        f"Cell {cell_idx}: Propagated all_stocks_scaled in function args"
                    )

    # Update cell if changed
    if source != original_source:
        cell["source"] = source.split("\n")
        # Ensure each line ends with \n except the last
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Fixed remaining TDD compliance issues")
print(f"  Changes made: {len(changes)}")
for change in changes:
    print(f"  - {change}")
