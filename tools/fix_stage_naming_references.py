"""
Fix outdated all_stocks references in ml_finance_model_main.ipynb
to align with DataFrame Stage Naming Convention (code_guidelines.md Section 8.2).

Fixes:
- Cell 12 line 7: all_stocks → all_stocks_winsorized (imputation input)
- Cell 12 line 17: all_stocks.isnull() → all_stocks_imputed.isnull() (validation)
- Cell 12 line 22: all_stocks → all_stocks_imputed (validation input)
- Cell 13 line 7: all_stocks.copy() → all_stocks_imputed.copy() (scaling input)
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

changes = []
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

# Fix Cell 12: Imputation cell
cell12 = code_cells[12]
source12 = "".join(cell12["source"])
original12 = source12

# Fix 1: Imputation input (line 7)
if "apply_enhanced_imputation_strategy_6step(\n        all_stocks," in source12:
    source12 = source12.replace(
        "apply_enhanced_imputation_strategy_6step(\n        all_stocks,",
        "apply_enhanced_imputation_strategy_6step(\n        all_stocks_winsorized,",
    )
    changes.append("Cell 12 line 7: Fixed imputation input all_stocks → all_stocks_winsorized")

# Fix 2: Missing values validation (line 17)
if "all_stocks.isnull().sum().sum()" in source12:
    source12 = source12.replace(
        "all_stocks.isnull().sum().sum()", "all_stocks_imputed.isnull().sum().sum()"
    )
    changes.append(
        "Cell 12 line 17: Fixed validation all_stocks.isnull() → all_stocks_imputed.isnull()"
    )

# Fix 3: Validation function input (line 22)
if "validate_imputation_completeness(\n        all_stocks," in source12:
    source12 = source12.replace(
        "validate_imputation_completeness(\n        all_stocks,",
        "validate_imputation_completeness(\n        all_stocks_imputed,",
    )
    changes.append("Cell 12 line 22: Fixed validation input all_stocks → all_stocks_imputed")

# Update cell 12 if changed
if source12 != original12:
    cell12["source"] = source12.split("\n")
    cell12["source"] = [line + "\n" for line in cell12["source"][:-1]] + [cell12["source"][-1]]

# Fix Cell 13: Scaling cell
cell13 = code_cells[13]
source13 = "".join(cell13["source"])
original13 = source13

# Fix 4: Scaling input (line 7)
if "scale_features(\n        all_stocks.copy()," in source13:
    source13 = source13.replace(
        "scale_features(\n        all_stocks.copy(),",
        "scale_features(\n        all_stocks_imputed.copy(),",
    )
    changes.append(
        "Cell 13 line 7: Fixed scaling input all_stocks.copy() → all_stocks_imputed.copy()"
    )

# Update cell 13 if changed
if source13 != original13:
    cell13["source"] = source13.split("\n")
    cell13["source"] = [line + "\n" for line in cell13["source"][:-1]] + [cell13["source"][-1]]

# Check for any other cells with problematic all_stocks references
# (excluding cells that correctly define stage names)
for i, cell in enumerate(code_cells):
    if i in [12, 13]:  # Already fixed
        continue

    source = "".join(cell["source"])

    # Look for standalone all_stocks references that should be stage-specific
    # Skip if cell is defining stage names or in config/early cells
    if i > 15:  # After preprocessing stages
        # Check for problematic patterns
        if "all_stocks." in source or "all_stocks[" in source or "(all_stocks," in source:
            # Make sure it's not a stage name
            if not any(
                stage in source
                for stage in [
                    "all_stocks_raw",
                    "all_stocks_normalized",
                    "all_stocks_typed",
                    "all_stocks_winsorized",
                    "all_stocks_imputed",
                    "all_stocks_scaled",
                    "all_stocks_features",
                    "all_stocks_enhanced",
                ]
            ):
                # Found potential issue - add to report
                snippet = source[:200].replace("\n", " ")
                changes.append(f"INFO: Cell {i} may need review: {snippet}...")

# Save updated notebook
if any("Fixed" in c for c in changes):
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print("✓ Fixed stage naming references in ml_finance_model_main.ipynb")
    print(f"  Changes applied: {len([c for c in changes if 'Fixed' in c])}")
    for change in changes:
        if "Fixed" in change:
            print(f"  - {change}")

    if any("INFO" in c for c in changes):
        print("\n  Additional cells that may need review:")
        for change in changes:
            if "INFO" in change:
                print(f"  - {change}")
else:
    print("No changes needed - all stage names already correct")
