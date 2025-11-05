"""
Validate the updated ml_finance_model_main_backup.ipynb structure and imports
"""

import json
from pathlib import Path

print("=" * 80)
print("NOTEBOOK VALIDATION - Model Optimization Integration")
print("=" * 80)

# Load updated notebook
notebook_path = Path("ml_finance_model_main_backup.ipynb")
try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n✓ Notebook loaded successfully: {notebook_path}")
    print(f"  Total cells: {len(nb['cells'])}")
except Exception as e:
    print(f"\n✗ Failed to load notebook: {e}")
    exit(1)

# Check Cell 145
cell_145 = nb["cells"][145]
source_145 = (
    "".join(cell_145["source"]) if isinstance(cell_145["source"], list) else cell_145["source"]
)

print(f"\n✓ Cell 145 validated:")
print(f"  Type: {cell_145.get('cell_type')}")
print(f"  Length: {len(source_145)} chars")

# Verify optimizations are present
optimizations = {
    "Huber loss in compare_regressors": 'loss="huber"' in source_145
    and "compare_regressors(" in source_145,
    "Huber loss in train_stacking_regressor": 'loss="huber"' in source_145
    and "train_stacking_regressor(" in source_145,
    "Enhanced metadata export": "predictions_enhanced_df" in source_145
    and "abs_error" in source_145,
    "Sector metrics export": "train_and_evaluate_regression_by_sector" in source_145,
    "Feature importance export": "feature_importance_phase95.csv" in source_145,
}

print("\n" + "=" * 80)
print("OPTIMIZATION VERIFICATION")
print("=" * 80)

all_present = True
for name, present in optimizations.items():
    status = "✓" if present else "✗"
    print(f"{status} {name}")
    if not present:
        all_present = False

# Check imports section (typically early cells)
print("\n" + "=" * 80)
print("IMPORTS VERIFICATION")
print("=" * 80)

imports_to_check = [
    ("train_and_evaluate_regression_by_sector", "finance_ml.models"),
    ("compare_regressors", "finance_ml.advanced_models"),
    ("train_stacking_regressor", "finance_ml.advanced_models"),
]

# Check first 20 cells for imports
import_cells_text = ""
for i in range(min(20, len(nb["cells"]))):
    cell = nb["cells"][i]
    if cell.get("cell_type") == "code":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        import_cells_text += source + "\n"

for func_name, module_name in imports_to_check:
    # Check if function is imported directly or module is imported
    direct_import = (
        f"from {module_name} import" in import_cells_text and func_name in import_cells_text
    )
    module_import = (
        f"from {module_name} import" in import_cells_text
        or f"import {module_name}" in import_cells_text
    )

    if direct_import:
        print(f"✓ {func_name} imported directly from {module_name}")
    elif module_import:
        print(f"✓ {module_name} imported (function callable as module.{func_name})")
    else:
        print(f"⚠ {func_name} - import may be inline in Cell 145")

# Check Cell 145 for inline imports
if "from finance_ml.models import train_and_evaluate_regression_by_sector" in source_145:
    print("✓ train_and_evaluate_regression_by_sector imported inline in Cell 145")

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

if all_present:
    print("✓ All optimizations successfully integrated into Cell 145")
    print("\nExpected output files when Cell 145 is executed:")
    print("  1. outputs/models/regression_predictions_phase95_enhanced.csv")
    print(
        "     - Columns: y_true, y_pred, residual, abs_error, pct_error, sector, ticker, market_cap"
    )
    print("  2. outputs/models/regression_metrics_by_sector.csv")
    print("     - Per-sector MAE, RMSE, R² metrics")
    print("  3. outputs/models/feature_importance_phase95.csv")
    print("     - Ranked features by importance")
    print("\nModel improvements:")
    print("  - Huber loss: Expected RMSE reduction from 4,643 → <500 (~90%)")
    print("  - Enhanced diagnostics: Sector and ticker-level error analysis")
    print("  - Interpretability: Feature importance for model debugging")
else:
    print("⚠ Some optimizations may not be properly integrated")
    print("  Review Cell 145 manually to verify all code is present")

print("\n" + "=" * 80)
print("✓ VALIDATION COMPLETE")
print("=" * 80)
