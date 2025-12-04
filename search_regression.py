"""Search notebook for regression imports and functions."""

import json
import sys
import re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    data = json.load(f)

cells = data["cells"]

# Search patterns
patterns = [
    r"from finance_ml.*regression",
    r"import.*regression",
    r"train_.*regressor",
    r"compare_regressors",
    r"train_ridge",
    r"train_lasso",
    r"train_elastic",
    r"train_bayesian",
    r"train_polynomial",
    r"train_xgboost",
    r"train_lightgbm",
    r"train_catboost",
    r"train_histgb",
    r"train_random_forest",
    r"train_extra_trees",
    r"train_neural_network",
    r"train_voting",
    r"train_stacking",
    r"all_stocks_enhanced",
    r"all_stocks_classification",
]

print("=" * 80)
print("REGRESSION-RELATED CODE IN NOTEBOOK")
print("=" * 80)

for i, cell in enumerate(cells):
    source = "".join(cell["source"])
    for pattern in patterns:
        matches = re.findall(pattern, source, re.IGNORECASE)
        if matches:
            print(f"\nCell {i} ({cell['cell_type']}): Found '{pattern}'")
            for match in matches[:3]:
                print(f"  - {match}")

# Also show all imports from finance_ml in cells 0-10
print("\n" + "=" * 80)
print("ALL FINANCE_ML IMPORTS (Cells 0-10)")
print("=" * 80)

for i in range(min(11, len(cells))):
    cell = cells[i]
    source = "".join(cell["source"])
    if "from finance_ml" in source or "import finance_ml" in source:
        lines = [l for l in source.split("\n") if "finance_ml" in l]
        print(f"\nCell {i}:")
        for line in lines[:20]:
            print(f"  {line[:100]}")
