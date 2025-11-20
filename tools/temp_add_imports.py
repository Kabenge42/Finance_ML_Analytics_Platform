#!/usr/bin/env python3
"""Add missing Phase 9.2 imports to Cell 4."""

import json
from pathlib import Path

notebook_path = Path("ml_finance_model_main.ipynb")

# Read notebook
with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

# Modify Cell 4 (imports cell)
cell4 = nb["cells"][4]
source = cell4["source"]

# Insert new imports after line 87 (after sector_distribution_summary closing paren)
new_imports = [
    "# Phase 9.2 enhanced analytics functions (eval.py)\n",
    "from finance_ml.ml_workflow.analytics.eval import (\n",
    "    calculate_financial_metrics_dashboard,\n",
    "    generate_data_quality_alerts,\n",
    "    perform_comprehensive_hypothesis_tests,\n",
    "    )\n",
    "\n",
]

# Insert at index 88 (after line 87 which is `)`)
cell4["source"] = source[:88] + new_imports + source[88:]

print(f"[OK] Added {len(new_imports)} import lines to Cell 4")
print(f"  Original length: {len(source)} lines")
print(f"  New length: {len(cell4['source'])} lines")

# Write back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Updated {notebook_path}")
