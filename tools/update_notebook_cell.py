"""Script to update notebook cell 135 with overfitting detection validation."""

import json

# Read the notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# New cell content with overfitting detection
new_source = [
    "# Comprehensive regression metrics using Phase 9.6 function\n",
    "metrics = evaluation_comprehensive_metrics(y_test, y_pred_stacking)\n",
    'print("📊 Overall Model Performance:")\n',
    "for metric, value in metrics.items():\n",
    '    print(f"  {metric}: {value:.4f}")\n',
    "\n",
    "# ============================================================================\n",
    "# VALIDATION CHECKPOINT: Overfitting Detection (ml_workflow_guidelines.md)\n",
    "# ============================================================================\n",
    "# Per ml_workflow_guidelines.md Phase 9.5:\n",
    "# - R² >= 0.95 suggests data leakage - audit features\n",
    "# - MAE = 0 is impossible - check for target in features\n",
    "# - R² = 1.0 with MAE = 0 is the classic leakage pattern\n",
    "\n",
    "from finance_ml.ml_workflow.regression.constraints import validate_model_metrics, detect_overfitting\n",
    "\n",
    'print("\\n" + "=" * 70)\n',
    'print("VALIDATION CHECKPOINT: Overfitting Detection")\n',
    'print("=" * 70)\n',
    "\n",
    "validation_result = validate_model_metrics(metrics)\n",
    "\n",
    'if validation_result["valid"]:\n',
    '    print("✓ Model metrics are within expected ranges (no overfitting detected)")\n',
    "else:\n",
    '    print("⚠️  WARNING: Model metrics validation failed!")\n',
    "    print(f\"   Issues detected: {validation_result['issues']}\")\n",
    "    print(f\"   Summary: {validation_result['summary']}\")\n",
    "    \n",
    "    # Check for specific overfitting indicators\n",
    '    if "overfitting" in validation_result["issues"]:\n',
    '        overfitting_details = validation_result["details"].get("overfitting", {})\n',
    '        print(f"\\n   Overfitting Details:")\n',
    "        print(f\"   - Severity: {overfitting_details.get('severity', 'unknown')}\")\n",
    "        print(f\"   - Warnings: {overfitting_details.get('warnings', [])}\")\n",
    '        for rec in overfitting_details.get("recommendations", []):\n',
    '            print(f"   - Recommendation: {rec}")\n',
    "\n",
    'print("=" * 70)\n',
]

# Update cell 135
nb["cells"][135]["source"] = new_source

# Write back
with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated cell 135 with overfitting detection validation")
