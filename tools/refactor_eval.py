import os

file_path = "finance_ml/ml_workflow/analytics/eval.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_marker = "# SHAP Detailed Analysis"
end_marker = "# Model Comparison Framework"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        # Go back 2 lines to include the separator line
        start_idx = i - 2
        break

for i, line in enumerate(lines):
    if end_marker in line:
        # Go back 2 lines to exclude the next separator line
        end_idx = i - 2
        break

if start_idx != -1 and end_idx != -1:
    print(f"Found block from line {start_idx} to {end_idx}")

    # Create replacement block
    replacement = [
        "# ============================================================================\n",
        "# SHAP Detailed Analysis (Moved to evaluation.explainability)\n",
        "# ============================================================================\n",
        "\n",
        "from finance_ml.ml_workflow.evaluation.explainability import (\n",
        "    compute_shap_values,\n",
        "    _detect_model_type,\n",
        "    create_shap_summary_plot,\n",
        "    create_shap_waterfall_plot,\n",
        "    create_shap_dependence_plot,\n",
        "    analyze_shap_by_sector,\n",
        "    explain_with_lime,\n",
        "    compare_lime_shap_consistency,\n",
        ")\n",
        "\n",
    ]

    new_lines = lines[:start_idx] + replacement + lines[end_idx:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("Successfully refactored eval.py")
else:
    print("Could not find markers")
    print(f"Start marker found: {start_idx != -1}")
    print(f"End marker found: {end_idx != -1}")
