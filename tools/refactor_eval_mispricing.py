import os

file_path = "finance_ml/ml_workflow/analytics/eval.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_marker = "def calculate_mispricing_score("
end_marker = "def simple_eda("

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        # The function definition starts here. But we want to replace from line 98 which is where this definition starts.
        # Line 98 in file matches lines[97].
        start_idx = i
        break

for i, line in enumerate(lines):
    if end_marker in line:
        # Go back 2 lines to exclude the separators or whitespace before simple_eda
        end_idx = i - 2
        break

if start_idx != -1 and end_idx != -1:
    print(f"Found block from line {start_idx} to {end_idx}")

    # Create replacement block
    replacement = [
        "# ============================================================================\n",
        "# Mispricing Functions (Moved to analytics.mispricing)\n",
        "# ============================================================================\n",
        "\n",
        "from finance_ml.ml_workflow.analytics.mispricing import (\n",
        "    calculate_mispricing_score,\n",
        "    calculate_mispricing_from_predictions_schema,\n",
        "    calculate_risk_adjusted_mispricing,\n",
        "    calculate_risk_adjusted_mispricing_from_predictions_schema,\n",
        "    rank_undervalued_stocks,\n",
        "    rank_overvalued_stocks,\n",
        "    rank_stocks_by_sector,\n",
        ")\n",
        "\n",
    ]

    # Check if there is extra whitespace to remove around start_idx
    # In the file view, line 97 is empty. Line 98 is def calculate...
    # So start_idx points to def...

    new_lines = lines[:start_idx] + replacement + lines[end_idx:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("Successfully refactored eval.py for mispricing")
else:
    print("Could not find markers")
    print(f"Start marker found: {start_idx != -1}")
    print(f"End marker found: {end_idx != -1}")
