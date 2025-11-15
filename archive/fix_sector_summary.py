import json
import sys

# Read the notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and fix cell 64
cell = nb["cells"][64]
lines = cell["source"]

# Find the line with sector_summary creation
for i, line in enumerate(lines):
    if "sector_summary = pd.DataFrame(sector_results).T" in line:
        print(f"Found problematic code at line index {i}")

        # Replace lines 321-325 with fixed version
        # Original:
        # 321:         # Display sector results
        # 322:         sector_summary = pd.DataFrame(sector_results).T
        # 323:         sector_summary = sector_summary.sort_values('r2', ascending=False)
        # 324:         print(f"\n📊 Sector Model Performance:")
        # 325:         print(sector_summary.head(10).to_string())

        new_lines = [
            "        # Display sector results\n",
            "        # Extract sector_metrics from results dict (train_sector_specific_models returns {'sector_metrics': {...}, 'n_sectors': N})\n",
            "        if 'sector_metrics' in sector_results:\n",
            "            sector_summary = pd.DataFrame(sector_results['sector_metrics']).T\n",
            "        else:\n",
            "            # Fallback: use sector_results directly if already in expected format\n",
            "            sector_summary = pd.DataFrame(sector_results).T\n",
            "        \n",
            "        # Debug: Check available columns\n",
            '        print(f"\\n🔍 Available columns in sector_summary: {sector_summary.columns.tolist()}")\n',
            "        \n",
            "        # Sort by r2 if it exists, otherwise by train_score, otherwise by mae\n",
            "        if 'r2' in sector_summary.columns:\n",
            "            sector_summary = sector_summary.sort_values('r2', ascending=False)\n",
            "            sort_metric = 'r2'\n",
            "        elif 'train_score' in sector_summary.columns:\n",
            "            sector_summary = sector_summary.sort_values('train_score', ascending=False)\n",
            "            sort_metric = 'train_score'\n",
            "        elif 'mae' in sector_summary.columns:\n",
            "            sector_summary = sector_summary.sort_values('mae', ascending=True)  # Lower MAE is better\n",
            "            sort_metric = 'mae'\n",
            "        else:\n",
            '            print(f"\\n⚠ Warning: None of the expected metric columns (r2, train_score, mae) found.")\n',
            '            print(f"   Available columns: {sector_summary.columns.tolist()}")\n',
            "            sort_metric = 'none'\n",
            "        \n",
            '        print(f"\\n📊 Sector Model Performance (sorted by {sort_metric}):")\n',
            "        \n",
            "        # Display results with available columns\n",
            "        if not sector_summary.empty:\n",
            "            display_cols = [c for c in ['train_score', 'r2', 'mae', 'rmse', 'cv_mean', 'model_type'] if c in sector_summary.columns]\n",
            "            if display_cols:\n",
            "                print(sector_summary[display_cols].head(10).to_string())\n",
            "            else:\n",
            "                print(sector_summary.head(10).to_string())\n",
            "        else:\n",
            '            print("   No sector regression trained (insufficient samples per sector)")\n',
        ]

        # Replace lines 321-325 (indices i-1 to i+4)
        lines[i - 1 : i + 4] = new_lines

        print(f"Fixed {len(new_lines)} lines")
        break

# Save the fixed notebook
with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook fixed successfully")
