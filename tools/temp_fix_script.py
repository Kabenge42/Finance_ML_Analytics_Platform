import json

# Load notebook
with open("ml_finance_model_main.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 91 (Step 3.5)
cell = nb["cells"][91]
source_lines = cell["source"]

# Find the line with the import statement
insert_idx = None
for i, line in enumerate(source_lines):
    if "from finance_ml.ml_workflow.analytics.eval import filter_stocks_by_criteria" in line:
        insert_idx = i
        break

if insert_idx is not None:
    # Create the fix code block to insert BEFORE the import
    fix_code = [
        "\n",
        "# ============================================================================\n",
        "# IMPORTANT: Ensure required columns exist before calling selection functions\n",
        "# ============================================================================\n",
        "\n",
        "# Step 3.4.5: Validate and compute required ranking metrics\n",
        'print("\\n🔍 Step 3.4.5: Validating required ranking metrics...")\n',
        "\n",
        "# 1. Ensure return_1y exists (realized 1-year return from prices)\n",
        'if "return_1y" not in valid_stocks_filtered.columns:\n',
        '    if {"last_price", "price_1y_ago"}.issubset(valid_stocks_filtered.columns):\n',
        '        valid_stocks_filtered["return_1y"] = (\n',
        '            valid_stocks_filtered["last_price"] / valid_stocks_filtered["price_1y_ago"] - 1.0\n',
        "        )\n",
        '        print(f"  ✓ Computed return_1y from price data: {valid_stocks_filtered[\\"return_1y\\"].notna().sum():,} stocks")\n',
        "    else:\n",
        "        raise KeyError(\n",
        '            "Notebook: cannot compute return_1y because columns "\n',
        "            \"'last_price' and 'price_1y_ago' are missing.\"\n",
        "        )\n",
        "else:\n",
        '    print(f"  ✓ return_1y already exists: {valid_stocks_filtered[\\"return_1y\\"].notna().sum():,} stocks")\n',
        "\n",
        "# 2. Ensure expected_return exists (model-based expected return)\n",
        'if "expected_return" not in valid_stocks_filtered.columns:\n',
        "    # Check for predicted_price_target column\n",
        '    if {"last_price", "predicted_price_target"}.issubset(valid_stocks_filtered.columns):\n',
        '        valid_stocks_filtered["expected_return"] = (\n',
        '            valid_stocks_filtered["predicted_price_target"] / valid_stocks_filtered["last_price"] - 1.0\n',
        "        )\n",
        '        print(f"  ✓ Computed expected_return from predictions: {valid_stocks_filtered[\\"expected_return\\"].notna().sum():,} stocks")\n',
        "    else:\n",
        "        raise KeyError(\n",
        '            "Notebook: cannot compute expected_return because columns "\n',
        "            \"'last_price' and 'predicted_price_target' are missing.\"\n",
        "        )\n",
        "else:\n",
        '    print(f"  ✓ expected_return already exists: {valid_stocks_filtered[\\"expected_return\\"].notna().sum():,} stocks")\n',
        "\n",
        "# 3. Ensure mispricing_score exists (required by select_portfolio_candidates)\n",
        'if "mispricing_score" not in valid_stocks_filtered.columns:\n',
        "    # Try to compute from mispricing_pct if available\n",
        '    if "mispricing_pct" in valid_stocks_filtered.columns:\n',
        "        # Simple normalization: convert percentage to score (0-100 scale)\n",
        '        valid_stocks_filtered["mispricing_score"] = valid_stocks_filtered["mispricing_pct"]\n',
        '        print(f"  ✓ Computed mispricing_score from mispricing_pct: {valid_stocks_filtered[\\"mispricing_score\\"].notna().sum():,} stocks")\n',
        "    else:\n",
        "        # Fallback: use expected_return as a proxy for mispricing score\n",
        '        print("  ⚠️  mispricing_pct not available; using expected_return as mispricing_score proxy")\n',
        '        valid_stocks_filtered["mispricing_score"] = valid_stocks_filtered["expected_return"] * 100\n',
        '        print(f"  ✓ Created mispricing_score from expected_return: {valid_stocks_filtered[\\"mispricing_score\\"].notna().sum():,} stocks")\n',
        "else:\n",
        '    print(f"  ✓ mispricing_score already exists: {valid_stocks_filtered[\\"mispricing_score\\"].notna().sum():,} stocks")\n',
        "\n",
        "# Display summary of ranking metrics\n",
        'print("\\n📊 Ranking metrics summary:")\n',
        'for metric in ["expected_return", "return_1y", "mispricing_score"]:\n',
        "    if metric in valid_stocks_filtered.columns:\n",
        "        vals = valid_stocks_filtered[metric].dropna()\n",
        "        if len(vals) > 0:\n",
        '            print(f"  {metric}: range [{vals.min():.3f}, {vals.max():.3f}], "\n',
        '                  f"mean={vals.mean():.3f}, median={vals.median():.3f}")\n',
        "\n",
    ]

    # Insert the fix code before the import line
    source_lines = source_lines[:insert_idx] + fix_code + source_lines[insert_idx:]

    # Update the cell
    cell["source"] = source_lines
    nb["cells"][91] = cell

    # Save the modified notebook
    with open("temp_cell_modified.txt", "w", encoding="utf-8") as f:
        f.write("".join(source_lines))

    print(f"Successfully inserted fix code at line {insert_idx}")
    print(f"Total lines in cell: {len(source_lines)}")
    print(f"Preview written to temp_cell_modified.txt")
else:
    print("ERROR: Could not find insertion point")
