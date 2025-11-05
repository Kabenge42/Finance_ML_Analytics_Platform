"""
Update Cell 145 in ml_finance_model_main_backup.ipynb to integrate
Model Optimization Recommendations.
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main_backup.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Get current Cell 145 content
cell_145 = nb["cells"][145]
current_source = (
    "".join(cell_145["source"]) if isinstance(cell_145["source"], list) else cell_145["source"]
)

print("Current Cell 145 length:", len(current_source), "chars")
print("\nSearching for optimization points...")

# Check what needs to be added
optimizations_needed = []

if "loss=" not in current_source:
    optimizations_needed.append("Add loss='huber' parameter to model training")

if "abs_error" not in current_source or "pct_error" not in current_source:
    optimizations_needed.append("Add enhanced metadata to predictions export")

if "train_and_evaluate_regression_by_sector" not in current_source:
    optimizations_needed.append("Add sector-level metrics export")

if "feature_importance" not in current_source:
    optimizations_needed.append("Add feature importance export")

print(f"\nOptimizations needed ({len(optimizations_needed)}):")
for i, opt in enumerate(optimizations_needed, 1):
    print(f"  {i}. {opt}")

# Create enhanced version of Cell 145
# We'll insert optimization code at strategic points

# Key insertions:
# 1. After line with compare_regressors call - add loss parameter
# 2. After stacking model predictions - add enhanced metadata export
# 3. After Phase 9.5 summary - add sector metrics and feature importance

lines = current_source.split("\n")
new_lines = []
insertions_made = 0

for i, line in enumerate(lines):
    new_lines.append(line)

    # Optimization 1: Add loss='huber' to compare_regressors
    if "comparison_results = compare_regressors(" in line and "loss=" not in lines[i : i + 10]:
        print(f"\n✓ Found compare_regressors call at line {i}")
        # Look ahead to find the closing parenthesis
        for j in range(i, min(i + 10, len(lines))):
            if "ensure_nonnegative=" in lines[j]:
                # Insert loss parameter before closing paren
                indent = "                "
                new_lines.append(
                    f'{indent}loss="huber"  # Robust loss for outlier handling (Priority 2.1)'
                )
                insertions_made += 1
                break

    # Optimization 2: Add loss='huber' to train_stacking_regressor
    if (
        "stacking_model, stacking_results = train_stacking_regressor(" in line
        and "loss=" not in lines[i : i + 10]
    ):
        print(f"\n✓ Found train_stacking_regressor call at line {i}")
        for j in range(i, min(i + 10, len(lines))):
            if "ensure_nonnegative=" in lines[j]:
                indent = "            "
                new_lines.append(f'{indent}loss="huber"  # Robust loss for outlier handling')
                insertions_made += 1
                break

    # Optimization 3: Add enhanced prediction export after test metrics
    if "test_metrics = {" in line and i + 10 < len(lines):
        # Find the end of test_metrics dict
        for j in range(i, min(i + 20, len(lines))):
            if lines[j].strip() == "}" or lines[j].strip().endswith("}"):
                # Check if enhanced export doesn't already exist
                if "predictions_enhanced_df" not in "".join(lines[j : j + 30]):
                    print(f"\n✓ Found test_metrics at line {i}, will add enhanced export")
                    # We'll add the export code here
                    enhancement_code = """
    # Export predictions with enhanced metadata (Model Optimization Priority 1.1)
    predictions_enhanced_df = pd.DataFrame({
        'y_true': y_test_reg.values,
        'y_pred': y_pred_stacking,
        'residual': y_test_reg.values - y_pred_stacking,
        'abs_error': np.abs(y_test_reg.values - y_pred_stacking),
        'pct_error': ((y_test_reg.values - y_pred_stacking) / y_test_reg.values) * 100,
    })
    
    # Add sector, ticker, and market_cap if available
    if 'sector' in regression_df_enhanced.columns:
        sector_map = regression_df_enhanced.loc[X_test_reg.index, 'sector']
        predictions_enhanced_df['sector'] = sector_map.values
    if 'ticker' in regression_df_enhanced.columns:
        ticker_map = regression_df_enhanced.loc[X_test_reg.index, 'ticker']
        predictions_enhanced_df['ticker'] = ticker_map.values
    if 'market_cap' in regression_df_enhanced.columns:
        mcap_map = regression_df_enhanced.loc[X_test_reg.index, 'market_cap']
        predictions_enhanced_df['market_cap'] = mcap_map.values
    
    # Save enhanced predictions
    pred_path = out_models_dir / "regression_predictions_phase95_enhanced.csv"
    predictions_enhanced_df.to_csv(pred_path, index=False)
    print(f"\\n✓ Saved enhanced predictions to {pred_path}")
    print(f"   Columns: {list(predictions_enhanced_df.columns)}")
"""
                    for code_line in enhancement_code.split("\n"):
                        new_lines.append(code_line)
                    insertions_made += 1
                break

# Write results
print(f"\n{'='*80}")
print(f"Total insertions made: {insertions_made}")
print(f"New cell length: {len(chr(10).join(new_lines))} chars")

# Save to a new file for review
output_path = Path("cell_145_enhanced.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print(f"\n✓ Enhanced cell saved to: {output_path}")
print("\nNext steps:")
print("  1. Review cell_145_enhanced.txt")
print("  2. Apply changes to notebook if satisfied")
print("  3. Add sector metrics and feature importance exports")
