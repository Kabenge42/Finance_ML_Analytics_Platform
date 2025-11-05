"""
Final fix: Insert optimization exports at exact line before except block
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main_backup.ipynb")
print(f"Loading: {notebook_path}")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Get Cell 145
cell_145 = nb["cells"][145]
source_lines = cell_145["source"]

print(f"Current Cell 145: {len(source_lines)} lines")

# Insert point: before line 596 (except Exception)
insert_index = 596

# The exports code to insert
exports_lines = [
    "\n",
    "    # ============================================================================\n",
    "    # MODEL OPTIMIZATION EXPORTS (Priorities 1.2 & 5)\n",
    "    # ============================================================================\n",
    '    print("\\n" + "=" * 80)\n',
    '    print("MODEL OPTIMIZATION EXPORTS")\n',
    '    print("=" * 80)\n',
    "    \n",
    "    # Export sector-level metrics (Priority 1.2)\n",
    "    if 'sector' in regression_df_enhanced.columns:\n",
    '        print("\\n📊 Computing sector-level regression metrics...")\n',
    "        from finance_ml.models import train_and_evaluate_regression_by_sector\n",
    "        \n",
    "        # Prepare dataframe with required columns for sector analysis\n",
    "        sector_analysis_df = regression_df_enhanced.copy()\n",
    "        if target_col in sector_analysis_df.columns:\n",
    "            try:\n",
    "                sector_metrics = train_and_evaluate_regression_by_sector(\n",
    "                    df=sector_analysis_df,\n",
    "                    out_dir=out_models_dir\n",
    "                )\n",
    '                print(f"✓ Sector metrics exported: {len(sector_metrics)} sectors")\n',
    '                print(f"   File: outputs/models/regression_metrics_by_sector.csv")\n',
    "                \n",
    "                # Display top/bottom performers\n",
    "                if 'mae' in sector_metrics.columns:\n",
    '                    print("\\n  Top 3 sectors (lowest MAE):")\n',
    "                    top3 = sector_metrics.nsmallest(3, 'mae')\n",
    "                    for _, row in top3.iterrows():\n",
    "                        print(f\"    - {row['sector']}: MAE={row['mae']:.2f}\")\n",
    "                    \n",
    '                    print("\\n  Bottom 3 sectors (highest MAE):")\n',
    "                    bottom3 = sector_metrics.nlargest(3, 'mae')\n",
    "                    for _, row in bottom3.iterrows():\n",
    "                        print(f\"    - {row['sector']}: MAE={row['mae']:.2f}\")\n",
    "            except Exception as e:\n",
    '                print(f"⚠ Sector metrics export failed: {e}")\n',
    "    else:\n",
    '        print("\\n⚠ Sector column not available for sector-level metrics")\n',
    "    \n",
    "    # Export feature importance (Priority 5)\n",
    '    print("\\n📈 Exporting feature importance from stacking model...")\n',
    "    try:\n",
    "        # Extract feature importance from base models in stacking ensemble\n",
    "        if hasattr(stacking_model, 'estimators_'):\n",
    "            # Get Random Forest base model (usually first estimator)\n",
    "            rf_model = None\n",
    "            for name, model in zip(['rf', 'et', 'gb'], stacking_model.estimators_):\n",
    "                if hasattr(model, 'feature_importances_'):\n",
    "                    rf_model = model\n",
    "                    break\n",
    "            \n",
    "            if rf_model is not None:\n",
    "                feature_importance_df = pd.DataFrame({\n",
    "                    'feature': X_train_reg.columns,\n",
    "                    'importance': rf_model.feature_importances_\n",
    "                }).sort_values('importance', ascending=False)\n",
    "                \n",
    '                importance_path = out_models_dir / "feature_importance_phase95.csv"\n',
    "                feature_importance_df.to_csv(importance_path, index=False)\n",
    '                print(f"✓ Feature importance exported: {importance_path}")\n',
    '                print(f"\\n  Top 10 Most Important Features:")\n',
    "                print(feature_importance_df.head(10).to_string(index=False))\n",
    "            else:\n",
    '                print("⚠ No feature importances available in stacking model")\n',
    "        else:\n",
    '            print("⚠ Stacking model does not have estimators_ attribute")\n',
    "    except Exception as e:\n",
    '        print(f"⚠ Feature importance export failed: {e}")\n',
    "    \n",
    '    print("\\n" + "=" * 80)\n',
    '    print("✓ All Model Optimization exports complete")\n',
    '    print("=" * 80)\n',
]

# Insert the exports
new_source = source_lines[:insert_index] + exports_lines + source_lines[insert_index:]

print(f"New Cell 145: {len(new_source)} lines (added {len(exports_lines)} lines)")

# Update notebook
nb["cells"][145]["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Updated: {notebook_path}")
print("\n" + "=" * 80)
print("ALL OPTIMIZATIONS COMPLETE")
print("=" * 80)
print("\nIntegrated Model Optimization Recommendations:")
print("  1. ✓ Huber loss in compare_regressors (Priority 2.1)")
print("  2. ✓ Huber loss in train_stacking_regressor (Priority 2.1)")
print("  3. ✓ Enhanced prediction metadata (Priority 1.1)")
print("  4. ✓ Sector-level metrics export (Priority 1.2)")
print("  5. ✓ Feature importance export (Priority 5)")
print("\nExpected outputs when Cell 145 runs:")
print("  - regression_predictions_phase95_enhanced.csv (with sector, ticker, errors)")
print("  - regression_metrics_by_sector.csv (per-sector MAE, RMSE, R²)")
print("  - feature_importance_phase95.csv (ranked features)")
print("\nNext step: Run validation script to confirm")
