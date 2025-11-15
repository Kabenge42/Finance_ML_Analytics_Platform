"""
Fix Cell 145 - Add missing sector metrics and feature importance exports
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
source_145 = (
    "".join(cell_145["source"]) if isinstance(cell_145["source"], list) else cell_145["source"]
)

print(f"Current Cell 145: {len(source_145)} chars")

# Find the right insertion point - after summary section, before checkpoint or exception handler
lines = source_145.split("\n")
new_lines = []
inserted = False

for i, line in enumerate(lines):
    new_lines.append(line)

    # Look for the summary printing section
    if 'print("✓ Checkpoint: regression_complete")' in line or (
        "except Exception as e:" in line and "Phase 9.5" in "".join(lines[max(0, i - 50) : i])
    ):
        # Insert exports BEFORE this line
        if not inserted:
            # Remove the line we just added (we'll add it after the exports)
            new_lines.pop()

            exports_code = """
    # ============================================================================
    # MODEL OPTIMIZATION EXPORTS (Priorities 1.2 & 5)
    # ============================================================================
    print("\\n" + "=" * 80)
    print("MODEL OPTIMIZATION EXPORTS")
    print("=" * 80)
    
    # Export sector-level metrics (Priority 1.2)
    if 'sector' in regression_df_enhanced.columns:
        print("\\n📊 Computing sector-level regression metrics...")
        from finance_ml.regression import train_and_evaluate_regression_by_sector
        
        # Prepare dataframe with required columns for sector analysis
        sector_analysis_df = regression_df_enhanced.copy()
        if target_col in sector_analysis_df.columns:
            try:
                sector_metrics = train_and_evaluate_regression_by_sector(
                    df=sector_analysis_df,
                    out_dir=out_models_dir
                )
                print(f"✓ Sector metrics exported: {len(sector_metrics)} sectors")
                print(f"   File: outputs/regression/regression_metrics_by_sector.csv")
                
                # Display top/bottom performers
                if 'mae' in sector_metrics.columns:
                    print("\\n  Top 3 sectors (lowest MAE):")
                    top3 = sector_metrics.nsmallest(3, 'mae')
                    for _, row in top3.iterrows():
                        print(f"    - {row['sector']}: MAE={row['mae']:.2f}")
                    
                    print("\\n  Bottom 3 sectors (highest MAE):")
                    bottom3 = sector_metrics.nlargest(3, 'mae')
                    for _, row in bottom3.iterrows():
                        print(f"    - {row['sector']}: MAE={row['mae']:.2f}")
            except Exception as e:
                print(f"⚠ Sector metrics export failed: {e}")
    else:
        print("\\n⚠ Sector column not available for sector-level metrics")
    
    # Export feature importance (Priority 5)
    print("\\n📈 Exporting feature importance from stacking model...")
    try:
        # Extract feature importance from base regression in stacking ensemble
        if hasattr(stacking_model, 'estimators_'):
            # Get Random Forest base model (usually first estimator)
            rf_model = None
            for name, model in zip(['rf', 'et', 'gb'], stacking_model.estimators_):
                if hasattr(model, 'feature_importances_'):
                    rf_model = model
                    break
            
            if rf_model is not None:
                feature_importance_df = pd.DataFrame({
                    'feature': X_train_reg.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = out_models_dir / "feature_importance_phase95.csv"
                feature_importance_df.to_csv(importance_path, index=False)
                print(f"✓ Feature importance exported: {importance_path}")
                print(f"\\n  Top 10 Most Important Features:")
                print(feature_importance_df.head(10).to_string(index=False))
            else:
                print("⚠ No feature importances available in stacking model")
        else:
            print("⚠ Stacking model does not have estimators_ attribute")
    except Exception as e:
        print(f"⚠ Feature importance export failed: {e}")
    
    print("\\n" + "=" * 80)
    print("✓ All Model Optimization exports complete")
    print("=" * 80)
"""
            # Add the exports
            for export_line in exports_code.split("\n"):
                new_lines.append(export_line)

            # Now add back the original line
            new_lines.append(line)
            inserted = True
            print(f"✓ Inserted exports before line {i}: {line[:50]}...")

# Join and update
final_source = "\n".join(new_lines)
print(f"\nFinal Cell 145: {len(final_source)} chars")
print(f"Added: {len(final_source) - len(source_145)} chars")

if inserted:
    # Update notebook
    nb["cells"][145]["source"] = final_source.split("\n")

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Updated: {notebook_path}")
    print("\nOptimizations now complete:")
    print("  1. ✓ Huber loss in compare_regressors")
    print("  2. ✓ Huber loss in train_stacking_regressor")
    print("  3. ✓ Enhanced prediction metadata")
    print("  4. ✓ Sector-level metrics export")
    print("  5. ✓ Feature importance export")
else:
    print("\n⚠ Could not find insertion point")
    print("Searching for alternative locations...")

    # Try to find where the try block ends
    for i, line in enumerate(lines[-20:]):
        print(f"  Line {len(lines)-20+i}: {line[:80]}")
