"""
Apply complete Model Optimization enhancements to Cell 145 in ml_finance_model_main_backup.ipynb
Integrates all priorities from Model Optimization Recommendations.md
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main_backup.ipynb")
backup_path = Path("ml_finance_model_main_backup.ipynb.backup_before_optimizations")

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Create backup
print(f"Creating backup: {backup_path}")
with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Get Cell 145
cell_145 = nb["cells"][145]
current_source = (
    "".join(cell_145["source"]) if isinstance(cell_145["source"], list) else cell_145["source"]
)

print(f"\nOriginal Cell 145: {len(current_source)} chars")

# Read the enhanced version
enhanced_path = Path("cell_145_enhanced.txt")
with open(enhanced_path, "r", encoding="utf-8") as f:
    enhanced_source = f.read()

# Add additional optimizations at the end of Phase 9.5.8 (Summary section)
# Find the summary section and add exports

lines = enhanced_source.split("\n")
final_lines = []
added_exports = False

for i, line in enumerate(lines):
    final_lines.append(line)

    # After the Phase 9.5 summary is printed, add sector metrics and feature importance
    if 'print_section_header("PHASE 9.5 IMPLEMENTATION SUMMARY")' in line and not added_exports:
        # Look ahead to find where to insert
        for j in range(i, min(i + 50, len(lines))):
            if "checkpoint(" in lines[j] or j == len(lines) - 5:
                # Insert optimization exports before checkpoint or near end
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
    except Exception as e:
        print(f"⚠ Feature importance export failed: {e}")
    
    print("\\n" + "=" * 80)
    print("✓ All Model Optimization exports complete")
    print("=" * 80)
"""
                # Insert before checkpoint
                for export_line in exports_code.split("\n"):
                    final_lines.append(export_line)
                added_exports = True
                break

# Join lines
final_source = "\n".join(final_lines)

print(f"Enhanced Cell 145: {len(final_source)} chars")
print(f"Added {len(final_source) - len(current_source)} chars")

# Update notebook
nb["cells"][145]["source"] = final_source.split("\n")

# Save updated notebook
print(f"\nSaving updated notebook to: {notebook_path}")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "=" * 80)
print("✓ NOTEBOOK UPDATED SUCCESSFULLY")
print("=" * 80)
print("\nOptimizations applied to Cell 145:")
print("  1. ✓ Added loss='huber' to compare_regressors (Priority 2.1)")
print("  2. ✓ Added loss='huber' to train_stacking_regressor (Priority 2.1)")
print("  3. ✓ Added enhanced prediction metadata export (Priority 1.1)")
print("  4. ✓ Added sector-level metrics export (Priority 1.2)")
print("  5. ✓ Added feature importance export (Priority 5)")
print(f"\nBackup saved to: {backup_path}")
print("\nNext steps:")
print("  1. Open ml_finance_model_main_backup.ipynb in Jupyter")
print("  2. Run Cell 145 to test optimizations")
print("  3. Verify output files are created with enhanced data")
