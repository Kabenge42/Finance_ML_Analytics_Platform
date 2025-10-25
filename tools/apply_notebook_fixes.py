import json
from pathlib import Path

# Paths
nb_path = Path(r"/ml_finance_model_main.ipynb")

# Load notebook JSON
with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = 0

# Corrected code blocks from issue description
DATA_VALIDATION_CODE = """# Unified validation reporting with error handling
try:
    from finance_ml.data import validate_schema, check_missing_values
    
    # Schema validation
    try:
        is_valid, missing_cols = validate_schema(all_stocks)
        if is_valid:
            print("\u2713 Schema validation passed")
        else:
            print(f"\u26a0 Schema validation: missing columns {missing_cols}")
    except Exception as e:
        print(f"\u26a0 Schema validation skipped: {e}")
    
    # Missing values check
    try:
        missing_report = all_stocks.isnull().sum()
        missing_pct = (missing_report / len(all_stocks) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing_report[missing_report > 0],
            'Missing %': missing_pct[missing_report > 0]
        }).sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print("\n\ud83d\udcca Missing Values Report:")
            print(missing_df.head(10).to_string())
        else:
            print("\u2713 No missing values detected")
    except Exception as e:
        print(f"\u26a0 Missing value check failed: {e}")
        
except Exception as e:
    logger.error(f"Validation failed: {e}")
    print(f"\u26a0 Validation checks incomplete")"""

EDA_CODE = """# Unified EDA display with proper output directory
try:
    from pathlib import Path
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    from finance_ml.data import simple_eda
    
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    simple_eda(all_stocks, out_dir=output_dir)
    print(f"\u2713 EDA completed - outputs saved to {output_dir}")
    
except Exception as e:
    logger.error(f"EDA failed: {e}")
    print(f"\u26a0 EDA failed: {e}")"""

CLASSIFICATION_CODE = """# Create event labels for classification
try:
    event_labels = create_event_labels(all_stocks_processed)
    print(f"\n\u2713 Event labels created")
    print(f"  Label distribution: {pd.Series(event_labels).value_counts().to_dict()}")

    # Remove duplicate columns before training
    df_for_classifier = all_stocks_processed.copy()
    duplicate_cols = df_for_classifier.columns[df_for_classifier.columns.duplicated()].unique()
    
    if len(duplicate_cols) > 0:
        print(f"  Removing duplicate columns: {list(duplicate_cols)}")
        # Keep only first occurrence of each column
        df_for_classifier = df_for_classifier.loc[:, ~df_for_classifier.columns.duplicated()]
    
    # Train event classifier using the cleaned DataFrame
    classifier_results = train_event_classifier(df_for_classifier, event_labels)
    print(f"\n\u2713 Event classifier trained")
    print(f"  Accuracy: {classifier_results.get('accuracy', 0):.4f}")
    print(f"  F1 Score (macro): {classifier_results.get('f1_macro', 0):.4f}")
    
except Exception as e:
    logger.error(f"Classification training failed: {e}")
    print(f"\u2717 Classification training failed: {e}")"""

VALUATION_CODE = """# Calculate mispricing scores
try:
    # Get predictions from regression model
    if 'predictions' in regression_results and regression_results['predictions'] is not None:
        predictions = regression_results['predictions']
        
        # Properly align predictions with dataframe using index
        if isinstance(predictions, pd.DataFrame):
            # predictions is a DataFrame with y_pred column
            pred_series = predictions['y_pred']
            all_stocks_processed.loc[pred_series.index, 'predicted_target'] = pred_series.values
        elif isinstance(predictions, pd.Series):
            all_stocks_processed.loc[predictions.index, 'predicted_target'] = predictions.values
        else:
            print("\u26a0 Unexpected prediction format")
            raise ValueError("Predictions must be DataFrame or Series")

        # Calculate mispricing only for rows with predictions
        mask = all_stocks_processed['predicted_target'].notna()
        df_with_pred = all_stocks_processed[mask].copy()
        
        mispricing = calculate_mispricing_score(df_with_pred)
        all_stocks_processed.loc[mask, 'mispricing_score'] = mispricing

        print(f"\n\u2713 Mispricing scores calculated for {mask.sum()} stocks")
        print(f"  Mean mispricing: {mispricing.mean():.4f}")
        print(f"  Std mispricing: {mispricing.std():.4f}")

        # Rank undervalued stocks (only from rows with mispricing scores)
        df_scored = all_stocks_processed[all_stocks_processed['mispricing_score'].notna()].copy()
        
        if len(df_scored) >= 10:
            undervalued = rank_undervalued_stocks(df_scored, top_n=10)
            print(f"\nTop 10 Undervalued Stocks:")
            display_cols = [c for c in ['ticker', 'name', 'sector', 'mispricing_score'] if c in undervalued.columns]
            print(undervalued[display_cols].to_string())

            # Rank overvalued stocks
            overvalued = rank_overvalued_stocks(df_scored, top_n=10)
            print(f"\nTop 10 Overvalued Stocks:")
            print(overvalued[display_cols].to_string())
        else:
            print(f"\u26a0 Insufficient scored stocks ({len(df_scored)}) for ranking")
    else:
        print("\u26a0 No predictions available from regression model")

except Exception as e:
    logger.error(f"Valuation analysis failed: {e}")
    print(f"\u2717 Valuation analysis failed: {e}")
    import traceback
    traceback.print_exc()"""

VISUALIZATION_CODE = """# Create sector heatmap
try:
    if 'sector' in all_stocks_processed.columns and 'mispricing_score' in all_stocks_processed.columns:
        df_for_viz = all_stocks_processed[all_stocks_processed['mispricing_score'].notna()].copy()
        if len(df_for_viz) > 0:
            fig = create_sector_heatmap(df_for_viz)
            print("\n\u2713 Sector heatmap created")
        else:
            print("\n\u26a0 No data with mispricing scores for sector heatmap")
except Exception as e:
    logger.error(f"Sector heatmap failed: {e}")
    print(f"\u2717 Sector heatmap creation failed: {e}")

# Create interactive prediction plot
try:
    # Remove duplicate columns before plotting
    df_for_plot = all_stocks_processed.copy()
    df_for_plot = df_for_plot.loc[:, ~df_for_plot.columns.duplicated()]
    
    if 'predicted_target' in df_for_plot.columns and 'price_target' in df_for_plot.columns:
        # Filter to rows with both values
        mask = df_for_plot['predicted_target'].notna() & df_for_plot['price_target'].notna()
        df_for_plot = df_for_plot[mask]
        
        if len(df_for_plot) > 0:
            fig = create_interactive_prediction_plot(df_for_plot)
            print("\u2713 Interactive prediction plot created")
        else:
            print("\u26a0 No data with both predicted and actual targets for plot")
    else:
        print("\u26a0 Missing required columns for prediction plot")
        
except Exception as e:
    logger.error(f"Prediction plot failed: {e}")
    print(f"\u2717 Prediction plot creation failed: {e}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)"""

ENHANCED_MODELS_CODE = """from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from finance_ml import (
        train_and_evaluate_regression_by_sector,
        train_quantile_regression_by_sector,
        predict_quantile_regression,
        train_stacking_ensemble_by_sector,
        export_predictions_to_excel,
    )
    HAVE_ENHANCED_MODELS = True
except Exception as e:
    print(f"\u26a0 Enhanced modeling imports unavailable: {e}")
    HAVE_ENHANCED_MODELS = False

# Ensure output directory
try:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

# Remove duplicate columns first
df_enhanced = all_stocks_processed.loc[:, ~all_stocks_processed.columns.duplicated()].copy()

# --- Per-sector regression metrics ---
if HAVE_ENHANCED_MODELS:
    try:
        print("\n" + "=" * 80)
        print("PER-SECTOR REGRESSION METRICS")
        print("=" * 80)
        sector_metrics = train_and_evaluate_regression_by_sector(df_enhanced, output_dir)
        display_cols = [c for c in ["sector", "n_train", "n_test", "mae", "rmse", "r2"] if c in sector_metrics.columns]
        if len(sector_metrics) > 0:
            print(sector_metrics[display_cols].to_string(index=False))
        else:
            print("(No sectors with sufficient data)")
    except Exception as e:
        print(f"\u26a0 Per-sector regression metrics step skipped: {e}")

# --- Quantile regression by sector (uncertainty bands) ---
quantile_models_by_sector = {}
if HAVE_ENHANCED_MODELS:
    try:
        # Determine target column
        target_col = None
        for cand in ["price_target", "price_target_median"]:
            if cand in df_enhanced.columns:
                target_col = cand
                break
        
        if target_col is None:
            raise ValueError("No target column found")

        # Select numeric feature columns only
        numeric_cols = [
            c for c in df_enhanced.columns
            if pd.api.types.is_numeric_dtype(df_enhanced[c])
        ]
        blacklist = {target_col, "predicted_target", "mispricing_score"}
        feature_cols = [c for c in numeric_cols if c not in blacklist]

        if len(feature_cols) < 3:
            raise ValueError(f"Insufficient numeric features ({len(feature_cols)}) for quantile regression")

        # Check we have enough clean data
        required_cols = feature_cols + [target_col]
        if 'sector' in df_enhanced.columns:
            required_cols.append('sector')
        
        df_clean = df_enhanced[required_cols].dropna()
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient clean data ({len(df_clean)} rows) for quantile regression")

        print("\n" + "=" * 80)
        print("QUANTILE REGRESSION BY SECTOR (q10/q50/q90)")
        print("=" * 80)
        
        quantile_models_by_sector = train_quantile_regression_by_sector(
            df_enhanced, feature_cols, target_col, quantiles=[0.1, 0.5, 0.9]
        )

        # Generate quantile predictions per sector
        prediction_count = 0
        for sector, model in quantile_models_by_sector.items():
            sec_mask = (df_enhanced.get("sector") == sector) if "sector" in df_enhanced.columns else pd.Series(False, index=df_enhanced.index)
            if sec_mask.sum() == 0:
                continue
                
            X_sec = df_enhanced.loc[sec_mask, feature_cols].dropna()
            if len(X_sec) == 0:
                continue
                
            preds_df = predict_quantile_regression(model, X_sec, quantiles=[0.1, 0.5, 0.9])
            
            # Assign predictions back to main frame
            for col in preds_df.columns:
                df_enhanced.loc[X_sec.index, col] = preds_df[col].values
            prediction_count += len(X_sec)

        if prediction_count > 0:
            print(f"\u2713 Quantile bands added for {prediction_count} stocks across {len(quantile_models_by_sector)} sectors")
        else:
            print("\u26a0 No quantile predictions generated")

    except Exception as e:
        print(f"\u26a0 Quantile regression step skipped: {e}")

# --- Stacking ensemble by sector ---
stacking_models_by_sector = {}
if HAVE_ENHANCED_MODELS:
    try:
        if 'feature_cols' in locals() and 'target_col' in locals():
            cols_needed = [c for c in (feature_cols + [target_col, 'sector']) if c in df_enhanced.columns]
            df_stack = df_enhanced[cols_needed].dropna()
            
            if len(df_stack) >= 100:  # Increased threshold for stability
                print("\n" + "=" * 80)
                print("STACKING ENSEMBLE BY SECTOR")
                print("=" * 80)
                stacking_models_by_sector = train_stacking_ensemble_by_sector(
                    df_stack, feature_cols, target_col
                )
                print(f"\u2713 Trained stacking ensembles for {len(stacking_models_by_sector)} sectors")
            else:
                print(f"\u26a0 Skipping stacking ensemble — insufficient clean rows: {len(df_stack)} (need 100+)")
        else:
            print("\u26a0 Skipping stacking ensemble — features/target not available")
    except Exception as e:
        print(f"\u26a0 Stacking ensemble step skipped: {e}")

# --- Excel export ---
if HAVE_ENHANCED_MODELS:
    try:
        if 'mispricing_score' in df_enhanced.columns and df_enhanced['mispricing_score'].notna().sum() > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = output_dir / f"Stock_Prediction_Analysis_Report_{ts}.xlsx"
            export_predictions_to_excel(df_enhanced, excel_path, include_summary=True)
            print(f"\n\u2713 Exported predictions and summaries to Excel: {excel_path}")
        else:
            print("\u26a0 Excel export skipped — mispricing_score not found; run valuation analysis first")
    except ImportError as e:
        print(f"\u26a0 Excel export skipped (engine missing): {e}")
    except Exception as e:
        print(f"\u26a0 Excel export failed: {e}")

# Update main dataframe with enhancements
all_stocks_processed = df_enhanced"""


def set_cell_source(cell, code_str):
    # Split into lines as Jupyter expects list of strings
    cell["source"] = [
        line + ("\n" if not line.endswith("\n") else "") for line in code_str.split("\n")
    ]


for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))

    # 1) Data Validation replacement
    if "validate_and_display_data(all_stocks)" in src:
        set_cell_source(cell, DATA_VALIDATION_CODE)
        replacements += 1
        continue

    # 2) EDA replacement
    if "perform_and_display_eda(all_stocks)" in src:
        set_cell_source(cell, EDA_CODE)
        replacements += 1
        continue

    # 3) Classification training replacement
    if "train_event_classifier(all_stocks_processed" in src:
        set_cell_source(cell, CLASSIFICATION_CODE)
        replacements += 1
        continue

    # 4) Valuation analysis replacement
    if (
        "Calculate mispricing scores" in src
        or "all_stocks_processed['predicted_target'] = predictions" in src
    ):
        set_cell_source(cell, VALUATION_CODE)
        replacements += 1
        continue

    # 5) Visualization replacement
    if (
        "create_interactive_prediction_plot(all_stocks_processed)" in src
        and "create_sector_heatmap" in src
    ):
        set_cell_source(cell, VISUALIZATION_CODE)
        replacements += 1
        continue

    # 6) Enhanced models replacement
    if (
        "train_and_evaluate_regression_by_sector" in src
        and "export_predictions_to_excel" in src
        and "quantile" in src.lower()
    ):
        set_cell_source(cell, ENHANCED_MODELS_CODE)
        replacements += 1
        continue

# Save updated notebook
with nb_path.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Applied replacements: {replacements}")
print(f"Notebook updated: {nb_path}")
