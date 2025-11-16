#!/usr/bin/env python
"""
Fix malformed Time-Series Cross-Validation cell in notebook.
The cell has all code compressed into a single line, making it unreadable.
"""
import json
from pathlib import Path


def find_malformed_cell(nb):
    """Find the malformed Time-Series CV cell."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # Check for Time-Series CV cell with very few lines but many characters
            # (indicating compressed/malformed code)
            if "Time-Series Cross-Validation" in source and "TimeSeriesSplit" in source:
                num_lines = len(cell["source"])
                total_chars = len(source)
                # If we have >2000 chars in <10 lines, it's malformed
                if total_chars > 2000 and num_lines < 10:
                    return i, cell
    return None, None


def format_timeseries_cv_code():
    """Return properly formatted Time-Series CV code."""
    return """print("\\n" + "=" * 80)
print("6.5.1 — Time-Series Cross-Validation (5 folds)")
print("=" * 80)
try:
    # Identify a date column
    date_col = None
    for cand in ['date', 'as_of_date', 'last_updated', 'income_statement_report_date']:
        if cand in all_stocks_enhanced.columns:
            date_col = cand
            break
    
    if date_col is None:
        print("⚠ No date column found; skipping Time-Series CV")
    else:
        df_cv = all_stocks_enhanced.copy()
        df_cv[date_col] = pd.to_datetime(df_cv[date_col], errors='coerce')
        df_cv = df_cv.sort_values(date_col).dropna(subset=[target_col])
        
        feature_cols_cv = list(X_train.columns)
        X_cv = df_cv[feature_cols_cv].fillna(0)
        y_cv = df_cv[target_col]
        
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
        rows = []
        fold_num = 0
        
        for train_idx, test_idx in tscv.split(X_cv):
            fold_num += 1
            X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
            y_tr, y_te = y_cv.iloc[train_idx], y_cv.iloc[test_idx]
            
            # Train a lightweight stacking model per fold (reuse robust settings)
            fold_result = regression_train_stacking(
                X_tr, winsorize_target(y_tr, 0.01, 0.99), cv=3, ensure_nonnegative=True, loss="huber"
            )
            fold_model = fold_result['model']
            
            # Apply adaptive clipping with percentile-based bounds
            fold_pred = fold_model.predict(X_te)
            clip_result_fold = adaptive_clip_predictions(fold_pred, y_tr)
            y_hat = clip_result_fold['clipped_predictions']
            
            # Optional: log clipping stats for first fold
            if fold_num == 1:
                print(f"  Fold clipping: lower=${clip_result_fold['lower_bound']:.2f}, "
                      f"upper=${clip_result_fold['upper_bound']:.2f}")
            
            mae = mean_absolute_error(y_te, y_hat)
            rmse = np.sqrt(mean_squared_error(y_te, y_hat))
            r2 = r2_score(y_te, y_hat)
            rows.append({'fold': fold_num, 'mae': mae, 'rmse': rmse, 'r2': r2, 'n_test': len(y_te)})
        
        tscv_df = pd.DataFrame(rows)
        eval_dir = OUTPUT_DIR / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        tscv_path = eval_dir / 'tscv_metrics.csv'
        tscv_df.to_csv(tscv_path, index=False)
        print(f"✓ Saved Time-Series CV metrics to {tscv_path}")
        print(tscv_df.describe().loc[['mean', 'std']])
except Exception as e:
    print(f"⚠ Time-Series CV evaluation skipped: {e}")
"""


def fix_notebook(notebook_path):
    """Fix the malformed Time-Series CV cell."""
    print("=" * 80)
    print("FIXING MALFORMED TIME-SERIES CV CELL")
    print("=" * 80)

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # Find the malformed cell
    cell_idx, cell = find_malformed_cell(nb)

    if cell_idx is None:
        print("⚠ Could not find malformed Time-Series CV cell")
        return False

    print(f"\n✓ Found malformed cell at index {cell_idx}")
    source = "".join(cell["source"])
    print(f"  Current length: {len(source)} characters")
    print(f"  Current lines: {len(cell['source'])} (should be ~60+)")

    # Replace with properly formatted code
    formatted_code = format_timeseries_cv_code()
    cell["source"] = formatted_code.split("\n")

    print(f"\n✓ Reformatted cell:")
    print(f"  New length: {len(formatted_code)} characters")
    print(f"  New lines: {len(cell['source'])}")

    # Save fixed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Fixed notebook saved to: {notebook_path}")
    return True


def verify_fix(notebook_path):
    """Verify the fix was applied correctly."""
    print("\n" + "=" * 80)
    print("VERIFYING FIX")
    print("=" * 80)

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the Time-Series CV cell again
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "Time-Series Cross-Validation" in source and "TimeSeriesSplit" in source:
                print(f"✓ Found Time-Series CV cell at index {i}")
                print(f"  Lines: {len(cell['source'])}")
                print(f"  First 5 lines:")
                for idx, line in enumerate(cell["source"][:5], 1):
                    print(f"    {idx}: {line}")

                # Verify it's properly formatted
                if len(cell["source"]) > 50:
                    print("\n✓ Cell is properly formatted (>50 lines)")
                    return True
                else:
                    print(f"\n⚠ Cell still appears malformed ({len(cell['source'])} lines)")
                    return False

    print("⚠ Could not find Time-Series CV cell")
    return False


if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_main.ipynb")

    # Create backup
    backup_path = Path("ml_finance_model_main.ipynb.before_tscv_fix")
    import shutil

    shutil.copy2(notebook_path, backup_path)
    print(f"✓ Backup created: {backup_path}\n")

    # Apply fix
    success = fix_notebook(notebook_path)

    if success:
        # Verify fix
        verify_fix(notebook_path)

        print("\n" + "=" * 80)
        print("FIX COMPLETE")
        print("=" * 80)
        print("\nThe Time-Series Cross-Validation cell has been reformatted.")
        print("The code is now properly structured with correct line breaks and indentation.")
    else:
        print("\n" + "=" * 80)
        print("FIX FAILED")
        print("=" * 80)
