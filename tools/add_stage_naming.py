"""
Script to add stage-based DataFrame naming to ml_finance_model_main.ipynb
Implements Phase 2 of TDD refactoring: DataFrame state tracking
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path("ml_finance_model_main.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

changes = []

# Process code cells
for cell_idx, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])
    original_source = source

    # Skip config cell
    if cell_idx == 0:
        continue

    # 1. Initial data loading (around line 523-528)
    # Change: all_stocks = load_from_db(...) → all_stocks_raw = load_from_db(...)
    if "all_stocks = load_from_db" in source or "all_stocks = load_from_csv" in source:
        source = source.replace("all_stocks = load_from_db", "all_stocks_raw = load_from_db")
        source = source.replace("all_stocks = load_from_csv", "all_stocks_raw = load_from_csv")
        source = source.replace("len(all_stocks)", "len(all_stocks_raw)")
        changes.append(f"Cell {cell_idx}: Added all_stocks_raw for initial load")

    # 2. Normalization (around line 532)
    # Change: all_stocks = normalize_columns(all_stocks) → all_stocks_normalized = normalize_columns(all_stocks_raw)
    if "all_stocks = normalize_columns" in source:
        source = source.replace(
            "all_stocks = normalize_columns(all_stocks)",
            "all_stocks_normalized = normalize_columns(all_stocks_raw)",
        )
        # Also update validate_schema call
        source = source.replace(
            "validate_schema(all_stocks", "validate_schema(all_stocks_normalized"
        )
        # Update print statements
        source = source.replace("all_stocks.shape", "all_stocks_normalized.shape")
        source = source.replace("all_stocks.isnull()", "all_stocks_normalized.isnull()")
        changes.append(f"Cell {cell_idx}: Added all_stocks_normalized after normalize_columns")

    # 3. Dtype detection (around line 552)
    # Change: all_stocks, dtype_diagnostics = detect_and_cast_dtypes(all_stocks)
    # to: all_stocks_typed, dtype_diagnostics = detect_and_cast_dtypes(all_stocks_normalized)
    if "all_stocks, dtype_diagnostics = detect_and_cast_dtypes" in source:
        source = source.replace(
            "all_stocks, dtype_diagnostics = detect_and_cast_dtypes(all_stocks)",
            "all_stocks_typed, dtype_diagnostics = detect_and_cast_dtypes(all_stocks_normalized)",
        )
        # Update references in same cell
        source = source.replace("all_stocks.columns", "all_stocks_typed.columns")
        changes.append(f"Cell {cell_idx}: Added all_stocks_typed after dtype detection")

    # 4. Update missing value analysis to use all_stocks_typed
    if "missing_report = check_missing_values(all_stocks)" in source:
        source = source.replace(
            "missing_report = check_missing_values(all_stocks)",
            "missing_report = check_missing_values(all_stocks_typed)",
        )

    # 5. Update catalog registration to use all_stocks_typed
    if "all_stocks_initial" in source and "shape" in source and "catalog_metadata" in source:
        source = source.replace("all_stocks.shape", "all_stocks_typed.shape")
        source = source.replace("all_stocks.columns", "all_stocks_typed.columns")

    # 6. Update outlier detection to use all_stocks_typed (becomes input)
    if "detect_outliers_iqr" in source and "all_stocks.select_dtypes" in source:
        source = source.replace("all_stocks.select_dtypes", "all_stocks_typed.select_dtypes")
        source = source.replace(
            "detect_outliers_iqr(\n            all_stocks,",
            "detect_outliers_iqr(\n            all_stocks_typed,",
        )
        source = source.replace(
            "detect_outliers_zscore(\n            all_stocks,",
            "detect_outliers_zscore(\n            all_stocks_typed,",
        )
        source = source.replace(
            "detect_outliers_isolation_forest(\n            all_stocks,",
            "detect_outliers_isolation_forest(\n            all_stocks_typed,",
        )

    # 7. Winsorization (around line 691)
    # Change: all_stocks = winsorize_by_sector(all_stocks, ...)
    # to: all_stocks_winsorized = winsorize_by_sector(all_stocks_typed, ...)
    if "all_stocks = winsorize_by_sector" in source:
        source = source.replace(
            "all_stocks = winsorize_by_sector(\n        all_stocks,",
            "all_stocks_winsorized = winsorize_by_sector(\n        all_stocks_typed,",
        )
        changes.append(f"Cell {cell_idx}: Added all_stocks_winsorized after winsorization")

    # 8. Update quality report to use winsorized data
    if "quality_report = preprocessing_calculate_quality(all_stocks)" in source:
        source = source.replace(
            "quality_report = preprocessing_calculate_quality(all_stocks)",
            "quality_report = preprocessing_calculate_quality(all_stocks_winsorized)",
        )

    # 9. Update visualizations to use winsorized data
    if "missing_pct = (all_stocks.isnull()" in source:
        source = source.replace("all_stocks.isnull()", "all_stocks_winsorized.isnull()")
        source = source.replace("len(all_stocks)", "len(all_stocks_winsorized)")

    # 10. 6-step imputation (around line 800+)
    # Change: all_stocks_imputed = apply_enhanced_imputation_strategy_6step(all_stocks, ...)
    # to: all_stocks_imputed = apply_enhanced_imputation_strategy_6step(all_stocks_winsorized, ...)
    if "apply_enhanced_imputation_strategy_6step" in source:
        source = source.replace(
            "apply_enhanced_imputation_strategy_6step(\n    all_stocks,",
            "apply_enhanced_imputation_strategy_6step(\n    all_stocks_winsorized,",
        )
        source = source.replace(
            "apply_enhanced_imputation_strategy_6step(all_stocks,",
            "apply_enhanced_imputation_strategy_6step(all_stocks_winsorized,",
        )
        if "all_stocks_imputed" not in source:
            # If not already assigned to all_stocks_imputed, do it
            source = source.replace(
                "all_stocks = apply_enhanced_imputation_strategy_6step",
                "all_stocks_imputed = apply_enhanced_imputation_strategy_6step",
            )
            changes.append(f"Cell {cell_idx}: Added all_stocks_imputed after 6-step imputation")

    # 11. Scaling (look for scale_features)
    if "scale_features" in source and "all_stocks" in source:
        # Change: all_stocks = scale_features(all_stocks, ...)
        # to: all_stocks_scaled = scale_features(all_stocks_imputed, ...)
        if "all_stocks = scale_features" in source:
            source = source.replace(
                "scale_features(all_stocks", "scale_features(all_stocks_imputed"
            )
            source = source.replace(
                "all_stocks = scale_features", "all_stocks_scaled = scale_features"
            )
            changes.append(f"Cell {cell_idx}: Added all_stocks_scaled after scaling")

    # Update cell if changed
    if source != original_source:
        cell["source"] = source.split("\n")
        # Ensure each line ends with \n except the last
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Added stage-based DataFrame naming")
print(f"  Changes made: {len(changes)}")
for change in changes:
    print(f"  - {change}")
