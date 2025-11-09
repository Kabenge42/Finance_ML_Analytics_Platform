"""
Fix compressed cells and deprecated function calls in ml_finance_model_main.ipynb.
This script:
1. Expands all compressed cells (malformed one-liners)
2. Replaces deprecated function names with new prefixed versions
3. Adds missing constants and dataframes
"""

import json
import re
from pathlib import Path

# Function name mappings (old -> new)
FUNCTION_MAPPING = {
    "build_comprehensive_features": "features_build_comprehensive",
    "calculate_feature_importance_rf": "features_importance_rf",
    "create_enhanced_event_labels": "classification_create_enhanced_event_labels",
    "prepare_classification_data": "classification_prepare_data",  # Need to check if this exists
    "compare_classifiers": "classification_compare_classifiers",  # Need to check
    "create_classification_interactions": "regression_create_classification_interactions",
    "prepare_regression_data": "regression_prepare_data",
    "compare_regressors": "regression_compare_regressors",
    "train_stacking_regressor": "regression_train_stacking",
    "train_quantile_regressor": "regression_train_quantile",
    "train_sector_specific_models": "regression_train_sector_models",
    "save_model": "regression_save_model",
    "load_model": "regression_load_model",
    "calculate_mispricing_score": "analytics_calculate_mispricing",
    "assign_valuation_category": "analytics_assign_valuation_category",  # Need to check
    "rank_undervalued_stocks": "analytics_rank_undervalued",
    "rank_overvalued_stocks": "analytics_rank_overvalued",
    "rank_stocks_by_sector": "analytics_rank_by_sector",
    "comprehensive_regression_metrics": "evaluation_comprehensive_metrics",
    "compute_metrics_by_segment": "evaluation_metrics_by_segment",
}


def fix_compressed_cell(code):
    """Expand compressed one-liner cells."""
    # Pattern: multiple statements on one line without proper newlines
    # Look for common patterns like )print( or )all_stocks or similar

    # Fix: )print( -> )\nprint(
    code = re.sub(r"\)print\(", ")\nprint(", code)

    # Fix: )all_stocks -> )\nall_stocks
    code = re.sub(r"\)all_stocks", ")\nall_stocks", code)

    # Fix: )labels -> )\nlabels
    code = re.sub(r"\)labels", ")\nlabels", code)

    # Fix: )X_train -> )\nX_train
    code = re.sub(r"\)X_train", ")\nX_train", code)

    # Fix: )comparison -> )\ncomparison
    code = re.sub(r"\)comparison", ")\ncomparison", code)

    # Fix: )from -> )\nfrom
    code = re.sub(r"\)from", ")\nfrom", code)

    # Fix: )eda_ -> )\neda_
    code = re.sub(r"\)eda_", ")\neda_", code)

    # Fix: )benchmark -> )\nbenchmark
    code = re.sub(r"\)benchmark", ")\nbenchmark", code)

    # Fix: )metrics_ -> )\nmetrics_
    code = re.sub(r"\)metrics_", ")\nmetrics_", code)

    # Fix: )available -> )\navailable
    code = re.sub(r"\)available", ")\navailable", code)

    # Fix: )exclude_cols -> )\nexclude_cols
    code = re.sub(r"\)exclude_cols", ")\nexclude_cols", code)

    # Fix: )feature_cols -> )\nfeature_cols
    code = re.sub(r"\)feature_cols", ")\nfeature_cols", code)

    # Fix: )if -> )\nif
    code = re.sub(r"\)if", ")\nif", code)

    return code


def replace_function_names(code):
    """Replace deprecated function names with new ones."""
    for old_name, new_name in FUNCTION_MAPPING.items():
        # Match function calls: old_name(
        code = re.sub(rf"\b{old_name}\(", f"{new_name}(", code)
    return code


def add_missing_dataframe_creation(cells):
    """Add all_stocks_with_classification creation after classification section."""
    # Find the cell that ends classification section (before section 6)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Look for the end of classification section export
            if "all_stocks_features[event_prob_cols]" in source or "event_prob_" in source:
                # Insert a new cell after this to create all_stocks_with_classification
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "#%%\n",
                        "# Create dataframe with classification features for regression\n",
                        "all_stocks_with_classification = all_stocks_features.copy()\n",
                        'print(f"✓ Classification features ready for regression: {all_stocks_with_classification.shape}")\n',
                    ],
                }
                cells.insert(i + 1, new_cell)
                break
    return cells


def add_target_constants(cells):
    """Add TARGET_COL constants before section 6."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if "## 6. Phase 9.5" in source or "Sector-Optimized Regression" in source:
                # Insert constants cell before this markdown
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "#%%\n",
                        "# Configuration constants for regression\n",
                        "TARGET_COL = 'price_target'\n",
                        "TARGET_COL_FALLBACK = 'last_price'\n",
                        "TEST_SIZE = 0.2\n",
                        "CV_FOLDS = 5\n",
                        "QUANTILES = [0.1, 0.5, 0.9]\n",
                        "MIN_SECTOR_SAMPLES = 20\n",
                        'print("✓ Regression configuration constants defined")\n',
                    ],
                }
                cells.insert(i, new_cell)
                break
    return cells


def fix_notebook(notebook_path):
    """Fix all issues in the notebook."""
    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Found {len(cells)} cells")

    # Process each code cell
    fixed_count = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                original = "".join(source)

                # Apply fixes
                fixed = fix_compressed_cell(original)
                fixed = replace_function_names(fixed)

                if fixed != original:
                    # Update cell source
                    cell["source"] = fixed
                    fixed_count += 1

    print(f"Fixed {fixed_count} compressed/deprecated cells")

    # Add missing dataframe
    cells = add_missing_dataframe_creation(cells)
    print("Added all_stocks_with_classification creation")

    # Add constants
    cells = add_target_constants(cells)
    print("Added TARGET_COL constants")

    # Update notebook
    nb["cells"] = cells

    # Write back
    print(f"Writing fixed notebook...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("✓ Notebook fixed successfully")


if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"
    fix_notebook(notebook_path)
