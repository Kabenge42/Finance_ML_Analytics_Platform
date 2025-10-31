#!/usr/bin/env python3
"""
Refactor notebook to remove inline function definitions and use only finance_ml imports.

This script:
1. Loads the notebook JSON
2. Identifies and removes cells with inline function definitions
3. Keeps only import cells and usage/demonstration cells
4. Creates a clean notebook that relies on the finance_ml package
"""
import json
import sys
from pathlib import Path


def is_function_definition_cell(cell):
    """Check if a cell contains function definitions that should be removed."""
    if cell["cell_type"] != "code":
        return False

    source = "".join(cell.get("source", []))

    # List of function names that are now in finance_ml package
    finance_ml_functions = [
        "def check_missing_values",
        "def detect_outliers_iqr",
        "def validate_numeric_ranges",
        "def validate_schema",
        "def engineer_basic_ratios",
        "def engineer_margin_features",
        "def engineer_volatility_features",
        "def engineer_revenue_cagr",
        "def build_features_and_target",
        "def create_event_labels",
        "def train_event_classifier",
        "def build_regression_pipeline",
        "def train_and_evaluate_regression",
        "def train_and_evaluate_regression_by_sector",
        "def train_quantile_regression",
        "def predict_quantile_regression",
        "def train_quantile_regression_by_sector",
        "def train_stacking_ensemble",
        "def train_stacking_ensemble_by_sector",
        "def simple_eda",
        "def calculate_mispricing_score",
        "def rank_undervalued_stocks",
        "def rank_overvalued_stocks",
        "def rank_stocks_by_sector",
        "def export_predictions_to_excel",
        "def create_sector_heatmap",
        "def create_interactive_prediction_plot",
        "def create_region_sector_heatmap",
        "def normalize_columns",
        "def infer_region_from_filename",
        "def load_from_csv",
        "def load_from_db",
        "def preprocess",
        "def setup_logging",
        "def get_env",
    ]

    # Check if cell contains any of these function definitions
    for func in finance_ml_functions:
        if func in source:
            return True

    return False


def is_legacy_import_cell(cell):
    """Check if cell contains legacy imports that are no longer needed."""
    if cell["cell_type"] != "code":
        return False

    source = "".join(cell.get("source", []))

    # Patterns that indicate legacy code to remove
    legacy_patterns = [
        "from finance_prediction.utils.logging import get_logger",
        "from finance_prediction.core.exceptions import",
        "def handle_optional_dependency",
        "def memory_monitor",
        "def cleanup_resources",
        "AVAILABLE_MODULES = {}",
        "tracemalloc.start()",
    ]

    for pattern in legacy_patterns:
        if pattern in source:
            return True

    return False


def should_keep_cell(cell, index):
    """Determine if a cell should be kept in the refactored notebook."""
    # Always keep markdown cells
    if cell["cell_type"] == "markdown":
        return True

    # Keep the first few cells (header and main imports)
    if index < 3:
        return True

    # Remove function definition cells
    if is_function_definition_cell(cell):
        return False

    # Remove legacy import cells
    if is_legacy_import_cell(cell):
        return False

    # Keep everything else (usage demonstrations)
    return True


def refactor_notebook(input_path, output_path):
    """Refactor the notebook to use only finance_ml imports."""
    print(f"Loading notebook from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    original_cell_count = len(nb["cells"])
    print(f"Original cell count: {original_cell_count}")

    # Filter cells
    kept_cells = []
    removed_count = 0

    for i, cell in enumerate(nb["cells"]):
        if should_keep_cell(cell, i):
            kept_cells.append(cell)
        else:
            removed_count += 1
            if cell["cell_type"] == "code":
                source_preview = "".join(cell.get("source", []))[:60]
                print(f"  Removing cell {i}: {source_preview}...")

    nb["cells"] = kept_cells

    print(f"Removed {removed_count} cells")
    print(f"Final cell count: {len(nb['cells'])}")

    # Save refactored notebook
    print(f"Saving refactored notebook to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Refactoring complete!")
    return len(nb["cells"]), removed_count


def main():
    notebook_path = Path("ml_finance_model_main.ipynb")
    backup_path = Path("ml_finance_model_v8_2.ipynb.bak_refactor")

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1

    # Create backup
    print(f"Creating backup at: {backup_path}")
    import shutil

    shutil.copy2(notebook_path, backup_path)

    # Refactor
    final_count, removed_count = refactor_notebook(notebook_path, notebook_path)

    print(f"\nSummary:")
    print(f"  Backup saved: {backup_path}")
    print(f"  Cells removed: {removed_count}")
    print(f"  Final cell count: {final_count}")
    print(f"  Refactored notebook: {notebook_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
