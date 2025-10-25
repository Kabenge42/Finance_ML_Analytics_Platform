"""
Update ml_finance_model_v8_3.ipynb to import from finance_ml package.

This script updates the notebook to use the modular finance_ml package
instead of defining functions inline.
"""

import json
import sys
from pathlib import Path


def update_notebook_imports(notebook_path: Path) -> bool:
    """Update notebook to use finance_ml imports."""
    print(f"Reading notebook: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # New import cell content
    new_imports = """# Finance ML Analytics Platform — Notebook (v0.3.0)
# This notebook now uses the modular finance_ml package

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import all functions from finance_ml package
from finance_ml import (
    # Version
    __version__,
    # Configuration
    FinanceMLConfig,
    load_config,
    get_config,
    # Utilities
    setup_logging,
    get_env,
    # Data loading and validation
    normalize_columns,
    infer_region_from_filename,
    load_from_csv,
    load_from_db,
    preprocess,
    validate_schema,
    check_missing_values,
    detect_outliers_iqr,
    validate_numeric_ranges,
    # Feature engineering
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target,
    # Modeling
    create_event_labels,
    train_event_classifier,
    build_regression_pipeline,
    train_and_evaluate_regression,
    train_and_evaluate_regression_by_sector,
    train_quantile_regression,
    predict_quantile_regression,
    train_quantile_regression_by_sector,
    train_stacking_ensemble,
    train_stacking_ensemble_by_sector,
    # Evaluation and analytics
    simple_eda,
    calculate_mispricing_score,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
    export_predictions_to_excel,
    create_sector_heatmap,
    create_interactive_prediction_plot,
    create_region_sector_heatmap,
)

# Setup logging
setup_logging()

print(f"Finance ML Analytics Platform v{__version__}")
print("All functions imported from finance_ml package")
"""

    # Configuration cell
    config_cell = """# Configuration
# Load configuration from environment or create default
config = get_config()

# Display configuration
print("Configuration:")
print(f"  Data directory: {config.data_dir}")
print(f"  Output directory: {config.output_dir}")
print(f"  Model version: {config.model_version}")
print(f"  Random seed: {config.random_seed}")
print(f"  N jobs: {config.n_jobs}")
print(f"  DB URL: {'configured' if config.db_url else 'not configured'}")
"""

    # Find the first code cell (likely imports)
    first_code_idx = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            first_code_idx = i
            break

    if first_code_idx is None:
        print("No code cells found in notebook")
        return False

    # Create new import cell
    import_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_imports.split("\n"),
    }

    # Create config cell
    config_cell_obj = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_cell.split("\n"),
    }

    # Create a markdown cell explaining the change
    explanation_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Finance ML Analytics Platform\n",
            "\n",
            "**Version 0.3.0** — Now using modular `finance_ml` package\n",
            "\n",
            "## What's New\n",
            "\n",
            "- All functions are now imported from the `finance_ml` package\n",
            "- No need to define functions inline — they're maintained in the package modules\n",
            "- Configuration management with `FinanceMLConfig`\n",
            "- Better code organization and testability\n",
            "\n",
            "## Modules\n",
            "\n",
            "- `finance_ml.data`: Data loading, normalization, validation\n",
            "- `finance_ml.features`: Feature engineering\n",
            "- `finance_ml.models`: Classification, regression, ensembles\n",
            "- `finance_ml.eval`: Analytics, visualizations, reporting\n",
            "- `finance_ml.config`: Configuration management\n",
            "- `finance_ml.cli`: Command-line interface\n",
            "\n",
            "## Usage\n",
            "\n",
            "This notebook demonstrates the ML workflow:\n",
            "1. Load and validate data\n",
            "2. Exploratory data analysis\n",
            "3. Feature engineering\n",
            "4. Model training (classification and regression)\n",
            "5. Evaluation and analytics\n",
        ],
    }

    # Insert cells at the beginning
    notebook["cells"] = [explanation_cell, import_cell, config_cell_obj] + notebook["cells"][
        first_code_idx:
    ]

    # Save updated notebook
    backup_path = notebook_path.with_suffix(".ipynb.bak")
    print(f"Creating backup: {backup_path}")
    notebook_path.rename(backup_path)

    print(f"Writing updated notebook: {notebook_path}")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("Notebook updated successfully!")
    print(f"Backup saved to: {backup_path}")
    return True


if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_v8_3.ipynb")

    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)

    success = update_notebook_imports(notebook_path)
    sys.exit(0 if success else 1)
