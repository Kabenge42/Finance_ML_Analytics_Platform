"""
Script to simplify Phase 9.8 in notebook by removing inline class definition.
The class is now in finance_ml.analyst_comparison package.
"""

import json
import sys
from pathlib import Path


def simplify_phase98_notebook(notebook_path: Path) -> None:
    """Remove inline PredictionAnalystAnalytics class from notebook."""

    # Load notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find Phase 9.8 cell
    modified = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

            # Check if this is the Phase 9.8 cell with the inline class
            if (
                "from finance_ml import PredictionAnalystAnalytics" in source
                and "class PredictionAnalystAnalytics:" in source
            ):
                print("Found Phase 9.8 cell with inline class definition")

                # Replace with simplified version
                new_source = [
                    "#%%\n",
                    "from finance_ml import PredictionAnalystAnalytics\n",
                    "\n",
                    'print_section_header("PHASE 9.8 — PREDICTION VS. ANALYST PRICE TARGET ANALYTICS")\n',
                    "\n",
                    "# Execute Phase 9.8\n",
                    "analytics = PredictionAnalystAnalytics(all_stocks_valued, config)\n",
                    "results = analytics.run_full_analysis()\n",
                ]

                cell["source"] = new_source
                modified = True
                print("✓ Replaced inline class with package import")
                break

    if not modified:
        print("⚠ Phase 9.8 cell not found or already simplified")
        return

    # Save modified notebook
    backup_path = notebook_path.with_suffix(".ipynb.backup_phase98")
    print(f"Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"Saving modified notebook: {notebook_path}")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print("\n✓ Notebook simplified successfully")
    print("  - Inline class definition removed")
    print("  - Package import retained")
    print("  - Execution code retained")


if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"

    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)

    simplify_phase98_notebook(notebook_path)
