#!/usr/bin/env python3
"""
Fix imports in ml_finance_model_main2_0.ipynb according to Option 1 from the recommendation.
"""
import json
import sys
from pathlib import Path


def fix_notebook_imports(notebook_path: Path) -> None:
    """Fix the problematic import in the notebook."""

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    fixes_applied = []

    # Iterate through cells
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

        # Fix 1: Replace train_sector_optimized_regressors with train_high_error_sector_models
        if "train_sector_optimized_regressors" in source:
            print(f"Found issue in cell {i} (id: {cell.get('id', 'unknown')})")

            # Update the source
            new_source = source.replace(
                "train_sector_optimized_regressors", "train_high_error_sector_models"
            )

            # Convert back to list format if needed
            if isinstance(cell["source"], list):
                cell["source"] = new_source.split("\n")
                # Add newlines back
                cell["source"] = [
                    line + "\n" if i < len(cell["source"]) - 1 else line
                    for i, line in enumerate(cell["source"])
                ]
            else:
                cell["source"] = new_source

            fixes_applied.append(
                f"Cell {i}: Fixed train_sector_optimized_regressors -> train_high_error_sector_models"
            )

    if not fixes_applied:
        print("No issues found to fix.")
        return

    # Save the fixed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[OK] Fixed {len(fixes_applied)} issue(s):")
    for fix in fixes_applied:
        print(f"  - {fix}")
    print(f"\n[OK] Saved to: {notebook_path}")


if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "ml_finance_model_main2_0.ipynb"

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    print(f"Fixing imports in: {notebook_path}")
    fix_notebook_imports(notebook_path)
    print("\n[OK] Done!")
