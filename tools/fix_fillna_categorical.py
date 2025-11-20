"""
Fix categorical fillna issue in notebook.

Replaces blanket fillna(0) with type-aware fillna_by_dtype() function
to prevent "TypeError: Cannot setitem on a Categorical with a new category (0)"
"""

import json
import sys
from pathlib import Path


def fix_notebook():
    """Fix the fillna(0) issue in the notebook."""
    notebook_path = Path("ml_finance_model_main.ipynb")

    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        sys.exit(1)

    # Load notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # The problematic code to replace
    old_code = "    # Final cleanup - replace any remaining NaN with 0\n    all_stocks_enhanced = all_stocks_enhanced.fillna(0)"

    new_code = """    # Final cleanup - use type-aware filling to handle mixed dtypes safely
    # This prevents "TypeError: Cannot setitem on a Categorical with a new category (0)"
    # by filling numeric/categorical/string columns with appropriate values
    all_stocks_enhanced = fillna_by_dtype(
        all_stocks_enhanced,
        numeric_fill=0,
        categorical_strategy="mode",
        string_fill="Unknown"
    )"""

    cells_modified = 0

    # Iterate through cells
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # Join source lines into a single string
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source

            # Check if this cell contains the problematic code
            if old_code in source_text:
                # Replace the problematic code
                new_source_text = source_text.replace(old_code, new_code)

                # Split back into lines for notebook format
                if isinstance(source, list):
                    cell["source"] = new_source_text.splitlines(keepends=True)
                else:
                    cell["source"] = new_source_text

                cells_modified += 1
                print(f"Fixed cell with fillna(0) issue (modification #{cells_modified})")

    if cells_modified == 0:
        print("Warning: No cells were modified. The problematic code may have already been fixed.")
        return False

    # Save the modified notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Successfully fixed {cells_modified} cell(s) in {notebook_path}")
    print("  Replaced blanket fillna(0) with type-aware fillna_by_dtype()")
    return True


if __name__ == "__main__":
    success = fix_notebook()
    sys.exit(0 if success else 1)
