#!/usr/bin/env python3
"""
Fix outputs_dir configuration issue in ml_finance_model_main.ipynb.

Adds outputs_dir definition cell after Phase 9.2 Continuation markdown header
to resolve NameError: name 'outputs_dir' is not defined.
"""

import json
from pathlib import Path


def fix_outputs_dir_issue():
    """Insert outputs_dir definition cell into notebook."""

    notebook_path = Path("ml_finance_model_main.ipynb")

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the Phase 9.2 Continuation markdown cell
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "Phase 9.2 Continuation: Advanced Correlation & Outlier Analysis" in source:
                # Insert after this markdown cell
                insert_idx = i + 1
                print(f"Found Phase 9.2 Continuation header at cell {i}")
                break

    if insert_idx is None:
        print("ERROR: Could not find Phase 9.2 Continuation header")
        return False

    # Check if outputs_dir is already defined in the next cell
    next_cell = nb["cells"][insert_idx]
    next_cell_source = (
        "".join(next_cell["source"])
        if isinstance(next_cell["source"], list)
        else next_cell["source"]
    )
    if "outputs_dir" in next_cell_source and "outputs_dir = " in next_cell_source:
        print("outputs_dir is already defined in the next cell. No changes needed.")
        return False

    # Create new cell with outputs_dir definition
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define output directory\n",
            "from pathlib import Path\n",
            "\n",
            "# Create outputs directory structure\n",
            "outputs_dir = Path('outputs')\n",
            "outputs_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Create subdirectories for different phases\n",
            "(outputs_dir / 'enhanced_eda').mkdir(parents=True, exist_ok=True)\n",
            "(outputs_dir / 'processed').mkdir(parents=True, exist_ok=True)\n",
            "(outputs_dir / 'regression').mkdir(parents=True, exist_ok=True)\n",
            "(outputs_dir / 'analytics').mkdir(parents=True, exist_ok=True)\n",
            "\n",
            'print(f"✓ Output directory configured: {outputs_dir.absolute()}")',
        ],
    }

    # Insert the new cell
    nb["cells"].insert(insert_idx, new_cell)

    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Inserted outputs_dir definition cell at position {insert_idx}")
    print(f"  Total cells now: {len(nb['cells'])} (was {len(nb['cells']) - 1})")
    print("\nCell defines:")
    print("  - outputs_dir = Path('outputs')")
    print("  - Subdirectories: enhanced_eda, processed, regression, analytics")
    print("\nThis fixes NameError at lines 1130, 1142, 1154, 1179, 1209")

    return True


if __name__ == "__main__":
    success = fix_outputs_dir_issue()
    if success:
        print("\n✅ Fix applied successfully!")
    else:
        print("\n⚠ No changes made")
