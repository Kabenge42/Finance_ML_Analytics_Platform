"""
Inspect notebook structure for ETL pipeline updates.
"""

import json
from pathlib import Path


def inspect_notebook(nb_path: Path):
    """Inspect notebook structure and print summary."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"\n{'='*80}")
    print(f"Notebook: {nb_path.name}")
    print(f"Total cells: {len(cells)}")
    print(f"{'='*80}\n")

    # Categorize cells
    markdown_cells = []
    code_cells = []

    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])

        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        # Get first line for preview
        first_line = source_text.split("\n")[0] if source_text else ""

        if cell_type == "markdown":
            markdown_cells.append((i, first_line[:80]))
        elif cell_type == "code":
            code_cells.append((i, first_line[:80]))

    print(f"Markdown cells: {len(markdown_cells)}")
    print(f"Code cells: {len(code_cells)}\n")

    # Print first 10 cells with preview
    print("First 10 cells:")
    for i in range(min(10, len(cells))):
        cell = cells[i]
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])

        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        first_line = source_text.split("\n")[0] if source_text else "(empty)"
        print(f"  {i:3d}. [{cell_type:8s}] {first_line[:70]}")

    # Look for ETL/data loading sections
    print("\n\nETL/Data Loading related cells:")
    for i, cell in enumerate(cells):
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source).lower()
        else:
            source_text = source.lower()

        if any(
            keyword in source_text
            for keyword in ["etl", "load_from", "preprocess", "imputation", "all_stocks"]
        ):
            first_line = source_text.split("\n")[0][:80]
            print(f"  {i:3d}. {first_line}")


if __name__ == "__main__":
    notebooks = [
        Path("ml_finance_model_main.ipynb"),
        Path("ml_finance_model_main2_0.ipynb"),
        Path("etl_data_explorer.ipynb"),
    ]

    for nb_path in notebooks:
        if nb_path.exists():
            inspect_notebook(nb_path)
        else:
            print(f"\nNotebook not found: {nb_path}")
