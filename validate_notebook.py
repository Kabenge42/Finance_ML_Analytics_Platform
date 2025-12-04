#!/usr/bin/env python
"""Validate notebook JSON structure."""
import json
import sys

try:
    with open("etl_data_explorer.ipynb", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("✓ Notebook is valid JSON")
    print(f'Total cells: {len(data["cells"])}')
    print(
        f'Notebook format: {data.get("nbformat", "unknown")}.{data.get("nbformat_minor", "unknown")}'
    )

    # Count cell types
    markdown_cells = sum(1 for c in data["cells"] if c["cell_type"] == "markdown")
    code_cells = sum(1 for c in data["cells"] if c["cell_type"] == "code")

    print(f"Markdown cells: {markdown_cells}")
    print(f"Code cells: {code_cells}")

    # Check for new cells
    cell_markers = []
    for i, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            if "Cell 10.5" in source:
                cell_markers.append(f"Cell 10.5 found at index {i}")
            if "Cell 10.6" in source:
                cell_markers.append(f"Cell 10.6 found at index {i}")

    if cell_markers:
        print("\n✓ New cells added:")
        for marker in cell_markers:
            print(f"  {marker}")

    print("\n✓ Notebook structure validated successfully")
    sys.exit(0)

except json.JSONDecodeError as e:
    print(f"✗ JSON validation error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
