"""
Extract sections 4-10 from notebook to analyze current visualization state
"""

import json
from pathlib import Path

notebook_path = Path(
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

sections = {}
current_section = None

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        source = "".join(cell.get("source", []))

        # Check for section headers
        if "## 4." in source or source.startswith("## 4 "):
            current_section = 4
            sections[4] = {"start": idx, "cells": []}
        elif "## 5." in source or source.startswith("## 5 "):
            current_section = 5
            sections[5] = {"start": idx, "cells": []}
        elif "## 6." in source or source.startswith("## 6 "):
            current_section = 6
            sections[6] = {"start": idx, "cells": []}
        elif "## 7." in source or source.startswith("## 7 "):
            current_section = 7
            sections[7] = {"start": idx, "cells": []}
        elif "## 8." in source or source.startswith("## 8 "):
            current_section = 8
            sections[8] = {"start": idx, "cells": []}
        elif "## 9." in source or source.startswith("## 9 "):
            current_section = 9
            sections[9] = {"start": idx, "cells": []}
        elif "## 10." in source or source.startswith("## 10 "):
            current_section = 10
            sections[10] = {"start": idx, "cells": []}
        elif "## 11." in source or "## Conclusion" in source:
            current_section = None

    if current_section is not None:
        sections[current_section]["cells"].append((idx, cell["cell_type"]))

# Print analysis
for sec_num in sorted(sections.keys()):
    sec_data = sections[sec_num]
    code_cells = [c for c in sec_data["cells"] if c[1] == "code"]
    markdown_cells = [c for c in sec_data["cells"] if c[1] == "markdown"]

    print(f"\n{'='*60}")
    print(f"SECTION {sec_num}")
    print(f"{'='*60}")
    print(f"Start cell: {sec_data['start']}")
    print(f"Total cells: {len(sec_data['cells'])}")
    print(f"Code cells: {len(code_cells)}")
    print(f"Markdown cells: {len(markdown_cells)}")

    # Check for visualization keywords in code cells
    viz_keywords = [
        "plot",
        "fig",
        "plt",
        "plotly",
        "px.",
        "go.",
        "sns.",
        "heatmap",
        "scatter",
        "chart",
    ]

    print(f"\nCode cells with visualizations:")
    for cell_idx, cell_type in code_cells:
        if cell_type == "code":
            cell = nb["cells"][cell_idx]
            source = "".join(cell.get("source", []))
            has_viz = any(keyword in source.lower() for keyword in viz_keywords)
            if has_viz:
                print(f"  Cell {cell_idx}: HAS VISUALIZATION")
                # Show first 100 chars
                preview = source[:100].replace("\n", " ")
                print(f"    Preview: {preview}...")

    # Count total viz cells
    viz_count = sum(
        1
        for cell_idx, cell_type in code_cells
        if any(
            kw in "".join(nb["cells"][cell_idx].get("source", [])).lower() for kw in viz_keywords
        )
    )
    print(f"\nTotal visualization cells: {viz_count}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for sec_num in sorted(sections.keys()):
    sec_data = sections[sec_num]
    code_cells = [c for c in sec_data["cells"] if c[1] == "code"]
    viz_count = sum(
        1
        for cell_idx, cell_type in code_cells
        if any(
            kw in "".join(nb["cells"][cell_idx].get("source", [])).lower()
            for kw in [
                "plot",
                "fig",
                "plt",
                "plotly",
                "px.",
                "go.",
                "sns.",
                "heatmap",
                "scatter",
                "chart",
            ]
        )
    )
    print(f"Section {sec_num}: {len(code_cells)} code cells, {viz_count} with visualizations")
