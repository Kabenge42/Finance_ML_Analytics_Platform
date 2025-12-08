#!/usr/bin/env python3
"""Find cells with inspection issues."""
import json

# Load notebook
with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    nb = json.load(f)


# Map line numbers to cell indices
def find_cell_by_line(cells, target_line):
    cumulative = 0
    for idx, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            lines = cell["source"]
            if isinstance(lines, list):
                num_lines = len(lines)
            else:
                num_lines = lines.count("\n") + 1
            cumulative += num_lines
            if cumulative >= target_line:
                return idx
    return None


# Issue lines
issue_lines = {
    "type_hints": [1662, 2016, 2821, 5302, 7803, 8648],
    "missing_docstrings": [2277, 2325],
    "default_args": [187],
}

print("CELL MAPPINGS FOR ISSUES:\n")
for issue_type, lines in issue_lines.items():
    print(f"{issue_type.upper()}:")
    cells_found = set()
    for line in lines:
        cell_idx = find_cell_by_line(nb["cells"], line)
        if cell_idx is not None:
            cells_found.add(cell_idx)
            print(f"  Line {line} -> Cell {cell_idx}")
    print(f"  Unique cells to fix: {sorted(cells_found)}\n")
