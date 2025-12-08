#!/usr/bin/env python3
"""Analyze notebook for inspection issues."""
import json
import re

# Load notebook
with open("ml_finance_model_main2_0.ipynb", encoding="utf-8") as f:
    nb = json.load(f)


# Map line numbers to cell indices (approximate)
def find_cell_by_line(cells, target_line):
    """Find cell index by cumulative line number."""
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
                return idx, cumulative - num_lines, num_lines
    return None, 0, 0


# Issue tracking
issues = {
    "type_hints": [1662, 2016, 2821, 5302, 7803, 8648, 8649, 8650, 8659, 8660, 8689, 8690],
    "missing_docstrings": [2277, 2325],
    "default_args": [187],
    "shap_import": [6063],
}

print("=" * 80)
print("NOTEBOOK ISSUE ANALYSIS")
print("=" * 80)
print(f"Total cells: {len(nb['cells'])}")
print(f"Code cells: {len([c for c in nb['cells'] if c['cell_type'] == 'code'])}")
print()

# Find cells with issues
for issue_type, lines in issues.items():
    print(f"\n{issue_type.upper().replace('_', ' ')}:")
    print("-" * 40)
    for line in lines:
        cell_idx, start_line, num_lines = find_cell_by_line(nb["cells"], line)
        if cell_idx is not None:
            print(
                f"  Line {line} -> Cell #{cell_idx} (lines {start_line}-{start_line+num_lines-1})"
            )
            # Show first few lines of cell
            cell = nb["cells"][cell_idx]
            if cell["cell_type"] == "code":
                source = (
                    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                )
                preview = source[:200].replace("\n", " ")
                print(f"    Preview: {preview}...")
        else:
            print(f"  Line {line} -> Cell not found")
    print()

# Search for specific patterns
print("\nSPECIFIC PATTERN SEARCHES:")
print("-" * 40)

patterns = {
    "shap import": r"import\s+shap",
    "typing.List[": r"typing\.List\[",
    "typing.Dict[": r"typing\.Dict\[",
    "List[str]": r"List\[str\]",
    "Dict[str": r"Dict\[str",
}

for name, pattern in patterns.items():
    count = 0
    found_cells = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if re.search(pattern, source):
                count += 1
                found_cells.append(idx)
    print(f"  {name}: {count} occurrences in cells {found_cells}")
print()
