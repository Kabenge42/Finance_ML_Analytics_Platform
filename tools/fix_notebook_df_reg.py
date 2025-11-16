#!/usr/bin/env python
"""
Fix notebook df_reg NameError issue by:
1. Removing duplicate section 6.1 and 6.2 cells
2. Replacing df_reg with all_stocks_enhanced in section 6.5.1
3. Adding validation guard
"""
import json
import sys
from pathlib import Path


def analyze_notebook(notebook_path):
    """Analyze notebook to find duplicate sections and df_reg references."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # Find cells with section headers
    section_cells = []
    df_reg_cells = []

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])

            # Find section headers
            if 'print("6.1 — Creating Classification Interaction Features")' in source:
                section_cells.append((i, "6.1", source[:200]))
                print(f"\nCell {i}: Section 6.1")
            elif 'print("6.2 — Preparing Regression Data")' in source:
                section_cells.append((i, "6.2", source[:200]))
                print(f"\nCell {i}: Section 6.2")

            # Find df_reg references
            if "df_reg" in source:
                df_reg_cells.append((i, source.count("df_reg")))

    print(f"\n\nFound {len(section_cells)} section header cells")
    print(f"Found {len(df_reg_cells)} cells with df_reg references")

    for i, count in df_reg_cells:
        print(f"  Cell {i}: {count} df_reg references")

    return nb, section_cells, df_reg_cells


def fix_notebook(notebook_path, output_path):
    """Fix the notebook by removing duplicates and fixing df_reg references."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells_to_remove = []
    cells_to_modify = []

    # Find duplicate cells
    section_61_cells = []
    section_62_cells = []

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])

            if 'print("6.1 — Creating Classification Interaction Features")' in source:
                section_61_cells.append(i)
            elif 'print("6.2 — Preparing Regression Data")' in source:
                section_62_cells.append(i)

    print(f"\nSection 6.1 cells: {section_61_cells}")
    print(f"Section 6.2 cells: {section_62_cells}")

    # Keep first occurrence, remove second
    if len(section_61_cells) > 1:
        for cell_idx in section_61_cells[1:]:
            cells_to_remove.append(cell_idx)
            print(f"Will remove duplicate 6.1 cell at index {cell_idx}")

    if len(section_62_cells) > 1:
        for cell_idx in section_62_cells[1:]:
            cells_to_remove.append(cell_idx)
            print(f"Will remove duplicate 6.2 cell at index {cell_idx}")

    # Remove cells in reverse order to maintain indices
    for cell_idx in sorted(cells_to_remove, reverse=True):
        print(f"Removing cell {cell_idx}")
        del nb["cells"][cell_idx]

    # Fix df_reg references in remaining cells
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])

            # Only fix df_reg in cells that don't create it
            # (i.e., cells after the duplicates are removed)
            if "df_reg" in source and "all_stocks_enhanced" not in source:
                # Check if this is in section 6.5.1 (Time-Series CV)
                if "Time-Series Cross-Validation" in source or "TimeSeriesSplit" in source:
                    print(f"\nFixing df_reg in cell {i} (Time-Series CV)")
                    # Replace df_reg with all_stocks_enhanced
                    new_source = source.replace("df_reg", "all_stocks_enhanced")
                    cell["source"] = new_source.split("\n")
                    if not cell["source"][-1]:
                        cell["source"] = cell["source"][:-1]

    # Add validation guard at the start of first 6.1 cell
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if 'print("6.1 — Creating Classification Interaction Features")' in source:
                print(f"\nAdding validation guard to cell {i}")
                lines = cell["source"]
                # Insert guard after the print statements
                guard_code = [
                    "\n",
                    "# Validation: Ensure all_stocks_with_classification exists\n",
                    "if 'all_stocks_with_classification' not in globals():\n",
                    "    raise RuntimeError(\n",
                    '        "all_stocks_with_classification is not defined. "\n',
                    '        "Make sure Phase 9.4 (Classification) has been executed before this cell."\n',
                    "    )\n",
                    "\n",
                ]
                # Find where to insert (after the print statements)
                insert_idx = 0
                for idx, line in enumerate(lines):
                    if 'print("=" * 80)' in line:
                        insert_idx = idx + 1
                        if insert_idx < len(lines) and '")' in lines[insert_idx]:
                            insert_idx += 1
                        break

                cell["source"] = lines[:insert_idx] + guard_code + lines[insert_idx:]
                break

    # Save fixed notebook
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Fixed notebook saved to: {output_path}")
    print(f"  Original cells: {len(nb['cells']) + len(cells_to_remove)}")
    print(f"  Fixed cells: {len(nb['cells'])}")
    print(f"  Removed: {len(cells_to_remove)} duplicate cells")


if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_main.ipynb")

    print("=" * 80)
    print("ANALYZING NOTEBOOK")
    print("=" * 80)
    analyze_notebook(notebook_path)

    print("\n" + "=" * 80)
    print("FIXING NOTEBOOK")
    print("=" * 80)
    fix_notebook(notebook_path, notebook_path)

    print("\n" + "=" * 80)
    print("FIX COMPLETE")
    print("=" * 80)
    print("\nThe notebook has been fixed:")
    print("  - Removed duplicate section 6.1 and 6.2 cells")
    print("  - Replaced df_reg with all_stocks_enhanced in Time-Series CV")
    print("  - Added validation guard to prevent similar issues")
    print("\nPlease run the notebook cells in order to verify the fix.")
