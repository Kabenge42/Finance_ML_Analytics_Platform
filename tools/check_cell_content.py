#!/usr/bin/env python3
"""Check what's actually in the Phase 9.7 cell."""

import json


def main():
    notebook_path = "ml_finance_model_main.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cell = notebook["cells"][149]
    source = "".join(cell["source"])

    print(f"Cell 149 total length: {len(source)} characters")
    print("\nFirst 1000 characters:")
    print(source[:1000])
    print("\n" + "=" * 80)

    # Search for function definitions
    print("\nSearching for 'def ' in source:")
    if "def " in source:
        idx = source.find("def ")
        print(f"First 'def ' found at index {idx}")
        print("Context around first def:")
        print(source[max(0, idx - 50) : idx + 150])
    else:
        print("No 'def ' found in source!")

    # Count lines
    lines = source.split("\n")
    print(f"\nTotal lines in source: {len(lines)}")

    # Find lines with 'def '
    def_lines = [(i, line) for i, line in enumerate(lines) if line.strip().startswith("def ")]
    print(f"Lines starting with 'def ': {len(def_lines)}")
    if def_lines:
        print("\nFirst 5 function definitions:")
        for i, (line_no, line) in enumerate(def_lines[:5]):
            print(f"  Line {line_no}: {line.strip()[:60]}")


if __name__ == "__main__":
    main()
