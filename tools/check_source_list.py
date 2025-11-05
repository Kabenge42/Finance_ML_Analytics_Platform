#!/usr/bin/env python3
"""Check source list structure."""

import json


def main():
    notebook_path = "ml_finance_model_main.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    source_list = notebook["cells"][149]["source"]

    print(f"Source is a list with {len(source_list)} elements")
    print(f"\nFirst 5 elements:")
    for i in range(min(5, len(source_list))):
        elem = source_list[i]
        print(f"{i}: {repr(elem[:80] if len(elem) > 80 else elem)}")


if __name__ == "__main__":
    main()
