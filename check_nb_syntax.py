#!/usr/bin/env python
"""Check notebook cells for Python syntax errors."""
import json
import ast


def check_notebook_syntax(notebook_path):
    """Check all code cells in a notebook for syntax errors."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    errors = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        # Skip magic commands and shell commands
        lines = source.strip().split("\n")
        if all(line.strip().startswith(("%", "!", "#")) or not line.strip() for line in lines):
            continue

        try:
            ast.parse(source)
        except SyntaxError as e:
            errors.append(
                {
                    "cell_index": i,
                    "line": e.lineno,
                    "offset": e.offset,
                    "msg": e.msg,
                    "text": e.text,
                    "source_preview": source[:500] if len(source) > 500 else source,
                }
            )

    return errors


if __name__ == "__main__":
    errors = check_notebook_syntax("ml_finance_model_main2_0.ipynb")
    if errors:
        print(f"Found {len(errors)} syntax error(s):")
        for err in errors:
            print(f"\n--- Cell {err['cell_index']} (line {err['line']}) ---")
            print(f"Error: {err['msg']}")
            print(f"Text: {err['text']}")
            print(f"Source preview:\n{err['source_preview']}")
    else:
        print("No syntax errors found!")
