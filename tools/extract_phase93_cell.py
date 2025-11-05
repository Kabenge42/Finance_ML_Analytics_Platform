#!/usr/bin/env python3
"""Extract Phase 9.3 cell from notebook for analysis."""

import json


def extract_phase93_cell():
    """Extract and analyze Phase 9.3 cell content."""
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Cell 85 should be Phase 9.3
    cell = notebook["cells"][85]
    source = "".join(cell["source"])

    print(f"Cell 85 Analysis:")
    print(f"  Type: {cell['cell_type']}")
    print(f"  Lines: {len(cell['source'])}")
    print(f"  Characters: {len(source)}")
    print(f"\nFirst 500 characters:")
    print(source[:500])
    print(f"\n... [content truncated] ...\n")
    print(f"Last 500 characters:")
    print(source[-500:])

    # Save full content to file
    with open("phase93_cell_content.txt", "w", encoding="utf-8") as f:
        f.write(source)

    print(f"\n✓ Full content saved to phase93_cell_content.txt")

    # Analyze structure
    lines = source.split("\n")
    print(f"\n📊 Structure Analysis:")
    print(f"  Total lines: {len(lines)}")

    # Count functions
    function_defs = [line for line in lines if line.strip().startswith("def ")]
    print(f"  Function definitions: {len(function_defs)}")

    # Count imports
    import_lines = [line for line in lines if "import " in line]
    print(f"  Import statements: {len(import_lines)}")

    # Count if/elif statements (measure complexity)
    if_statements = [line for line in lines if line.strip().startswith(("if ", "elif "))]
    print(f"  If/elif statements: {len(if_statements)}")

    # Count for loops
    for_loops = [line for line in lines if line.strip().startswith("for ")]
    print(f"  For loops: {len(for_loops)}")

    # Identify long functions
    print(f"\n📝 Function definitions found:")
    for func_def in function_defs[:15]:  # Show first 15
        print(f"    {func_def.strip()}")
    if len(function_defs) > 15:
        print(f"    ... and {len(function_defs) - 15} more")

    return source


if __name__ == "__main__":
    extract_phase93_cell()
