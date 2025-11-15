#!/usr/bin/env python3
"""Extract Phase 9.5 cell from notebook for analysis."""

import json


def extract_phase95_cell():
    """Extract and analyze Phase 9.5 cell content."""
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Cell 144 should be Phase 9.5
    cell = notebook["cells"][144]
    source = "".join(cell["source"])

    print(f"Cell 144 Analysis:")
    print(f"  Type: {cell['cell_type']}")
    print(f"  Lines: {len(cell['source'])}")
    print(f"  Characters: {len(source)}")
    print(f"\nFirst 500 characters:")
    print(source[:500])
    print(f"\n... [content truncated] ...\n")
    print(f"Last 500 characters:")
    print(source[-500:])

    # Save full content to file
    with open("phase95_cell_content.txt", "w", encoding="utf-8") as f:
        f.write(source)

    print(f"\n✓ Full content saved to phase95_cell_content.txt")

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

    # Identify long functions
    print(f"\n📝 Function definitions found:")
    for func_def in function_defs[:10]:  # Show first 10
        print(f"    {func_def.strip()}")
    if len(function_defs) > 10:
        print(f"    ... and {len(function_defs) - 10} more")

    return source


if __name__ == "__main__":
    extract_phase95_cell()
