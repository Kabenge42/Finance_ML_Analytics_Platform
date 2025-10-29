#!/usr/bin/env python3
"""Verify that notebook fixes were applied correctly."""

import json
from pathlib import Path


def verify_notebook_fixes():
    """Check if all fixes were properly applied."""
    notebook_path = Path("ml_finance_model_main.ipynb")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook["cells"]
    print(f"Total cells: {len(cells)}\n")

    # Check each fix
    fixes_status = {}

    # Fix #2: create_sample_financial_dataset import
    found_import = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "create_sample_financial_dataset" in source:
                found_import = True
                fixes_status["Fix #2 - Missing import"] = f"✓ Found in cell {i}"
                break
    if not found_import:
        fixes_status["Fix #2 - Missing import"] = "✗ Not found"

    # Fix #3: AttributeError handling
    found_attr_error = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "except AttributeError as e:" in source and "simple_eda" in source:
                found_attr_error = True
                fixes_status["Fix #3 - AttributeError handling"] = f"✓ Found in cell {i}"
                break
    if not found_attr_error:
        fixes_status["Fix #3 - AttributeError handling"] = "✗ Not found"

    # Fix #4: Config immutability comment
    found_config = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "Configuration immutability" in source:
                found_config = True
                fixes_status["Fix #4 - Config immutability"] = f"✓ Found in cell {i}"
                break
    if not found_config:
        fixes_status["Fix #4 - Config immutability"] = "✗ Not found"

    # Fix #5: Deprecation notices
    found_deprecation = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "DEPRECATED: Use cfg.attribute_name" in source:
                found_deprecation = True
                fixes_status["Fix #5 - Deprecation notices"] = f"✓ Found in cell {i}"
                break
    if not found_deprecation:
        fixes_status["Fix #5 - Deprecation notices"] = "✗ Not found"

    # Fix #8: Path import comment
    found_path_comment = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "Path already imported at top of notebook" in source:
                found_path_comment = True
                fixes_status["Fix #8 - Path import comment"] = f"✓ Found in cell {i}"
                break
    if not found_path_comment:
        fixes_status["Fix #8 - Path import comment"] = "✗ Not found"

    # Fix #9: Logger name
    found_logger = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "logger = logging.getLogger('finance_ml_notebook')" in source:
                found_logger = True
                fixes_status["Fix #9 - Logger name"] = f"✓ Found in cell {i}"
                break
    if not found_logger:
        fixes_status["Fix #9 - Logger name"] = "✗ Not found"

    # Fix #13-15: Utility functions
    found_utilities = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "print_section_header" in source and "_CHECKPOINTS" in source:
                found_utilities = True
                fixes_status["Fix #13-15 - Utility functions"] = f"✓ Found in cell {i}"
                break
    if not found_utilities:
        fixes_status["Fix #13-15 - Utility functions"] = "✗ Not found"

    # Fix #15: Checkpoint calls
    found_checkpoint = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if 'checkpoint("config_loaded")' in source or 'checkpoint("data_loaded"' in source:
                found_checkpoint = True
                fixes_status["Fix #15 - Checkpoint calls"] = f"✓ Found in cell {i}"
                break
    if not found_checkpoint:
        fixes_status["Fix #15 - Checkpoint calls"] = "✗ Not found"

    # Print results
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    for fix, status in fixes_status.items():
        print(f"{fix}: {status}")

    # Count successes
    successful = sum(1 for s in fixes_status.values() if "✓" in s)
    total = len(fixes_status)
    print(f"\n{successful}/{total} fixes verified")

    return successful == total


if __name__ == "__main__":
    success = verify_notebook_fixes()
    exit(0 if success else 1)
