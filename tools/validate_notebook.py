#!/usr/bin/env python3
"""Quick validation of notebook structure after inspection fixes."""
import json
import sys

try:
    with open("../ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"✓ Notebook JSON is valid")
    print(f"✓ Total cells: {len(nb['cells'])}")
    print(f"✓ Metadata present: {bool(nb.get('metadata'))}")

    # Count code vs markdown cells
    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"✓ Code cells: {code_cells}")
    print(f"✓ Markdown cells: {md_cells}")

    print("\n✓ All validation checks passed!")
    sys.exit(0)

except json.JSONDecodeError as e:
    print(f"✗ JSON parse error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Validation error: {e}")
    sys.exit(1)
