"""Quick validation script for notebook structure."""

import json

try:
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("✓ Notebook is valid JSON")
    print(f"✓ Total cells: {len(data.get('cells', []))}")
    print(
        f"✓ Notebook format version: {data.get('nbformat', 'unknown')}.{data.get('nbformat_minor', 'unknown')}"
    )
    print("\n✓ All validation checks passed!")
except json.JSONDecodeError as e:
    print(f"✗ JSON decode error: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Validation error: {e}")
    exit(1)
