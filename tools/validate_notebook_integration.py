"""
Validate the notebook integration was successful.

Checks:
1. Notebook JSON is valid
2. Cell count increased correctly
3. New cells are at correct positions
4. Cell types are correct
5. Cell content includes expected markers
"""

import json
from pathlib import Path


def validate_notebook_integration():
    """Validate notebook integration."""

    print("\n" + "=" * 80)
    print("NOTEBOOK INTEGRATION VALIDATION")
    print("=" * 80 + "\n")

    notebook_path = Path("ml_finance_model_main_backup.ipynb")
    backup_path = Path("ml_finance_model_main_backup.ipynb.backup_before_integration")

    # Check files exist
    if not notebook_path.exists():
        print("❌ ERROR: Notebook file not found")
        return False

    if not backup_path.exists():
        print("⚠️  WARNING: Backup file not found")
    else:
        print("✓ Backup file exists")

    # Load notebook
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
        print("✓ Notebook JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON - {e}")
        return False

    # Check cell count
    cell_count = len(notebook["cells"])
    expected_count = 159  # 154 original + 5 new
    print(f"✓ Total cells: {cell_count}")

    if cell_count != expected_count:
        print(f"⚠️  WARNING: Expected {expected_count} cells, got {cell_count}")

    # Check notebook format
    nbformat = notebook.get("nbformat", "unknown")
    print(f"✓ Notebook format version: {nbformat}")

    # Validate integrated cells (should be at positions 140-144)
    print("\n" + "-" * 80)
    print("INTEGRATED CELLS (positions 140-144):")
    print("-" * 80 + "\n")

    expected_cells = [
        (140, "markdown", "Phase 9.5.1"),
        (141, "code", "MODEL OPTIMIZATION ENHANCEMENTS"),
        (142, "markdown", "Phase 9.6.1"),
        (143, "code", "ENHANCED ERROR ANALYSIS"),
        (144, "code", "SUMMARY AND VALIDATION"),
    ]

    all_valid = True

    for pos, expected_type, expected_marker in expected_cells:
        if pos >= len(notebook["cells"]):
            print(f"❌ Cell {pos}: OUT OF RANGE")
            all_valid = False
            continue

        cell = notebook["cells"][pos]
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))

        # Check type
        type_match = "✓" if cell_type == expected_type else "❌"

        # Check content
        content_match = "✓" if expected_marker in source else "❌"

        # Display
        preview = source[:80].replace("\n", " ") if source else "(empty)"
        print(f"Cell {pos}:")
        print(f"  Type: {type_match} {cell_type} (expected: {expected_type})")
        print(f"  Content: {content_match} Contains '{expected_marker}'")
        print(f"  Preview: {preview}...")
        print()

        if cell_type != expected_type or expected_marker not in source:
            all_valid = False

    # Check surrounding cells to confirm positioning
    print("-" * 80)
    print("SURROUNDING CELLS:")
    print("-" * 80 + "\n")

    for i in [138, 139, 145, 146]:
        if i < len(notebook["cells"]):
            cell = notebook["cells"][i]
            source = "".join(cell.get("source", []))
            preview = source[:80].replace("\n", " ") if source else "(empty)"
            marker = ""
            if "Phase 9.5" in source and "Phase 9.5.1" not in source:
                marker = " ← PHASE 9.5 (original)"
            elif "Phase 9.6" in source and "Phase 9.6.1" not in source:
                marker = " ← PHASE 9.6 (original)"
            print(f"Cell {i}: {cell['cell_type']:8s} - {preview}...{marker}")

    # Final summary
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ VALIDATION PASSED - All cells integrated correctly")
    else:
        print("⚠️  VALIDATION WARNINGS - Some cells may need review")
    print("=" * 80 + "\n")

    return all_valid


def check_notebook_imports():
    """Check if key imports from finance_ml are present."""

    print("\n" + "=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80 + "\n")

    notebook_path = Path("ml_finance_model_main_backup.ipynb")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find cells with imports
    import_cells = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "from finance_ml" in source or "import finance_ml" in source:
                import_cells.append((i, source[:200]))

    print(f"Found {len(import_cells)} cells with finance_ml imports\n")

    # Check for key imports
    key_imports = [
        "from finance_ml.models import train_and_evaluate_regression",
        "from finance_ml.models import train_and_evaluate_regression_by_sector",
        "from finance_ml import __version__",
    ]

    all_notebook_source = "\n".join(
        [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
    )

    print("Key imports status:")
    for imp in key_imports:
        found = "✓" if imp in all_notebook_source else "❌"
        print(f"  {found} {imp}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    success = validate_notebook_integration()
    check_notebook_imports()

    if success:
        print("\n💡 Next steps:")
        print("   1. Open Jupyter: jupyter notebook ml_finance_model_main_backup.ipynb")
        print("   2. Navigate to cell 140 (Phase 9.5.1)")
        print("   3. Run cells 140-144 to test execution")
        print("   4. Verify outputs in outputs/models/")
        print("\n✓ Integration validation complete\n")
    else:
        print("\n⚠️  Please review the warnings above\n")
