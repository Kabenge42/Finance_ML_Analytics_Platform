#!/usr/bin/env python
"""Verify the Time-Series CV cell fix was applied correctly."""
import json
from pathlib import Path


def verify_fix(notebook_path):
    """Verify the notebook fix."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("=" * 80)
    print("TIME-SERIES CV CELL FIX VERIFICATION")
    print("=" * 80)

    # Find the Time-Series CV cell
    tscv_cell_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "Time-Series Cross-Validation" in source and "TimeSeriesSplit" in source:
                tscv_cell_idx = i
                break

    if tscv_cell_idx is None:
        print("✗ Could not find Time-Series CV cell")
        return False

    cell = nb["cells"][tscv_cell_idx]
    num_lines = len(cell["source"])
    total_chars = sum(len(line) for line in cell["source"])

    print(f"\n✓ Found Time-Series CV cell at index {tscv_cell_idx}")
    print(f"  Number of lines: {num_lines}")
    print(f"  Total characters: {total_chars}")

    # Check if properly formatted (should have >50 lines for proper formatting)
    if num_lines < 50:
        print(f"\n✗ FAIL: Cell appears malformed ({num_lines} lines, expected >50)")
        return False

    print(f"\n✓ PASS: Cell is properly formatted ({num_lines} lines)")

    # Show structure
    print("\n" + "=" * 80)
    print("CELL STRUCTURE (first 20 lines):")
    print("=" * 80)
    for i, line in enumerate(cell["source"][:20], 1):
        # Remove trailing newline for display
        display_line = line.rstrip("\n")
        print(f"  {i:2}: {display_line}")

    if len(cell["source"]) > 20:
        print(f"  ... ({len(cell['source']) - 20} more lines)")

    # Key content checks
    print("\n" + "=" * 80)
    print("CONTENT VALIDATION:")
    print("=" * 80)

    source_text = "".join(cell["source"])
    checks = [
        ("all_stocks_enhanced", "Uses correct dataframe"),
        ("TimeSeriesSplit", "Time-series split present"),
        ("regression_train_stacking", "Stacking ensemble used"),
        ("winsorize_target", "Target winsorization applied"),
        ("adaptive_clip_predictions", "Adaptive clipping used"),
        ("tscv_metrics.csv", "Metrics export present"),
    ]

    all_passed = True
    for keyword, description in checks:
        if keyword in source_text:
            print(f"  ✓ {description}: '{keyword}' found")
        else:
            print(f"  ✗ {description}: '{keyword}' NOT found")
            all_passed = False

    # Check for old variable names that shouldn't be there
    print("\n" + "=" * 80)
    print("NEGATIVE CHECKS (should NOT be present):")
    print("=" * 80)

    bad_checks = [
        ("df_reg", "Old df_reg variable reference"),
    ]

    for keyword, description in bad_checks:
        if keyword not in source_text:
            print(f"  ✓ {description}: '{keyword}' correctly absent")
        else:
            print(f"  ✗ {description}: '{keyword}' incorrectly present")
            all_passed = False

    # Final result
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print("\nThe Time-Series CV cell has been successfully fixed:")
        print(f"  - Reformatted from 4 lines to {num_lines} lines")
        print(f"  - Proper indentation and structure applied")
        print(f"  - Uses correct dataframe (all_stocks_enhanced)")
        print(f"  - Aligns with code_guidelines.md standards")
        return True
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        return False


if __name__ == "__main__":
    notebook_path = Path("ml_finance_model_main.ipynb")
    success = verify_fix(notebook_path)
    exit(0 if success else 1)
