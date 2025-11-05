#!/usr/bin/env python3
"""Verify that all imports needed by Phase 9.7 are available."""

import json
import re


def verify_imports():
    """Check if all required imports for Phase 9.7 are in the notebook."""
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Functions used in Phase 9.7
    required_functions = [
        "calculate_mispricing_score",
        "calculate_risk_adjusted_mispricing",
        "assign_valuation_category",
        "calculate_sector_zscores",
        "calculate_percentile_ranks",
        "calculate_multi_factor_score",
        "rank_undervalued_stocks",
        "rank_overvalued_stocks",
        "filter_stocks_by_criteria",
        "create_valuation_scatter_plot",
        "create_sector_heatmap",
        "create_region_sector_heatmap",
        "export_predictions_to_excel",
        "generate_pdf_report",
        "print_section_header",
    ]

    # Check import cells
    imports_found = set()

    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])

        # Look for import statements
        if "import" in source and idx < 50:  # Early cells are typically imports
            # Check for finance_ml imports
            if "from finance_ml" in source or "import finance_ml" in source:
                print(f"Cell {idx}: Found finance_ml imports")
                # Extract imported functions
                for func in required_functions:
                    if func in source:
                        imports_found.add(func)
                        print(f"  ✓ {func}")

    print("\n" + "=" * 80)
    print("Import verification summary:")
    print("=" * 80)

    missing_imports = set(required_functions) - imports_found

    if missing_imports:
        print(f"\n⚠ Missing imports ({len(missing_imports)}):")
        for func in sorted(missing_imports):
            print(f"  - {func}")
    else:
        print("\n✓ All required functions are imported")

    print(f"\nTotal required: {len(required_functions)}")
    print(f"Found: {len(imports_found)}")
    print(f"Missing: {len(missing_imports)}")

    return len(missing_imports) == 0


if __name__ == "__main__":
    success = verify_imports()
    exit(0 if success else 1)
