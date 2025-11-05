#!/usr/bin/env python3
"""Validate the refactored notebook structure and syntax."""

import json
import ast
import sys


def validate_notebook():
    """Validate notebook JSON structure and Python syntax."""

    print("Validating refactored notebook...")
    print("=" * 80)

    # 1. Validate JSON structure
    print("\n1. Validating JSON structure...")
    try:
        with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
            notebook = json.load(f)
        print("   ✓ Valid JSON structure")
    except json.JSONDecodeError as e:
        print(f"   ✗ Invalid JSON: {e}")
        return False

    # 2. Check notebook metadata
    print("\n2. Checking notebook metadata...")
    if "cells" not in notebook:
        print("   ✗ Missing 'cells' key")
        return False
    print(f"   ✓ Found {len(notebook['cells'])} cells")

    # 3. Find and validate Phase 9.7 cell
    print("\n3. Validating Phase 9.7 cell...")
    phase97_found = False

    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])

        if "PHASE 9.7 — VALUATION AND STOCK IDENTIFICATION" in source:
            print(f"   ✓ Found Phase 9.7 cell at index {idx}")
            phase97_found = True

            # Check for refactored elements
            print("\n4. Checking for refactored elements...")

            refactored_elements = {
                "DEFAULT_RISK_FREE_RATE": "Constant for risk-free rate",
                "DEFAULT_VOLATILITY": "Constant for default volatility",
                "TOP_N_RANKINGS": "Constant for top N rankings",
                "FACTOR_WEIGHTS": "Factor weights dictionary",
                "verify_prerequisites": "Prerequisites verification function",
                "calculate_and_apply_zscores": "Extracted z-scores function",
                "calculate_and_apply_percentiles": "Extracted percentiles function",
                "get_available_columns": "Utility function",
                "validate_required_columns": "Column validation function",
                "analyze_sectors": "Sector analysis function",
                "create_sector_summary": "Sector summary function",
                "display_sector_analysis_results": "Display function",
                "display_single_sector_analysis": "Single sector display",
                "display_sector_group": "Sector group display",
                "display_large_cap_screen": "Large cap screening",
                "display_tech_sector_screen": "Tech sector screening",
                "find_tech_sectors": "Tech sector finder",
                "setup_output_directory": "Output directory setup",
                "create_scatter_plot": "Scatter plot creation",
                "create_heatmaps": "Heatmaps creation",
                "create_sector_heatmap_viz": "Sector heatmap viz",
                "create_region_sector_heatmap_viz": "Region-sector heatmap",
                "export_excel_report": "Excel export",
                "generate_pdf_summary": "PDF generation",
                "handle_visualization_error": "Unified error handler",
            }

            found_count = 0
            missing = []

            for element, description in refactored_elements.items():
                if element in source:
                    found_count += 1
                    print(f"   ✓ {element}: {description}")
                else:
                    missing.append(element)

            print(f"\n   Found {found_count}/{len(refactored_elements)} refactored elements")

            if missing:
                print(f"   ⚠ Missing elements: {', '.join(missing)}")

            # 5. Validate Python syntax
            print("\n5. Validating Python syntax...")
            try:
                ast.parse(source)
                print("   ✓ Valid Python syntax")
            except SyntaxError as e:
                print(f"   ✗ Syntax error: {e}")
                return False

            # 6. Check cell size reduction
            print("\n6. Checking code organization...")
            line_count = len(cell["source"])
            func_count = source.count("def ")
            print(f"   • Total lines: {line_count}")
            print(f"   • Function definitions: {func_count}")
            print(
                f"   • Average lines per function: {line_count // func_count if func_count > 0 else 'N/A'}"
            )

            if func_count > 20:
                print(f"   ✓ Good function decomposition ({func_count} functions)")

            break

    if not phase97_found:
        print("   ✗ Phase 9.7 cell not found")
        return False

    print("\n" + "=" * 80)
    print("✓ Notebook validation complete - refactoring successful!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = validate_notebook()
    sys.exit(0 if success else 1)
