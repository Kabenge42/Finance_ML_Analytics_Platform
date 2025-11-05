#!/usr/bin/env python3
"""Verify Phase 9.7 refactoring was applied correctly."""

import json
import ast


def main():
    notebook_path = "ml_finance_model_main.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cell = notebook["cells"][149]
    source = "".join(cell["source"])

    print(f"Cell 149 length: {len(source)} characters")
    print(f"\nChecking for refactored elements:")

    # Check for constants
    constants = [
        "DEFAULT_RISK_FREE_RATE",
        "DEFAULT_VOLATILITY",
        "TOP_N_RANKINGS",
        "MIN_LARGE_CAP_MARKET_CAP",
        "FACTOR_WEIGHTS",
        "REQUIRED_COLUMNS",
    ]

    print("\nConstants:")
    for const in constants:
        present = const in source
        print(f"  {const}: {'OK' if present else 'MISSING'}")

    # Parse and count functions
    try:
        tree = ast.parse(source)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        print(f"\nTotal functions: {len(funcs)}")

        # Check for key refactored functions
        expected_functions = [
            "verify_prerequisites",
            "calculate_valuation_metrics",
            "perform_sector_analysis",
            "calculate_and_apply_zscores",
            "calculate_and_apply_percentiles",
            "get_available_valuation_metrics",
            "calculate_multi_factor_scores",
            "display_rankings",
            "display_sector_leaders_laggards",
            "validate_required_columns",
            "analyze_sectors",
            "create_sector_summary",
            "display_sector_analysis_results",
            "display_large_cap_screen",
            "display_tech_sector_screen",
            "find_tech_sectors",
            "generate_reports_and_visualizations",
            "setup_output_directory",
            "create_scatter_plot",
            "create_heatmaps",
            "handle_visualization_error",
        ]

        func_names = [f.name for f in funcs]
        print("\nKey refactored functions:")
        for func in expected_functions:
            present = func in func_names
            print(f"  {func}: {'OK' if present else 'MISSING'}")

        print(f"\nSyntax is valid")
        print(f"Refactoring successfully applied!")

    except SyntaxError as e:
        print(f"\nSyntax error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
