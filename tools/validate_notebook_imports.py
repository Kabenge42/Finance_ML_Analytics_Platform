"""
Validate notebook imports - check for NameError issues in ml_finance_model_main.ipynb

This script:
1. Extracts all function calls from the notebook
2. Extracts all imports
3. Identifies potential missing imports
4. Reports any issues
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict


def load_notebook(notebook_path: Path) -> dict:
    """Load Jupyter notebook JSON."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_imports(notebook: dict) -> Dict[str, Set[str]]:
    """Extract all imports from notebook cells."""
    imports = defaultdict(set)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))

            # Match: from module import (...) with better handling of closing paren
            # Use greedy matching to get to the final closing paren
            pattern = r"from\s+(finance_ml[\w.]*)\s+import\s+\((.*)\)"
            matches = re.findall(pattern, source, re.DOTALL)
            for module, funcs in matches:
                # Split by comma and filter out comments
                lines = funcs.split("\n")
                func_list = []
                for line in lines:
                    # Remove inline comments
                    line = re.sub(r"#.*$", "", line)
                    # Split by comma and clean
                    for func in line.split(","):
                        func = func.strip()
                        if func and not func.startswith("#"):
                            func_list.append(func)
                imports[module].update(func_list)

            # Match: from module import func (single line, no parens)
            pattern = r"from\s+(finance_ml[\w.]*)\s+import\s+([^\n(]+)"
            matches = re.findall(pattern, source)
            for module, funcs in matches:
                func_list = [f.strip() for f in funcs.split(",")]
                func_list = [f.split(" as ")[0].strip() for f in func_list]
                func_list = [f for f in func_list if f and not f.startswith("#")]
                imports[module].update(func_list)

    return imports


def extract_function_calls(notebook: dict) -> Set[str]:
    """Extract potential function calls from notebook cells."""
    function_calls = set()

    # Common finance_ml functions to look for
    finance_ml_functions = [
        "assign_valuation_category",
        "calculate_mispricing_score",
        "calculate_risk_adjusted_mispricing",
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
        "identify_sector_leaders_laggards",
        "get_sector_specific_thresholds",
        "calculate_peer_comparisons",
        "generate_pdf_report",
        "simple_eda",
        "calculate_correlation_matrix",
        "find_top_correlations",
        "perform_pca",
        "compare_sector_means",
        "comprehensive_regression_metrics",
        "compute_metrics_by_segment",
        "residual_analysis_suite",
        "build_comprehensive_features",
        "prepare_regression_data",
        "train_stacking_regressor",
        "train_quantile_regressor",
        "test_normality",
        "calculate_feature_importance_rf",
        "generate_eda_report",
    ]

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))

            # Look for function calls
            for func in finance_ml_functions:
                if re.search(rf"\b{func}\s*\(", source):
                    function_calls.add(func)

    return function_calls


def check_imports(notebook_path: Path) -> Tuple[List[str], List[str]]:
    """Check notebook for missing imports."""
    notebook = load_notebook(notebook_path)

    imports = extract_imports(notebook)
    function_calls = extract_function_calls(notebook)

    # Flatten all imports
    all_imported = set()
    for module_funcs in imports.values():
        all_imported.update(module_funcs)

    # Find potentially missing imports
    missing = []
    for func in function_calls:
        if func not in all_imported:
            missing.append(func)

    # Create report
    issues = []
    successes = []

    for func in sorted(function_calls):
        if func in all_imported:
            successes.append(f"✓ {func} - imported")
        else:
            issues.append(f"✗ {func} - NOT IMPORTED (potential NameError)")

    return issues, successes


def main():
    """Main validation function."""
    notebook_path = Path(__file__).parent / "ml_finance_model_main.ipynb"

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1

    print("=" * 80)
    print("NOTEBOOK IMPORT VALIDATION")
    print("=" * 80)
    print(f"\nNotebook: {notebook_path.name}")

    issues, successes = check_imports(notebook_path)

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    if issues:
        print(f"\n⚠ Found {len(issues)} potential import issues:\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No missing imports detected!")

    print(f"\n✓ Successfully imported functions: {len(successes)}")

    if issues:
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("\nAdd missing imports to the appropriate sections of the notebook:")
        print("- Main imports (around line 155-163): for functions used throughout")
        print("- Phase-specific imports: for functions used in specific phases")
        return 1
    else:
        print("\n" + "=" * 80)
        print("✅ ALL IMPORTS VALIDATED SUCCESSFULLY")
        print("=" * 80)
        return 0


if __name__ == "__main__":
    exit(main())
