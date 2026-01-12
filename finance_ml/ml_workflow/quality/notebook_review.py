"""
Static analyzer for Jupyter notebook function calls.

Extends script_review.py to analyze notebook function calls and validate
them against actual function signatures to prevent parameter mismatches.

Following code_guidelines.md §6.2 - Python Script/Module Review Checklist.
"""

import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def parse_notebook(notebook_path: Path) -> List[str]:
    """
    Parse Jupyter notebook and extract Python code from code cells.

    Parameters
    ----------
    notebook_path : Path
        Path to .ipynb file

    Returns
    -------
    List[str]
        List of code cell sources
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if source:
                code_cells.append("".join(source))

    return code_cells


def extract_function_calls(code: str) -> List[Tuple[str, int, ast.Call]]:
    """
    Extract function calls from Python code using AST.

    Parameters
    ----------
    code : str
        Python source code

    Returns
    -------
    List[Tuple[str, int, ast.Call]]
        List of (function_name, line_number, ast.Call node) tuples
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"Syntax error in code: {e}")
        return []

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name:
                line_no = getattr(node, "lineno", 0)
                calls.append((func_name, line_no, node))

    return calls


def get_function_signature(module_name: str, function_name: str) -> Optional[inspect.Signature]:
    """
    Get function signature from module.

    Parameters
    ----------
    module_name : str
        Module path (e.g., 'finance_ml.ml_workflow.evaluation')
    function_name : str
        Function name

    Returns
    -------
    Optional[inspect.Signature]
        Function signature or None if not found
    """
    try:
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name, None)
        if func and callable(func):
            return inspect.signature(func)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import {module_name}.{function_name}: {e}")

    return None


def validate_call_against_signature(
    call_node: ast.Call, signature: inspect.Signature, function_name: str
) -> List[Dict[str, Any]]:
    """
    Validate function call against its signature.

    Parameters
    ----------
    call_node : ast.Call
        AST node for function call
    signature : inspect.Signature
        Expected function signature
    function_name : str
        Function name for error reporting

    Returns
    -------
    List[Dict[str, Any]]
        List of validation issues
    """
    issues = []
    sig_params = set(signature.parameters.keys())

    # Extract keyword arguments from call
    call_kwargs = set()
    for keyword in call_node.keywords:
        if keyword.arg:  # Skip **kwargs
            call_kwargs.add(keyword.arg)

    # Check for invalid parameters
    invalid_params = call_kwargs - sig_params
    for param in invalid_params:
        issues.append(
            {
                "function": function_name,
                "issue_type": "invalid_parameter",
                "parameter": param,
                "line": getattr(call_node, "lineno", 0),
                "message": f"Parameter '{param}' not in function signature",
                "valid_parameters": list(sig_params),
            }
        )

    return issues


def check_notebook_function_signatures(
    notebook_path: Path, target_functions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Check notebook function calls against actual signatures.

    Parameters
    ----------
    notebook_path : Path
        Path to Jupyter notebook file
    target_functions : Optional[Dict[str, str]]
        Dict mapping function names to module paths to check.
        If None, checks common evaluation functions.

    Returns
    -------
    Dict[str, Any]
        Validation report with issues and summary
    """
    if target_functions is None:
        # Default: check evaluation module functions
        target_functions = {
            "safety_rails_sensitivity_app": "finance_ml.ml_workflow.evaluation",
            "estimate_sector_bias": "finance_ml.ml_workflow.evaluation",
            "plot_metrics_by_sector_time": "finance_ml.ml_workflow.evaluation",
            "create_sector_bias_dashboard": "finance_ml.ml_workflow.evaluation",
            "compute_stacking_contributions": "finance_ml.ml_workflow.evaluation",
            "meta_error_maps": "finance_ml.ml_workflow.evaluation",
            "build_lineage_json": "finance_ml.ml_workflow.evaluation",
        }

    logger.info(f"Analyzing notebook: {notebook_path}")

    # Parse notebook
    code_cells = parse_notebook(notebook_path)

    all_issues = []
    calls_checked = 0

    # Check each code cell
    for cell_idx, code in enumerate(code_cells):
        calls = extract_function_calls(code)

        for func_name, line_no, call_node in calls:
            if func_name in target_functions:
                calls_checked += 1

                # Get expected signature
                module_name = target_functions[func_name]
                signature = get_function_signature(module_name, func_name)

                if signature is None:
                    logger.warning(f"Could not get signature for {func_name}")
                    continue

                # Validate call
                issues = validate_call_against_signature(call_node, signature, func_name)

                for issue in issues:
                    issue["cell_index"] = cell_idx
                    all_issues.append(issue)

    # Generate summary
    summary = {
        "notebook": str(notebook_path),
        "cells_analyzed": len(code_cells),
        "calls_checked": calls_checked,
        "issues_found": len(all_issues),
        "status": "PASS" if len(all_issues) == 0 else "FAIL",
    }

    return {"summary": summary, "issues": all_issues}


def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Print validation report to console.

    Parameters
    ----------
    report : Dict[str, Any]
        Validation report from check_notebook_function_signatures
    """
    summary = report["summary"]
    issues = report["issues"]

    print("\n" + "=" * 80)
    print("NOTEBOOK FUNCTION SIGNATURE VALIDATION REPORT")
    print("=" * 80)
    print(f"\nNotebook: {summary['notebook']}")
    print(f"Cells analyzed: {summary['cells_analyzed']}")
    print(f"Function calls checked: {summary['calls_checked']}")
    print(f"Issues found: {summary['issues_found']}")
    print(f"Status: {summary['status']}")

    if issues:
        print("\n" + "-" * 80)
        print("ISSUES FOUND:")
        print("-" * 80)

        for issue in issues:
            print(f"\n❌ {issue['function']}() - Cell {issue['cell_index']}, Line {issue['line']}")
            print(f"   Issue: {issue['message']}")
            print(f"   Invalid parameter: '{issue['parameter']}'")
            print(f"   Valid parameters: {', '.join(issue['valid_parameters'])}")
    else:
        print("\n✅ No issues found! All function calls match their signatures.")

    print("\n" + "=" * 80)


def main():
    """Command-line interface for notebook validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m finance_ml.ml_workflow.quality.notebook_review <notebook_path>")
        sys.exit(1)

    notebook_path = Path(sys.argv[1])

    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)

    report = check_notebook_function_signatures(notebook_path)
    print_validation_report(report)

    # Exit with error code if issues found
    sys.exit(0 if report["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
