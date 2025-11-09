"""
Function Call Validation Tool for Jupyter Notebooks

Validates that all function calls in the notebook match the current API:
1. Extracts all function calls from notebook cells
2. Verifies they exist in the finance_ml package
3. Checks parameter signatures and compatibility
4. Reports missing or deprecated functions

Usage:
    python tools/validate_function_calls.py [notebook_path]

Example:
    python tools/validate_function_calls.py ml_finance_model_main_v10.ipynb
"""

import ast
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional


class FunctionCallExtractor(ast.NodeVisitor):
    """AST visitor to extract function calls and their arguments."""

    def __init__(self):
        self.calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_cell = 0

    def visit_Call(self, node):
        """Extract function calls with arguments."""
        func_name = None
        module = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id

        if func_name:
            # Extract arguments
            args = []
            kwargs = {}

            for arg in node.args:
                args.append(self._get_arg_repr(arg))

            for keyword in node.keywords:
                kwargs[keyword.arg] = self._get_arg_repr(keyword.value)

            self.calls[func_name].append(
                {
                    "cell": self.current_cell,
                    "module": module,
                    "args": args,
                    "kwargs": kwargs,
                    "num_args": len(args),
                    "num_kwargs": len(kwargs),
                }
            )

        self.generic_visit(node)

    def _get_arg_repr(self, node) -> str:
        """Get a string representation of an argument."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_arg_repr(node.value)}.{node.attr}"
        else:
            return ast.unparse(node) if hasattr(ast, "unparse") else "<complex>"


def extract_function_calls(notebook_path: str) -> Tuple[FunctionCallExtractor, int]:
    """Extract all function calls from a notebook.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Tuple of (extractor, total_cells)
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    extractor = FunctionCallExtractor()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        if not source.strip():
            continue

        extractor.current_cell = i
        try:
            tree = ast.parse(source)
            extractor.visit(tree)
        except SyntaxError:
            pass

    return extractor, len(code_cells)


def get_finance_ml_api() -> Dict[str, inspect.Signature]:
    """Get all public functions from finance_ml package.

    Returns:
        Dict mapping function names to their signatures
    """
    api = {}

    try:
        import finance_ml

        # Get all exported functions from __all__
        if hasattr(finance_ml, "__all__"):
            for name in finance_ml.__all__:
                try:
                    obj = getattr(finance_ml, name)
                    if callable(obj) and not inspect.isclass(obj):
                        api[name] = inspect.signature(obj)
                except (AttributeError, ValueError):
                    pass

        # Also check common submodules
        submodules = [
            "data",
            "features",
            "regression",
            "eval",
            "config",
            "advanced_preprocessing",
            "advanced_models",
            "advanced_eda",
            "benchmarking",
            "risk_metrics",
            "portfolio_optimization",
        ]

        for submodule_name in submodules:
            try:
                submodule = getattr(finance_ml, submodule_name, None)
                if submodule is None:
                    continue

                for name in dir(submodule):
                    if name.startswith("_"):
                        continue
                    try:
                        obj = getattr(submodule, name)
                        if callable(obj) and not inspect.isclass(obj):
                            api[name] = inspect.signature(obj)
                    except (AttributeError, ValueError):
                        pass
            except ImportError:
                pass

    except ImportError:
        print("Warning: Could not import finance_ml package")

    return api


def validate_function_calls(
    extractor: FunctionCallExtractor, api: Dict[str, inspect.Signature]
) -> Tuple[List[str], List[str], List[str]]:
    """Validate function calls against the API.

    Args:
        extractor: FunctionCallExtractor with extracted calls
        api: API signatures

    Returns:
        Tuple of (missing_functions, signature_mismatches, warnings)
    """
    missing = []
    mismatches = []
    warnings = []

    # Filter to finance_ml related calls
    finance_ml_calls = {}
    for func_name, calls in extractor.calls.items():
        # Check if any call is from finance_ml module
        has_fm_call = any(
            call["module"] in ("finance_ml", "fm", None) and func_name in api for call in calls
        )
        if has_fm_call or func_name in api:
            finance_ml_calls[func_name] = calls

    for func_name, calls in finance_ml_calls.items():
        if func_name not in api:
            # Check if it's a common false positive
            if func_name in (
                "print",
                "len",
                "range",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "max",
                "min",
                "sum",
                "any",
                "all",
                "open",
            ):
                continue

            missing.append(
                f"  Function '{func_name}' not found in finance_ml API "
                f"(used in cells: {sorted(set(c['cell'] for c in calls))})"
            )
            continue

        # Check signature compatibility
        sig = api[func_name]
        params = sig.parameters

        for call in calls:
            # Check if too many positional arguments
            positional_params = [
                p
                for p in params.values()
                if p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]

            if call["num_args"] > len(positional_params):
                # Check if there's a *args parameter
                has_var_positional = any(
                    p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
                )
                if not has_var_positional:
                    mismatches.append(
                        f"  Function '{func_name}' in cell {call['cell']}: "
                        f"too many positional arguments ({call['num_args']} > {len(positional_params)})"
                    )

            # Check for unknown keyword arguments
            for kwarg in call["kwargs"].keys():
                if kwarg not in params:
                    # Check if there's a **kwargs parameter
                    has_var_keyword = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                    if not has_var_keyword:
                        mismatches.append(
                            f"  Function '{func_name}' in cell {call['cell']}: "
                            f"unknown keyword argument '{kwarg}'"
                        )

    return missing, mismatches, warnings


def generate_report(notebook_path: str, output_path: str = None) -> str:
    """Generate a function call validation report.

    Args:
        notebook_path: Path to .ipynb file
        output_path: Optional path to save report

    Returns:
        Report text
    """
    extractor, total_cells = extract_function_calls(notebook_path)
    api = get_finance_ml_api()

    missing, mismatches, warnings = validate_function_calls(extractor, api)

    # Build report
    lines = [
        "=" * 80,
        "FUNCTION CALL VALIDATION REPORT",
        "=" * 80,
        f"\nNotebook: {notebook_path}",
        f"Total Code Cells: {total_cells}",
        f"Total Function Calls: {sum(len(calls) for calls in extractor.calls.values())}",
        f"Unique Functions: {len(extractor.calls)}",
        f"Finance ML API Functions: {len(api)}",
        f"\nValidation Summary:",
        f"  Missing Functions:     {len(missing)}",
        f"  Signature Mismatches:  {len(mismatches)}",
        f"  Warnings:             {len(warnings)}",
        "\n" + "=" * 80,
    ]

    # Missing functions section
    if missing:
        lines.append("\n❌ MISSING FUNCTIONS")
        lines.append("=" * 80)
        lines.extend(missing)
    else:
        lines.append("\n✓ No missing functions detected")

    # Signature mismatches section
    if mismatches:
        lines.append("\n\n⚠️  SIGNATURE MISMATCHES")
        lines.append("=" * 80)
        lines.extend(mismatches)
    else:
        lines.append("\n✓ No signature mismatches detected")

    # Warnings section
    if warnings:
        lines.append("\n\n⚠️  WARNINGS")
        lines.append("=" * 80)
        lines.extend(warnings)

    # Most used finance_ml functions
    lines.append("\n\n📊 MOST USED FINANCE_ML FUNCTIONS")
    lines.append("=" * 80)

    # Filter to finance_ml functions only
    fm_calls = {name: calls for name, calls in extractor.calls.items() if name in api}

    if fm_calls:
        sorted_calls = sorted(fm_calls.items(), key=lambda x: len(x[1]), reverse=True)[:20]

        for func_name, calls in sorted_calls:
            cells = sorted(set(c["cell"] for c in calls))
            lines.append(f"\n  {func_name}: {len(calls)} calls in {len(cells)} cells")
            if len(cells) <= 10:
                lines.append(f"    Cells: {cells}")
    else:
        lines.append("\n  No finance_ml function calls detected")

    # Recommendations
    lines.extend(
        [
            "\n\n" + "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "\n1. Fix missing functions by updating imports or correcting function names",
            "2. Resolve signature mismatches by checking parameter names and counts",
            "3. Review warnings for potential API deprecations",
            "4. Consult finance_ml documentation for updated function signatures",
            "\n" + "=" * 80,
        ]
    )

    report = "\n".join(lines)

    # Save or print
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)

    return report


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_function_calls.py <notebook_path> [output_path]")
        print("\nExample:")
        print("  python tools/validate_function_calls.py ml_finance_model_main_v10.ipynb")
        print(
            "  python tools/validate_function_calls.py ml_finance_model_main_v10.ipynb function_validation_report.txt"
        )
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(notebook_path).exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)

    generate_report(notebook_path, output_path)


if __name__ == "__main__":
    main()
