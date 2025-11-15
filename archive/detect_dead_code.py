"""
Dead Code Detection Tool for Jupyter Notebooks

Analyzes notebook cells to identify:
1. Unused variables (defined but never referenced)
2. Unused functions (defined but never called)
3. Unused imports (imported but never used)

Usage:
    python tools/detect_dead_code.py [notebook_path]

Example:
    python tools/detect_dead_code.py ml_finance_model_main_v10.ipynb
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DeadCodeDetector(ast.NodeVisitor):
    """AST visitor to detect unused variables, functions, and imports."""

    def __init__(self):
        self.definitions: Dict[str, List[int]] = defaultdict(list)  # name -> [cell_indices]
        self.usages: Dict[str, List[int]] = defaultdict(list)  # name -> [cell_indices]
        self.imports: Dict[str, List[int]] = defaultdict(list)  # name -> [cell_indices]
        self.functions: Dict[str, List[int]] = defaultdict(list)  # name -> [cell_indices]
        self.calls: Dict[str, List[int]] = defaultdict(list)  # name -> [cell_indices]
        self.current_cell = 0
        self.in_function_def = False

    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name].append(self.current_cell)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from...import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name].append(self.current_cell)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Track function definitions."""
        if not self.in_function_def:
            self.functions[node.name].append(self.current_cell)
        old_in_function = self.in_function_def
        self.in_function_def = True
        self.generic_visit(node)
        self.in_function_def = old_in_function

    def visit_Assign(self, node):
        """Track variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if not self.in_function_def:
                    self.definitions[target.id].append(self.current_cell)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Track variable usage."""
        if isinstance(node.ctx, ast.Load):
            self.usages[node.id].append(self.current_cell)
        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls."""
        if isinstance(node.func, ast.Name):
            self.calls[node.func.id].append(self.current_cell)
        elif isinstance(node.func, ast.Attribute):
            # Track method calls (e.g., obj.method())
            if isinstance(node.func.value, ast.Name):
                self.usages[node.func.value.id].append(self.current_cell)
        self.generic_visit(node)


def analyze_notebook(notebook_path: str) -> Tuple[DeadCodeDetector, int]:
    """Analyze a Jupyter notebook for dead code.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Tuple of (detector, total_cells)
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    detector = DeadCodeDetector()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        if not source.strip():
            continue

        detector.current_cell = i
        try:
            tree = ast.parse(source)
            detector.visit(tree)
        except SyntaxError:
            # Skip cells with syntax errors (e.g., IPython magic commands)
            pass

    return detector, len(code_cells)


def find_unused_imports(detector: DeadCodeDetector) -> List[Tuple[str, List[int]]]:
    """Find imports that are never used.

    Args:
        detector: DeadCodeDetector instance

    Returns:
        List of (import_name, cell_indices) tuples
    """
    unused = []
    for name, cells in detector.imports.items():
        # Check if imported name is used (excluding the import cell itself)
        usage_cells = detector.usages.get(name, [])
        call_cells = detector.calls.get(name, [])

        # Remove cells where it's imported from usage
        actual_usage = [c for c in usage_cells if c not in cells]
        actual_calls = [c for c in call_cells if c not in cells]

        if not actual_usage and not actual_calls:
            unused.append((name, cells))

    return sorted(unused, key=lambda x: x[0])


def find_unused_functions(detector: DeadCodeDetector) -> List[Tuple[str, List[int]]]:
    """Find functions that are defined but never called.

    Args:
        detector: DeadCodeDetector instance

    Returns:
        List of (function_name, cell_indices) tuples
    """
    unused = []
    for name, cells in detector.functions.items():
        # Check if function is called (excluding the definition cell itself)
        call_cells = detector.calls.get(name, [])
        actual_calls = [c for c in call_cells if c not in cells]

        if not actual_calls:
            unused.append((name, cells))

    return sorted(unused, key=lambda x: x[0])


def find_unused_variables(detector: DeadCodeDetector) -> List[Tuple[str, List[int]]]:
    """Find variables that are defined but never used.

    Args:
        detector: DeadCodeDetector instance

    Returns:
        List of (variable_name, cell_indices) tuples
    """
    # Common patterns to exclude from dead code detection
    excluded_patterns = {
        "_",  # Underscore (convention for unused)
        "fig",
        "ax",
        "axes",  # Matplotlib figures (side effects)
        "plt",  # Matplotlib pyplot (side effects)
        "logger",  # Logger (used for side effects)
        "cfg",
        "config",  # Config objects (used implicitly)
    }

    unused = []
    for name, cells in detector.definitions.items():
        # Skip if it's a private variable or in excluded patterns
        if name.startswith("_") or name in excluded_patterns:
            continue

        # Check if variable is used (excluding the definition cell itself)
        usage_cells = detector.usages.get(name, [])
        actual_usage = [c for c in usage_cells if c not in cells]

        if not actual_usage:
            unused.append((name, cells))

    return sorted(unused, key=lambda x: x[0])


def generate_report(notebook_path: str, output_path: str = None) -> str:
    """Generate a dead code detection report.

    Args:
        notebook_path: Path to .ipynb file
        output_path: Optional path to save report (default: stdout)

    Returns:
        Report text
    """
    detector, total_cells = analyze_notebook(notebook_path)

    unused_imports = find_unused_imports(detector)
    unused_functions = find_unused_functions(detector)
    unused_variables = find_unused_variables(detector)

    # Build report
    lines = [
        "=" * 80,
        "DEAD CODE DETECTION REPORT",
        "=" * 80,
        f"\nNotebook: {notebook_path}",
        f"Total Code Cells: {total_cells}",
        f"\nSummary:",
        f"  Unused Imports:   {len(unused_imports)}",
        f"  Unused Functions: {len(unused_functions)}",
        f"  Unused Variables: {len(unused_variables)}",
        "\n" + "=" * 80,
    ]

    # Unused imports section
    if unused_imports:
        lines.append("\n📦 UNUSED IMPORTS")
        lines.append("=" * 80)
        for name, cells in unused_imports:
            lines.append(f"\n  Import: {name}")
            lines.append(f"    Defined in cells: {cells}")
            lines.append(f"    Never used in any cell")
    else:
        lines.append("\n✓ No unused imports detected")

    # Unused functions section
    if unused_functions:
        lines.append("\n\n🔧 UNUSED FUNCTIONS")
        lines.append("=" * 80)
        for name, cells in unused_functions:
            lines.append(f"\n  Function: {name}")
            lines.append(f"    Defined in cells: {cells}")
            lines.append(f"    Never called in any cell")
    else:
        lines.append("\n✓ No unused functions detected")

    # Unused variables section
    if unused_variables:
        lines.append("\n\n📊 UNUSED VARIABLES")
        lines.append("=" * 80)
        lines.append("\n  (Excluding matplotlib figures, loggers, and private variables)")
        for name, cells in unused_variables[:20]:  # Limit to top 20
            lines.append(f"\n  Variable: {name}")
            lines.append(f"    Defined in cells: {cells}")
            lines.append(f"    Never referenced in any cell")

        if len(unused_variables) > 20:
            lines.append(f"\n  ... and {len(unused_variables) - 20} more")
    else:
        lines.append("\n✓ No unused variables detected")

    # Recommendations
    lines.extend(
        [
            "\n\n" + "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "\n1. Remove unused imports to reduce namespace pollution",
            "2. Remove or document unused functions (may be utility functions)",
            "3. Review unused variables - they may indicate incomplete code",
            "4. Consider adding _ prefix to intentionally unused variables",
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
        print("Usage: python detect_dead_code.py <notebook_path> [output_path]")
        print("\nExample:")
        print("  python tools/detect_dead_code.py ml_finance_model_main_v10.ipynb")
        print(
            "  python tools/detect_dead_code.py ml_finance_model_main_v10.ipynb dead_code_report.txt"
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
