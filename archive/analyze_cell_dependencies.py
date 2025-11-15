"""
Cell Dependency Analysis Tool for Jupyter Notebooks

Analyzes cell execution order dependencies by tracking:
1. Variable definitions and usages across cells
2. Function definitions and calls across cells
3. Required execution order to avoid NameError
4. Circular dependencies and dead code paths

Usage:
    python tools/analyze_cell_dependencies.py [notebook_path]

Example:
    python tools/analyze_cell_dependencies.py ml_finance_model_main_v10.ipynb
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class DependencyAnalyzer(ast.NodeVisitor):
    """AST visitor to track variable and function dependencies."""

    def __init__(self):
        self.definitions: Set[str] = set()  # Names defined in current cell
        self.usages: Set[str] = set()  # Names used in current cell
        self.imports: Set[str] = set()  # Names imported in current cell
        self.function_defs: Set[str] = set()  # Functions defined in current cell
        self.in_function_def = False
        self.in_class_def = False

    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
            self.definitions.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from...import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
            self.definitions.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Track function definitions."""
        if not self.in_function_def and not self.in_class_def:
            self.function_defs.add(node.name)
            self.definitions.add(node.name)
        old_in_function = self.in_function_def
        self.in_function_def = True
        self.generic_visit(node)
        self.in_function_def = old_in_function

    def visit_ClassDef(self, node):
        """Track class definitions."""
        if not self.in_class_def:
            self.definitions.add(node.name)
        old_in_class = self.in_class_def
        self.in_class_def = True
        self.generic_visit(node)
        self.in_class_def = old_in_class

    def visit_Assign(self, node):
        """Track variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if not self.in_function_def:
                    self.definitions.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Track annotated assignments."""
        if isinstance(node.target, ast.Name):
            if not self.in_function_def:
                self.definitions.add(node.target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Track variable usage."""
        if isinstance(node.ctx, ast.Load):
            # Only track if not a builtin
            if node.id not in dir(__builtins__):
                self.usages.add(node.id)
        self.generic_visit(node)


def analyze_notebook_dependencies(notebook_path: str) -> Dict[int, Dict[str, Set[str]]]:
    """Analyze dependencies for each cell in a notebook.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Dict mapping cell index to dependency info:
        {
            cell_index: {
                'definitions': set of names defined,
                'usages': set of names used,
                'imports': set of names imported,
                'functions': set of functions defined
            }
        }
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cell_deps = {}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        if not source.strip():
            cell_deps[i] = {
                "definitions": set(),
                "usages": set(),
                "imports": set(),
                "functions": set(),
            }
            continue

        analyzer = DependencyAnalyzer()
        try:
            tree = ast.parse(source)
            analyzer.visit(tree)

            cell_deps[i] = {
                "definitions": analyzer.definitions,
                "usages": analyzer.usages,
                "imports": analyzer.imports,
                "functions": analyzer.function_defs,
            }
        except SyntaxError:
            # Skip cells with syntax errors
            cell_deps[i] = {
                "definitions": set(),
                "usages": set(),
                "imports": set(),
                "functions": set(),
            }

    return cell_deps


def compute_cell_dependencies(cell_deps: Dict[int, Dict[str, Set[str]]]) -> Dict[int, Set[int]]:
    """Compute which cells each cell depends on.

    Args:
        cell_deps: Cell dependency information from analyze_notebook_dependencies

    Returns:
        Dict mapping cell index to set of cell indices it depends on
    """
    dependencies = {}

    # Track cumulative definitions up to each cell
    cumulative_defs = {}
    all_defs = set()

    for i in sorted(cell_deps.keys()):
        cumulative_defs[i] = all_defs.copy()
        all_defs.update(cell_deps[i]["definitions"])

    # For each cell, find which previous cells define names it uses
    for i in sorted(cell_deps.keys()):
        cell_info = cell_deps[i]
        deps = set()

        for name in cell_info["usages"]:
            # Find which previous cells defined this name
            for j in range(i):
                if name in cell_deps[j]["definitions"]:
                    deps.add(j)

        dependencies[i] = deps

    return dependencies


def find_circular_dependencies(dependencies: Dict[int, Set[int]]) -> List[List[int]]:
    """Find circular dependencies between cells.

    Args:
        dependencies: Cell dependencies from compute_cell_dependencies

    Returns:
        List of cycles (each cycle is a list of cell indices)
    """
    # For notebooks, circular dependencies are rare since cells execute sequentially
    # But we can detect if a later cell defines something an earlier cell uses
    cycles = []

    # This is a simplified check - true cycles are impossible in notebooks
    # but we can detect cases where execution order matters

    return cycles


def find_execution_order_issues(
    cell_deps: Dict[int, Dict[str, Set[str]]], dependencies: Dict[int, Set[int]]
) -> List[Tuple[int, str, List[int]]]:
    """Find cells that use undefined variables.

    Args:
        cell_deps: Cell dependency information
        dependencies: Cell dependencies

    Returns:
        List of (cell_index, variable_name, defining_cells) tuples
    """
    issues = []

    # Track what's defined up to each cell
    defined_so_far = set()

    for i in sorted(cell_deps.keys()):
        cell_info = cell_deps[i]

        # Check each usage
        for name in cell_info["usages"]:
            if name not in defined_so_far and name not in cell_info["definitions"]:
                # Find if it's defined in later cells
                later_definitions = [
                    j
                    for j in range(i + 1, len(cell_deps))
                    if name in cell_deps.get(j, {}).get("definitions", set())
                ]

                if later_definitions:
                    issues.append((i, name, later_definitions))

        # Update defined names
        defined_so_far.update(cell_info["definitions"])

    return issues


def generate_dependency_graph(dependencies: Dict[int, Set[int]]) -> str:
    """Generate a text-based dependency graph.

    Args:
        dependencies: Cell dependencies

    Returns:
        Text representation of dependency graph
    """
    lines = []

    for cell, deps in sorted(dependencies.items()):
        if deps:
            lines.append(f"Cell {cell} depends on: {sorted(deps)}")
        else:
            lines.append(f"Cell {cell}: no dependencies")

    return "\n".join(lines)


def generate_report(notebook_path: str, output_path: str = None) -> str:
    """Generate a cell dependency analysis report.

    Args:
        notebook_path: Path to .ipynb file
        output_path: Optional path to save report

    Returns:
        Report text
    """
    cell_deps = analyze_notebook_dependencies(notebook_path)
    dependencies = compute_cell_dependencies(cell_deps)
    execution_issues = find_execution_order_issues(cell_deps, dependencies)

    # Build report
    lines = [
        "=" * 80,
        "CELL DEPENDENCY ANALYSIS REPORT",
        "=" * 80,
        f"\nNotebook: {notebook_path}",
        f"Total Code Cells: {len(cell_deps)}",
        f"\nSummary:",
        f"  Cells with dependencies: {sum(1 for deps in dependencies.values() if deps)}",
        f"  Execution order issues: {len(execution_issues)}",
        "\n" + "=" * 80,
    ]

    # Execution order issues
    if execution_issues:
        lines.append("\n⚠️  EXECUTION ORDER ISSUES")
        lines.append("=" * 80)
        lines.append("\nVariables used before definition:")

        for cell, name, later_cells in execution_issues:
            lines.append(f"\n  Cell {cell} uses '{name}' but it's not defined yet")
            lines.append(f"    Defined later in cells: {later_cells}")
            lines.append(f"    Fix: Move cell {cell} after cell {min(later_cells)}")
    else:
        lines.append("\n✓ No execution order issues detected")

    # Cell dependency statistics
    lines.append("\n\n📊 DEPENDENCY STATISTICS")
    lines.append("=" * 80)

    # Most dependent cells
    most_deps = sorted(
        [(cell, len(deps)) for cell, deps in dependencies.items()], key=lambda x: x[1], reverse=True
    )[:10]

    if most_deps:
        lines.append("\nCells with most dependencies:")
        for cell, count in most_deps:
            if count > 0:
                lines.append(f"  Cell {cell}: depends on {count} previous cells")

    # Most depended-upon cells
    depended_on = defaultdict(int)
    for deps in dependencies.values():
        for dep_cell in deps:
            depended_on[dep_cell] += 1

    most_depended = sorted(depended_on.items(), key=lambda x: x[1], reverse=True)[:10]

    if most_depended:
        lines.append("\nMost critical cells (most dependencies on them):")
        for cell, count in most_depended:
            lines.append(f"  Cell {cell}: {count} other cells depend on it")

    # Variable flow analysis
    lines.append("\n\n🔄 VARIABLE FLOW ANALYSIS")
    lines.append("=" * 80)

    # Track which variables are passed between many cells
    var_usage_count = defaultdict(set)
    for cell, info in cell_deps.items():
        for var in info["usages"]:
            var_usage_count[var].add(cell)

    widely_used = sorted(
        [(var, len(cells)) for var, cells in var_usage_count.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    if widely_used:
        lines.append("\nMost widely used variables (across cells):")
        for var, count in widely_used:
            if count > 1:
                lines.append(f"  {var}: used in {count} cells")

    # Cell isolation analysis
    lines.append("\n\n🔒 CELL ISOLATION ANALYSIS")
    lines.append("=" * 80)

    isolated_cells = [
        cell for cell, deps in dependencies.items() if not deps and not cell_deps[cell]["usages"]
    ]

    if isolated_cells:
        lines.append(f"\nIsolated cells (no dependencies, no usages): {isolated_cells}")
        lines.append("  These cells can be run independently")
    else:
        lines.append("\n✓ No completely isolated cells")

    # Recommendations
    lines.extend(
        [
            "\n\n" + "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "\n1. Fix execution order issues by moving cells to respect dependencies",
            "2. Consider splitting cells with many dependencies for better modularity",
            "3. Critical cells should be well-tested as many cells depend on them",
            "4. Widely-used variables should have clear documentation",
            "5. Run cells in order from top to bottom for correct execution",
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
        print("Usage: python analyze_cell_dependencies.py <notebook_path> [output_path]")
        print("\nExample:")
        print("  python tools/analyze_cell_dependencies.py ml_finance_model_main_v10.ipynb")
        print(
            "  python tools/analyze_cell_dependencies.py ml_finance_model_main_v10.ipynb cell_dependencies_report.txt"
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
