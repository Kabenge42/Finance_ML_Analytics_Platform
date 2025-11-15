"""
Performance Profiling Tool for Jupyter Notebooks

Analyzes notebook cells to identify performance bottlenecks:
1. Estimates computational complexity based on code patterns
2. Identifies expensive operations (loops, large data operations)
3. Detects redundant computations
4. Suggests optimization opportunities

Usage:
    python tools/profile_notebook_performance.py [notebook_path]

Example:
    python tools/profile_notebook_performance.py ml_finance_model_main_v10.ipynb
"""

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class PerformanceAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze performance characteristics of code."""

    def __init__(self):
        self.loop_count = 0
        self.nested_loop_depth = 0
        self.current_loop_depth = 0
        self.comprehension_count = 0
        self.dataframe_operations = []
        self.large_iterations = []
        self.function_calls = defaultdict(int)
        self.expensive_patterns = []

    def visit_For(self, node):
        """Track for loops."""
        self.loop_count += 1
        self.current_loop_depth += 1
        self.nested_loop_depth = max(self.nested_loop_depth, self.current_loop_depth)

        # Check if iterating over large range
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if node.iter.args and isinstance(node.iter.args[0], ast.Constant):
                    if node.iter.args[0].value > 1000:
                        self.large_iterations.append(("for", node.iter.args[0].value))

        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node):
        """Track while loops."""
        self.loop_count += 1
        self.current_loop_depth += 1
        self.nested_loop_depth = max(self.nested_loop_depth, self.current_loop_depth)
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_ListComp(self, node):
        """Track list comprehensions."""
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """Track dict comprehensions."""
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """Track set comprehensions."""
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls and detect expensive operations."""
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

            # Check for pandas/numpy operations
            if isinstance(node.func.value, ast.Name):
                obj = node.func.value.id
                if obj in ("df", "data", "stocks", "all_stocks"):
                    self.dataframe_operations.append(func_name)

        if func_name:
            self.function_calls[func_name] += 1

            # Detect expensive operations
            expensive_ops = {
                "apply",
                "iterrows",
                "itertuples",  # Pandas expensive
                "fit",
                "fit_transform",
                "predict",  # ML operations
                "sort",
                "sort_values",
                "groupby",  # Sorting/grouping
                "merge",
                "join",
                "concat",  # Data joining
            }

            if func_name in expensive_ops:
                self.expensive_patterns.append(func_name)

        self.generic_visit(node)


def estimate_cell_complexity(source: str) -> Dict[str, any]:
    """Estimate computational complexity of a cell.

    Args:
        source: Cell source code

    Returns:
        Dict with complexity metrics
    """
    analyzer = PerformanceAnalyzer()

    try:
        tree = ast.parse(source)
        analyzer.visit(tree)

        # Calculate complexity score
        score = 0
        score += analyzer.loop_count * 10
        score += analyzer.nested_loop_depth * 20
        score += analyzer.comprehension_count * 5
        score += len(analyzer.expensive_patterns) * 15
        score += len(analyzer.dataframe_operations) * 3

        return {
            "complexity_score": score,
            "loop_count": analyzer.loop_count,
            "nested_loop_depth": analyzer.nested_loop_depth,
            "comprehension_count": analyzer.comprehension_count,
            "dataframe_operations": analyzer.dataframe_operations,
            "expensive_patterns": analyzer.expensive_patterns,
            "large_iterations": analyzer.large_iterations,
            "function_calls": dict(analyzer.function_calls),
        }
    except SyntaxError:
        return {
            "complexity_score": 0,
            "loop_count": 0,
            "nested_loop_depth": 0,
            "comprehension_count": 0,
            "dataframe_operations": [],
            "expensive_patterns": [],
            "large_iterations": [],
            "function_calls": {},
        }


def detect_redundant_operations(cell_sources: List[str]) -> List[Tuple[int, int, str]]:
    """Detect potentially redundant operations across cells.

    Args:
        cell_sources: List of cell source code strings

    Returns:
        List of (cell1, cell2, operation) tuples indicating redundancy
    """
    redundancies = []

    # Track expensive operations in each cell
    cell_operations = []

    for source in cell_sources:
        ops = set()
        # Look for common expensive patterns
        patterns = [
            r"\.fit\(",
            r"\.fit_transform\(",
            r"\.groupby\(",
            r"\.merge\(",
            r"\.sort_values\(",
            r"\.read_csv\(",
            r"\.read_sql\(",
        ]

        for pattern in patterns:
            if re.search(pattern, source):
                ops.add(pattern.strip(r"\.()"))

        cell_operations.append(ops)

    # Find operations that appear in multiple cells
    for i in range(len(cell_operations)):
        for j in range(i + 1, len(cell_operations)):
            common = cell_operations[i] & cell_operations[j]
            for op in common:
                redundancies.append((i, j, op))

    return redundancies


def suggest_optimizations(complexity: Dict[str, any]) -> List[str]:
    """Suggest optimizations based on complexity analysis.

    Args:
        complexity: Complexity metrics from estimate_cell_complexity

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    # Check for nested loops
    if complexity["nested_loop_depth"] > 1:
        suggestions.append(
            f"Nested loops detected (depth {complexity['nested_loop_depth']}). "
            "Consider vectorizing with NumPy/Pandas operations."
        )

    # Check for many loops
    if complexity["loop_count"] > 3:
        suggestions.append(
            f"{complexity['loop_count']} loops detected. "
            "Consider using vectorized operations or comprehensions."
        )

    # Check for expensive patterns
    if "iterrows" in complexity["expensive_patterns"]:
        suggestions.append(
            "Using df.iterrows() is slow. Consider vectorized operations or apply()."
        )

    if "apply" in complexity["expensive_patterns"]:
        suggestions.append(
            "Using df.apply() can be slow. Consider vectorized operations if possible."
        )

    # Check for large iterations
    if complexity["large_iterations"]:
        for loop_type, size in complexity["large_iterations"]:
            suggestions.append(
                f"Large iteration detected ({size} iterations). "
                "Consider batching or parallel processing."
            )

    # Check for multiple groupby operations
    groupby_count = complexity["dataframe_operations"].count("groupby")
    if groupby_count > 2:
        suggestions.append(
            f"Multiple groupby operations ({groupby_count}). "
            "Consider combining them or caching results."
        )

    # Check for sorting
    if (
        "sort_values" in complexity["expensive_patterns"]
        or "sort" in complexity["expensive_patterns"]
    ):
        suggestions.append("Sorting detected. Ensure data is only sorted when necessary.")

    return suggestions


def analyze_notebook_performance(notebook_path: str) -> Dict[int, Dict[str, any]]:
    """Analyze performance characteristics of all cells.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Dict mapping cell index to performance metrics
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cell_performance = {}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    cell_sources = []

    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        cell_sources.append(source)

        if not source.strip():
            cell_performance[i] = {
                "complexity_score": 0,
                "suggestions": [],
                "cell_size": 0,
            }
            continue

        complexity = estimate_cell_complexity(source)
        suggestions = suggest_optimizations(complexity)

        cell_performance[i] = {
            **complexity,
            "suggestions": suggestions,
            "cell_size": len(source),
        }

    # Detect redundancies
    redundancies = detect_redundant_operations(cell_sources)

    # Add redundancy warnings
    for cell1, cell2, op in redundancies:
        if cell1 in cell_performance:
            if "redundancies" not in cell_performance[cell1]:
                cell_performance[cell1]["redundancies"] = []
            cell_performance[cell1]["redundancies"].append((cell2, op))

    return cell_performance


def generate_report(notebook_path: str, output_path: str = None) -> str:
    """Generate a performance profiling report.

    Args:
        notebook_path: Path to .ipynb file
        output_path: Optional path to save report

    Returns:
        Report text
    """
    cell_performance = analyze_notebook_performance(notebook_path)

    # Build report
    lines = [
        "=" * 80,
        "PERFORMANCE PROFILING REPORT",
        "=" * 80,
        f"\nNotebook: {notebook_path}",
        f"Total Code Cells: {len(cell_performance)}",
        "\n" + "=" * 80,
    ]

    # Overall statistics
    total_complexity = sum(c["complexity_score"] for c in cell_performance.values())
    avg_complexity = total_complexity / len(cell_performance) if cell_performance else 0

    lines.append("\n📊 OVERALL STATISTICS")
    lines.append("=" * 80)
    lines.append(f"\nTotal Complexity Score: {total_complexity}")
    lines.append(f"Average Complexity per Cell: {avg_complexity:.1f}")

    # Identify slowest cells
    slowest_cells = sorted(
        [(i, c["complexity_score"]) for i, c in cell_performance.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    lines.append("\n\n🐌 POTENTIALLY SLOW CELLS")
    lines.append("=" * 80)
    lines.append("\nCells ranked by estimated complexity:")

    for cell, score in slowest_cells:
        if score > 0:
            details = cell_performance[cell]
            lines.append(f"\n  Cell {cell}: Complexity Score = {score}")

            if details["loop_count"] > 0:
                lines.append(f"    - {details['loop_count']} loops")
            if details["nested_loop_depth"] > 0:
                lines.append(f"    - Nested loops (max depth {details['nested_loop_depth']})")
            if details["expensive_patterns"]:
                lines.append(
                    f"    - Expensive operations: {', '.join(set(details['expensive_patterns']))}"
                )
            if details["dataframe_operations"]:
                lines.append(f"    - {len(details['dataframe_operations'])} DataFrame operations")

    # Optimization suggestions
    cells_with_suggestions = [
        (i, c["suggestions"]) for i, c in cell_performance.items() if c["suggestions"]
    ]

    if cells_with_suggestions:
        lines.append("\n\n💡 OPTIMIZATION SUGGESTIONS")
        lines.append("=" * 80)

        for cell, suggestions in cells_with_suggestions[:15]:  # Limit to 15
            lines.append(f"\nCell {cell}:")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")
    else:
        lines.append("\n\n✓ No obvious optimization opportunities detected")

    # Redundant operations
    redundancies = []
    for cell, info in cell_performance.items():
        if "redundancies" in info:
            for other_cell, op in info["redundancies"]:
                redundancies.append((cell, other_cell, op))

    if redundancies:
        lines.append("\n\n🔄 REDUNDANT OPERATIONS")
        lines.append("=" * 80)
        lines.append("\nSame expensive operations in multiple cells:")

        for cell1, cell2, op in redundancies[:10]:  # Limit to 10
            lines.append(f"\n  '{op}' in cells {cell1} and {cell2}")
            lines.append(f"    Consider caching results or combining cells")

    # Performance hotspots
    lines.append("\n\n🔥 PERFORMANCE HOTSPOTS")
    lines.append("=" * 80)

    # Aggregate function calls across all cells
    all_calls = defaultdict(int)
    for info in cell_performance.values():
        for func, count in info.get("function_calls", {}).items():
            all_calls[func] += count

    top_calls = sorted(all_calls.items(), key=lambda x: x[1], reverse=True)[:15]

    if top_calls:
        lines.append("\nMost frequently called functions:")
        for func, count in top_calls:
            lines.append(f"  {func}: {count} calls")

    # Recommendations
    lines.extend(
        [
            "\n\n" + "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "\n1. Focus optimization efforts on cells with highest complexity scores",
            "2. Replace iterrows() with vectorized operations or apply()",
            "3. Cache results of expensive operations when used multiple times",
            "4. Use profiling tools (%%timeit, cProfile) to measure actual execution time",
            "5. Consider using parallel processing for independent expensive operations",
            "6. Reduce DataFrame operations by chaining or using method chaining",
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
        print("Usage: python profile_notebook_performance.py <notebook_path> [output_path]")
        print("\nExample:")
        print("  python tools/profile_notebook_performance.py ml_finance_model_main_v10.ipynb")
        print(
            "  python tools/profile_notebook_performance.py ml_finance_model_main_v10.ipynb performance_report.txt"
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
