#!/usr/bin/env python3
"""Analyze notebook phases to identify refactoring opportunities."""

import json


def analyze_notebook():
    """Analyze all phase cells for potential refactoring opportunities."""
    with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)

    print("Analyzing notebook for refactoring opportunities...")
    print("=" * 80)

    refactoring_candidates = []

    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])

        # Look for Phase markers
        if "PHASE 9." in source:
            # Extract phase number
            import re

            phase_match = re.search(r"PHASE 9\.(\d+)", source)
            if phase_match:
                phase_num = phase_match.group(1)

                # Count lines and check for potential refactoring indicators
                line_count = len(cell["source"])

                # Check for refactoring indicators
                has_magic_numbers = any(c.isdigit() and "0." in source for c in source)
                has_long_functions = "def " in source and line_count > 100
                has_duplicated_error_handling = source.count("try:") > 3
                has_nested_conditions = source.count("if ") > 10

                score = 0
                issues = []

                if line_count > 200:
                    score += 2
                    issues.append(f"Large cell ({line_count} lines)")

                if has_long_functions:
                    score += 2
                    issues.append("Contains long functions")

                if has_duplicated_error_handling:
                    score += 1
                    issues.append("Duplicated error handling")

                if has_nested_conditions:
                    score += 1
                    issues.append("Many conditional branches")

                if score > 0:
                    refactoring_candidates.append(
                        {
                            "index": idx,
                            "phase": f"9.{phase_num}",
                            "lines": line_count,
                            "score": score,
                            "issues": issues,
                        }
                    )

    # Sort by score (highest first)
    refactoring_candidates.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nFound {len(refactoring_candidates)} phases with refactoring opportunities:\n")

    for candidate in refactoring_candidates:
        print(f"Phase {candidate['phase']} (Cell {candidate['index']}):")
        print(f"  Score: {candidate['score']}")
        print(f"  Lines: {candidate['lines']}")
        print(f"  Issues:")
        for issue in candidate["issues"]:
            print(f"    - {issue}")
        print()

    print("=" * 80)
    print(f"\nPhase 9.7 has been refactored (Cell 151)")
    print(
        f"Remaining candidates: {len([c for c in refactoring_candidates if c['phase'] != '9.7'])}"
    )

    return refactoring_candidates


if __name__ == "__main__":
    analyze_notebook()
