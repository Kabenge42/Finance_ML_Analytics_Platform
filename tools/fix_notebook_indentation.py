#!/usr/bin/env python
"""Fix indentation issues in notebook cells."""
import json
import re
import ast
import sys


def fix_cell_indentation(source_lines):
    """
    Fix indentation issues in a code cell.

    The main pattern is:
    - if/else/try/except/for/while statements should have indented blocks
    - Lines following these without indentation need to be indented
    """
    if not source_lines:
        return source_lines

    fixed_lines = []
    current_indent = 0
    in_block = False
    block_indent = 0

    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            i += 1
            continue

        # Get current line indentation
        leading_spaces = len(line) - len(line.lstrip())

        # Check if this is a block start (if, for, while, try, with, def, class, elif, else, except, finally)
        block_starters = [
            "if ",
            "for ",
            "while ",
            "try:",
            "with ",
            "def ",
            "class ",
            "elif ",
            "async ",
        ]
        is_block_start = any(stripped.startswith(s) for s in block_starters)
        is_block_start = is_block_start or stripped == "try:"

        # Check if this line ends with a colon (indicating a block)
        ends_with_colon = stripped.endswith(":") and not stripped.startswith("#")

        # Check for else/elif/except/finally that should match an if/try
        continuation_keywords = ["else:", "elif ", "except:", "except ", "finally:"]
        is_continuation = any(
            stripped.startswith(s) or stripped == s.rstrip() for s in continuation_keywords
        )

        fixed_lines.append(line)

        # If this line starts a block, check the next line's indentation
        if ends_with_colon and not is_continuation:
            block_indent = leading_spaces + 4
            # Check if next non-empty line needs indentation
            j = i + 1
            while j < len(source_lines) and not source_lines[j].strip():
                j += 1

            if j < len(source_lines):
                next_line = source_lines[j]
                next_stripped = next_line.strip()
                next_leading = len(next_line) - len(next_line.lstrip())

                # If next line isn't indented enough and isn't else/elif/except/finally
                is_next_continuation = any(
                    next_stripped.startswith(s) or next_stripped == s.rstrip()
                    for s in continuation_keywords
                )

                if next_leading <= leading_spaces and not is_next_continuation and next_stripped:
                    # Need to indent subsequent lines until we hit else/except/finally or dedent
                    # This is the broken pattern we're fixing
                    pass  # We'll handle this in a second pass

        i += 1

    return fixed_lines


def fix_notebook_cells(notebook_path, output_path=None):
    """Fix indentation in problematic notebook cells."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if output_path is None:
        output_path = notebook_path

    # Cells that need fixing based on syntax check
    problem_cells = [
        71,
        72,
        73,
        76,
        78,
        82,
        83,
        91,
        92,
        94,
        96,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        117,
        119,
        120,
    ]

    fixed_count = 0
    for cell_idx in problem_cells:
        if cell_idx >= len(nb["cells"]):
            continue

        cell = nb["cells"][cell_idx]
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, str):
            source = source.split("\n")

        # Join and try to parse
        source_text = "".join(source)

        try:
            ast.parse(source_text)
            continue  # No error, skip
        except SyntaxError as e:
            print(f"Fixing cell {cell_idx}: {e.msg} at line {e.lineno}")

            # Apply specific fixes based on common patterns
            fixed_source = fix_if_else_indentation(source)

            # Verify fix worked
            fixed_text = "".join(fixed_source)
            try:
                ast.parse(fixed_text)
                cell["source"] = fixed_source
                fixed_count += 1
                print(f"  ✓ Cell {cell_idx} fixed successfully")
            except SyntaxError as e2:
                print(f"  ✗ Cell {cell_idx} still has error: {e2.msg}")

    # Save the fixed notebook
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"\nFixed {fixed_count} cells. Saved to {output_path}")
    return fixed_count


def fix_if_else_indentation(source_lines):
    """
    Fix the specific pattern where code after if: isn't indented.

    Pattern:
    if condition:
        print("start")
    code_that_should_be_indented()  # <- needs 4 more spaces
    more_code()  # <- needs 4 more spaces
    else:  # <- should stay at if's level
    unindented_else_body()  # <- needs 4 spaces
    """
    if isinstance(source_lines, str):
        source_lines = source_lines.split("\n")
        source_lines = [
            line + "\n" if i < len(source_lines) - 1 else line
            for i, line in enumerate(source_lines)
        ]

    result = []
    i = 0

    while i < len(source_lines):
        line = source_lines[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        result.append(line)

        # Check if this line ends a block-starting statement
        if stripped.endswith(":") and not stripped.startswith("#"):
            block_start_indent = current_indent
            expected_body_indent = current_indent + 4

            i += 1

            # Process lines until we hit a dedent back to or below block_start_indent
            while i < len(source_lines):
                next_line = source_lines[i]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())

                # Empty line - keep as is
                if not next_stripped:
                    result.append(next_line)
                    i += 1
                    continue

                # Check for else/elif/except/finally at block level
                is_continuation = (
                    next_stripped.startswith("else:")
                    or next_stripped.startswith("elif ")
                    or next_stripped.startswith("except")
                    or next_stripped.startswith("finally:")
                )

                if is_continuation and next_indent <= block_start_indent:
                    # This should be at the same level as the if/try
                    if next_indent < block_start_indent:
                        # Need to indent to match
                        spaces = " " * block_start_indent
                        if next_line.endswith("\n"):
                            next_line = spaces + next_stripped + "\n"
                        else:
                            next_line = spaces + next_stripped
                    result.append(next_line)
                    i += 1
                    continue

                # If line is at or below the block start indent (and not a continuation)
                # but should be inside the block, indent it
                if next_indent <= block_start_indent and not is_continuation:
                    # Check if we're still logically inside the block
                    # Heuristic: if the next line with same indent is else/except, we're in the block
                    in_block = False
                    for j in range(i, len(source_lines)):
                        check_line = source_lines[j].strip()
                        check_indent = len(source_lines[j]) - len(source_lines[j].lstrip())
                        if not check_line:
                            continue
                        if check_indent <= block_start_indent:
                            if (
                                check_line.startswith("else:")
                                or check_line.startswith("elif ")
                                or check_line.startswith("except")
                                or check_line.startswith("finally:")
                            ):
                                in_block = True
                            break

                    if in_block:
                        # Indent this line
                        spaces = " " * expected_body_indent
                        if next_line.endswith("\n"):
                            next_line = spaces + next_stripped + "\n"
                        else:
                            next_line = spaces + next_stripped
                    else:
                        # We've exited the block
                        break

                result.append(next_line)
                i += 1

                # If we just added a continuation (else/except), don't break
                if is_continuation:
                    continue

            continue

        i += 1

    return result


if __name__ == "__main__":
    notebook_path = "ml_finance_model_main2_0.ipynb"
    output_path = notebook_path  # Overwrite

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    fix_notebook_cells(notebook_path, output_path)
