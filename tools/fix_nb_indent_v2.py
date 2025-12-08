#!/usr/bin/env python
"""Fix indentation issues in notebook cells - Version 2."""
import json
import ast


def add_indent(line, spaces=4):
    """Add indentation to a line, preserving newline."""
    if line.strip() == "":
        return line
    return " " * spaces + line.lstrip() if not line.startswith(" " * spaces) else line


def fix_cell_71(source):
    """Fix cell 71: if block with unindented body and else."""
    # Lines 14-21 need 4 spaces, line 23 needs 4 spaces
    fixed = source.copy()
    # Add indent to lines 14-21 (the if body that's missing indent)
    for i in range(14, 22):
        if i < len(fixed) and fixed[i].strip():
            fixed[i] = "    " + fixed[i].lstrip()
    # Line 23 (else body) needs indent
    if len(fixed) > 23 and fixed[23].strip():
        fixed[23] = "    " + fixed[23].lstrip()
    return fixed


def fix_cell_72(source):
    """Fix cell 72: if block without indented body."""
    fixed = source.copy()
    # Line 8 is 'if isinstance(fold_assignments, dict):\n'
    # Lines 9+ need indentation until we see else or dedent
    i = 9
    while i < len(fixed):
        line = fixed[i]
        stripped = line.strip()
        if stripped.startswith("else:") or stripped.startswith("elif "):
            break
        if stripped and not line.startswith("    "):
            fixed[i] = "    " + line.lstrip()
        i += 1
    # Fix else body too
    if i < len(fixed) and fixed[i].strip().startswith("else:"):
        i += 1
        while i < len(fixed):
            line = fixed[i]
            if line.strip() and not line.startswith("    "):
                fixed[i] = "    " + line.lstrip()
            i += 1
    return fixed


def generic_fix(source):
    """Generic fix: find if/else/try blocks and fix indentation."""
    fixed = []
    i = 0
    indent_stack = []  # Stack of (indent_level, is_block_body)

    while i < len(source):
        line = source[i]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())

        # Check if line ends with colon (block starter)
        if stripped.endswith(":") and not stripped.startswith("#"):
            # This starts a new block
            fixed.append(line)
            expected_indent = current_indent + 4
            i += 1

            # Process block body
            while i < len(source):
                next_line = source[i]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())

                # Empty line - keep as is
                if not next_stripped:
                    fixed.append(next_line)
                    i += 1
                    continue

                # Check for else/elif/except/finally at same level as block start
                is_continuation = (
                    next_stripped.startswith("else:")
                    or next_stripped.startswith("elif ")
                    or next_stripped.startswith("except")
                    or next_stripped.startswith("finally:")
                )

                if is_continuation:
                    # Should be at block_start level
                    if next_indent != current_indent:
                        next_line = " " * current_indent + next_stripped
                        if not next_line.endswith("\n"):
                            next_line += "\n"
                    fixed.append(next_line)
                    i += 1
                    # Continue processing the continuation's body
                    continue

                # Regular line - should be indented
                if next_indent < expected_indent:
                    # Check if there's an else/except coming up
                    has_continuation = False
                    for j in range(i, min(i + 20, len(source))):
                        check = source[j].strip()
                        if (
                            check.startswith("else:")
                            or check.startswith("except")
                            or check.startswith("finally:")
                        ):
                            has_continuation = True
                            break
                        # If we hit another block at same or lower indent, stop checking
                        check_indent = len(source[j]) - len(source[j].lstrip())
                        if check and check_indent <= current_indent and not check.startswith("#"):
                            break

                    if has_continuation:
                        # Indent this line
                        next_line = " " * expected_indent + next_stripped
                        if not next_line.endswith("\n"):
                            next_line += "\n"
                    else:
                        # We've exited the block, put back and break
                        break

                fixed.append(next_line)
                i += 1

            continue

        fixed.append(line)
        i += 1

    return fixed


def fix_notebook(notebook_path, output_path=None):
    """Fix all cells in the notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if output_path is None:
        output_path = notebook_path

    fixed_count = 0
    still_broken = []

    for cell_idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]

        source_text = "".join(source)
        if not source_text.strip():
            continue

        # Check if cell has syntax error
        try:
            ast.parse(source_text)
            continue  # No error
        except SyntaxError as e:
            print(f"Cell {cell_idx}: {e.msg} at line {e.lineno}")

            # Try generic fix
            fixed_source = generic_fix(source)
            fixed_text = "".join(fixed_source)

            try:
                ast.parse(fixed_text)
                cell["source"] = fixed_source
                fixed_count += 1
                print(f"  ✓ Fixed")
            except SyntaxError as e2:
                print(f"  ✗ Still broken: {e2.msg} at line {e2.lineno}")
                still_broken.append(cell_idx)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"\nFixed {fixed_count} cells")
    if still_broken:
        print(f"Still broken: {still_broken}")

    return fixed_count, still_broken


if __name__ == "__main__":
    fix_notebook("ml_finance_model_main2_0.ipynb")
