#!/usr/bin/env python3
"""Remove AttributeError workaround from notebook since bug is now fixed."""

import json
from pathlib import Path


def fix_notebook():
    """Remove AttributeError workaround from ml_finance_model_main.ipynb."""

    notebook_path = Path("ml_finance_model_main.ipynb")

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find and fix the cell with AttributeError workaround
    modified = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            # Convert to string for easier checking
            source_str = "".join(source) if isinstance(source, list) else source

            # Check if this cell contains the AttributeError workaround
            if "except AttributeError as e:" in source_str and "known package bug" in source_str:
                print("Found cell with AttributeError workaround")

                # Remove the AttributeError except block
                # Keep only the try block and the general Exception handler
                new_source = []
                skip_lines = False
                in_attribute_error = False

                for line in source if isinstance(source, list) else source.split("\n"):
                    # Detect start of AttributeError block
                    if "except AttributeError as e:" in line:
                        in_attribute_error = True
                        skip_lines = True
                        print(f"  Removing line: {line.strip()}")
                        continue

                    # Detect end of AttributeError block (next except or end of try)
                    if in_attribute_error and (
                        "except Exception as e:" in line
                        or "except " in line
                        and "AttributeError" not in line
                    ):
                        in_attribute_error = False
                        skip_lines = False
                        # Include this line (it's the next except block)
                        new_source.append(line)
                        continue

                    # Skip lines inside AttributeError block
                    if skip_lines and in_attribute_error:
                        print(f"  Removing line: {line.strip()}")
                        continue

                    # Keep all other lines
                    new_source.append(line)

                # Update cell source
                cell["source"] = new_source
                modified = True
                print(f"  Removed AttributeError workaround block")
                break

    if modified:
        # Write back
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"\n✓ Updated {notebook_path}")
        print(
            "  Removed AttributeError workaround - bug is now fixed in finance_ml.eval.simple_eda()"
        )
    else:
        print("\n⚠ AttributeError workaround not found in notebook")

    return modified


if __name__ == "__main__":
    fix_notebook()
