"""
Script to remove redundant import statements from ml_finance_model_main.ipynb.
These imports are redundant because all functions are already imported at the top.
"""

import json
import re
from pathlib import Path


def remove_redundant_imports(notebook_path):
    """Remove redundant import statements from notebook cells."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Patterns for redundant imports to remove
    redundant_patterns = [
        r"from finance_ml\.advanced_preprocessing import .+\n",
        r"from finance_ml\.advanced_eda import .+\n",
        r"from finance_ml\.benchmarking import .+\n",
        r"from finance_ml\.eval import .+\n",
        r"from finance_ml\.advanced_features import .+\n",
        r"from finance_ml\.classification import .+\n",
        r"from finance_ml\.advanced_models import .+\n",
        r"from finance_ml\.data import .+\n",
        r"from finance_ml\.data_catalog import .+\n",
    ]

    changes_made = 0

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source

            original_text = source_text

            # Remove each redundant pattern
            for pattern in redundant_patterns:
                source_text = re.sub(pattern, "", source_text)

            # Also remove multi-line imports
            source_text = re.sub(
                r"from finance_ml\.(advanced_preprocessing|advanced_eda|benchmarking|eval|advanced_features|classification|advanced_models|data|data_catalog) import \(\n.*?\n.*?\)",
                "",
                source_text,
                flags=re.DOTALL,
            )

            if source_text != original_text:
                changes_made += 1
                # Convert back to list format
                cell["source"] = source_text.split("\n") if "\n" in source_text else [source_text]
                # Add a comment where import was removed
                if source_text.strip() and not source_text.strip().startswith("#"):
                    lines = source_text.split("\n")
                    if lines and not any(
                        pattern in line
                        for pattern in [
                            "# Functions already imported",
                            "# Function already imported",
                        ]
                        for line in lines[:3]
                    ):
                        cell["source"] = [
                            "# Functions already imported from finance_ml at the top\n"
                        ] + cell["source"]

    # Write back to file
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"✓ Removed redundant imports from {changes_made} cells")
    return changes_made


if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / "ml_finance_model_main.ipynb"
    changes = remove_redundant_imports(notebook_path)
    print(f"Total changes: {changes}")
