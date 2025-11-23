"""
Fix the logging import error in ml_finance_model_main.ipynb.

This script replaces the incorrect 'setup_logger' import with the correct
'configure_logging' and 'get_logger' imports, following code_guidelines.md.
"""

import json
import sys
from pathlib import Path


def fix_logging_import(notebook_path: Path) -> None:
    """
    Fix the logging import error in the specified notebook.

    Args:
        notebook_path: Path to the Jupyter notebook file
    """
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell with the incorrect import
    fixed = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # Check if this is the cell with the incorrect import
        if 'from finance_ml.logging_config import setup_logger' in source:
            print(f"Found incorrect import in cell {i}")

            # Replace the import statement
            old_import = 'from finance_ml.logging_config import setup_logger'
            new_import = 'from finance_ml.logging_config import configure_logging, get_logger'

            old_usage = 'logger = setup_logger(__name__, level=logging.INFO)'
            new_usage = '''configure_logging(level=logging.INFO, console=True)
    logger = get_logger(__name__)'''

            # Fix the source
            fixed_source = source.replace(old_import, new_import)
            fixed_source = fixed_source.replace(old_usage, new_usage)

            # Update cell source (preserve newlines)
            cell['source'] = fixed_source.splitlines(keepends=True)

            # Clear outputs
            cell['outputs'] = []
            cell['execution_count'] = None

            fixed = True
            print(f"Fixed logging import in cell {i}")
            break

    if not fixed:
        print("Warning: Could not find cell with incorrect import")
        return

    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Saved fixed notebook to {notebook_path}")


if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main.ipynb')

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    fix_logging_import(notebook_path)
    print("\nFix completed successfully!")
    print("\nChanged:")
    print("  OLD: from finance_ml.logging_config import setup_logger")
    print("       logger = setup_logger(__name__, level=logging.INFO)")
    print("\n  NEW: from finance_ml.logging_config import configure_logging, get_logger")
    print("       configure_logging(level=logging.INFO, console=True)")
    print("       logger = get_logger(__name__)")
