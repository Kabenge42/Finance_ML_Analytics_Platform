"""Fix snapshot_date references in notebook Section 10.2"""
import json
import sys

def fix_notebook():
    """Replace snapshot_date with last_updated in Cell 107"""

    # Read notebook
    with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find and fix Cell 107
    cell = nb['cells'][107]

    # Get source as string
    if isinstance(cell['source'], list):
        source_lines = cell['source']
    else:
        source_lines = [cell['source']]

    # Process each line
    fixed_lines = []
    comment_added = False
    for line in source_lines:
        # Replace snapshot_date with last_updated
        fixed_line = line.replace('snapshot_date', 'last_updated')

        # Add comment before the required_cols line
        if "required_cols = ['ticker', 'last_updated', 'last_price']" in fixed_line and not comment_added:
            # Get indentation
            indent = len(fixed_line) - len(fixed_line.lstrip())
            comment = ' ' * indent + '# Schema v1.3 uses \'last_updated\' as canonical date column (code_guidelines.md Section 2.2)\n'
            fixed_lines.append(comment)
            comment_added = True

        fixed_lines.append(fixed_line)

    # Update cell
    nb['cells'][107]['source'] = fixed_lines

    # Write back
    with open('ml_finance_model_main.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True

if __name__ == '__main__':
    try:
        if fix_notebook():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        sys.exit(1)
