"""Fix Cell 107 to have proper newline characters for Jupyter notebook format"""
import json

def fix_cell_newlines():
    """Add proper newline characters to each line in Cell 107"""

    # Read notebook
    with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Get Cell 107
    cell = nb['cells'][107]
    source_lines = cell['source']

    # Add newline to all lines except the last one
    fixed_lines = []
    for i, line in enumerate(source_lines):
        if i < len(source_lines) - 1:
            # Not the last line - ensure it ends with \n
            if not line.endswith('\n'):
                fixed_lines.append(line + '\n')
            else:
                fixed_lines.append(line)
        else:
            # Last line - should NOT end with \n
            if line.endswith('\n'):
                fixed_lines.append(line.rstrip('\n'))
            else:
                fixed_lines.append(line)

    # Update cell
    nb['cells'][107]['source'] = fixed_lines

    # Write back
    with open('ml_finance_model_main.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True

if __name__ == '__main__':
    try:
        if fix_cell_newlines():
            print("Cell 107 newlines fixed")
            print("- Added newline chars to lines 0-126")
            print("- Last line (127) has no trailing newline")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
