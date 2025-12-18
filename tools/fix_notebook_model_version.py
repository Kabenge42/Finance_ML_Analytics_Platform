#!/usr/bin/env python3
"""
Fix ml_finance_model_main.ipynb by adding MODEL_VERSION constant.

This script:
1. Parses the notebook JSON
2. Adds MODEL_VERSION constant after RANDOM_SEED in the configuration cell
3. Updates validate_configuration to print MODEL_VERSION
4. Writes the modified notebook back
"""
import json
import sys

NOTEBOOK_PATH = r'C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main.ipynb'

def main():
    # Read the notebook
    print(f"Reading notebook from: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Get first cell (configuration cell)
    first_cell = notebook['cells'][0]
    if first_cell['cell_type'] != 'code':
        print(f"ERROR: First cell is not a code cell, it's: {first_cell['cell_type']}")
        sys.exit(1)

    source_lines = first_cell['source']
    print(f"Original cell has {len(source_lines)} lines")

    # Find line 40 which contains RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
    random_seed_line_idx = None
    for idx, line in enumerate(source_lines):
        if 'RANDOM_SEED = int(os.getenv' in line:
            random_seed_line_idx = idx
            print(f"Found RANDOM_SEED at line index {idx}: {line.strip()}")
            break

    if random_seed_line_idx is None:
        print("ERROR: Could not find RANDOM_SEED line")
        sys.exit(1)

    # Check if MODEL_VERSION already exists
    has_model_version = any('MODEL_VERSION' in line and '=' in line and 'os.getenv' in line for line in source_lines)
    if has_model_version:
        print("MODEL_VERSION constant already exists in the cell")
        return

    # Insert MODEL_VERSION after the np.random.seed line (which is after RANDOM_SEED)
    # Line 40: RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
    # Line 41: np.random.seed(RANDOM_SEED)
    # Line 42: (blank line)
    # Insert after line 41
    insert_idx = random_seed_line_idx + 2  # After np.random.seed line

    # Verify we're at the right place
    print(f"Inserting MODEL_VERSION after line {insert_idx-1}: {source_lines[insert_idx-1].strip()}")

    # Create the new line
    model_version_line = "MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10')\n"

    # Insert the new line
    source_lines.insert(insert_idx, model_version_line)
    print(f"Added: {model_version_line.strip()}")

    # Now update validate_configuration to include MODEL_VERSION in the print statement
    # Find the line that prints RANDOM_SEED
    # Looking for: print(f"  RANDOM_SEED: {RANDOM_SEED}")
    print("\nSearching for validation print statement...")
    for idx, line in enumerate(source_lines):
        if 'print' in line and 'RANDOM_SEED' in line and 'Configuration validated' not in line:
            print(f"Found RANDOM_SEED print at line {idx}: {line.strip()}")
            # Add MODEL_VERSION print after this line
            indent = '    '  # Standard 4-space indent
            model_version_print = f"{indent}print(f\"  MODEL_VERSION: {{MODEL_VERSION}}\")\n"
            source_lines.insert(idx + 1, model_version_print)
            print(f"Added: {model_version_print.strip()}")
            break

    # Update the cell source
    first_cell['source'] = source_lines
    print(f"\nUpdated cell now has {len(source_lines)} lines")

    # Write back the notebook
    print(f"\nWriting modified notebook to: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print("SUCCESS: Notebook updated successfully!")
    print("\nChanges made:")
    print("1. Added MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10') after RANDOM_SEED")
    print("2. Added MODEL_VERSION to validation print output")

if __name__ == '__main__':
    main()
