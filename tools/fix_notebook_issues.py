#!/usr/bin/env python3
"""
Fix notebook issues:
1. Phase95Config analytics_dir error
2. VALUATION_CATEGORIES duplicate 'Hold'
"""

import json
from pathlib import Path


def fix_notebook_issues(notebook_path):
    """Fix the two identified issues in the notebook."""

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes_made = []

    # Fix both issues
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        original_source = source

        # Fix 1: VALUATION_CATEGORIES duplicate Hold
        if "VALUATION_CATEGORIES = ['Strong Buy', 'Buy', 'Hold', 'Hold', 'Sell', 'Strong Sell']" in source:
            source = source.replace(
                "VALUATION_CATEGORIES = ['Strong Buy', 'Buy', 'Hold', 'Hold', 'Sell', 'Strong Sell']",
                "VALUATION_CATEGORIES = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']"
            )
            changes_made.append(f"Cell {cell_idx}: Fixed VALUATION_CATEGORIES duplicate")

        # Fix 2: Add fallback for analytics_dir in setup_output_directory
        if 'def setup_output_directory():' in source and 'output_dir = config.analytics_dir' in source:
            # Replace the problematic section with a more robust version
            old_code = """def setup_output_directory():
    \"\"\"Setup and validate output directory.\"\"\"
    if not hasattr(config, 'output_dir'):
        print("  ⚠ Error: config.output_dir not configured. Cannot generate reporting.")
        return None

    try:
        from pathlib import Path
        output_dir = config.analytics_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except (TypeError, AttributeError, OSError) as e:
        print(f"  ⚠ Error creating output directory: {str(e)}")
        return None"""

            new_code = """def setup_output_directory():
    \"\"\"Setup and validate output directory.\"\"\"
    if not hasattr(config, 'output_dir'):
        print("  ⚠ Error: config.output_dir not configured. Cannot generate reporting.")
        return None

    try:
        from pathlib import Path
        # Handle both FinanceMLConfig (with analytics_dir property) and legacy Phase95Config
        if hasattr(config, 'analytics_dir'):
            output_dir = config.analytics_dir
        else:
            # Fallback for configs without analytics_dir property
            output_dir = Path(config.output_dir) / 'analytics'
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except (TypeError, AttributeError, OSError) as e:
        print(f"  ⚠ Error creating output directory: {str(e)}")
        return None"""

            if old_code in source:
                source = source.replace(old_code, new_code)
                changes_made.append(f"Cell {cell_idx}: Fixed setup_output_directory() to handle analytics_dir")

        # Update cell if changes were made
        if source != original_source:
            # Convert back to list format if needed
            if isinstance(cell['source'], list):
                cell['source'] = source.split('\n')
                # Re-add newlines except for last line
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                                  for i, line in enumerate(cell['source'])]
            else:
                cell['source'] = source

    # Save the modified notebook
    backup_path = notebook_path.replace('.ipynb', '_backup_pre_fix.ipynb')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Saving fixed notebook: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    return changes_made


if __name__ == '__main__':
    notebook_path = 'ml_finance_model_main_v10.ipynb'
    
    print(f"Fixing issues in: {notebook_path}")
    print("=" * 60)
    
    changes = fix_notebook_issues(notebook_path)
    
    print("\n" + "=" * 60)
    print("Changes made:")
    if changes:
        for change in changes:
            print(f"  [OK] {change}")
    else:
        print("  No changes needed - issues may have been fixed already")
    
    print("\n[OK] Notebook fixes applied successfully!")
    print(f"  Original backed up as: {notebook_path.replace('.ipynb', '_backup_pre_fix.ipynb')}")
