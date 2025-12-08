#!/usr/bin/env python3
"""
Fix TypeError in Phase 9.5: Tree-Based Models Comparison
Issue: get_r2_score function not properly extracting numeric R² scores
"""

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("ml_finance_model_main2_0.ipynb")


def find_phase95_cell(cells):
    """Find the Phase 9.5 cell that needs fixing."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "PHASE 9.5: TREE-BASED MODELS COMPARISON" in source:
                return i, cell
    return None, None


def fix_get_r2_score_function():
    """
    Generate fixed get_r2_score function aligned with code_guidelines.md Section 7.1.

    Per guidelines, training functions return dict with structure:
    {
        "model": fitted_estimator,
        "metrics": Dict[str, float],  # Contains 'r2', 'r2_score', 'mae', 'rmse'
        "y_pred": array_like,
        "artifacts": Optional[Dict]
    }
    """
    return '''    def get_r2_score(item):
        """
        Safely extract R² score from various data structures.
        
        Handles training function return formats per code_guidelines.md Section 7.1:
        - Dict with 'metrics' sub-dict containing 'r2' or 'r2_score'
        - Dict with direct 'r2_score' key (legacy format)
        - Tuple from dict.items() -> (key, value)
        - Direct numeric values
        
        Args:
            item: Result from training function or (key, value) tuple
            
        Returns:
            float: Extracted R² score, defaults to 0.0 if not found
        """
        # Handle tuple from dict.items()
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            value = item[1]
        else:
            value = item
        
        # Case 1: Standard format - dict with 'metrics' sub-dict
        if isinstance(value, dict):
            # Check for metrics sub-dict (standard per Section 7.1)
            if 'metrics' in value and isinstance(value['metrics'], dict):
                metrics = value['metrics']
                # Try 'r2_score' first, then 'r2'
                return metrics.get('r2_score', metrics.get('r2', 0.0))
            
            # Case 2: Legacy format - direct 'r2_score' key
            if 'r2_score' in value:
                return value['r2_score']
            
            # Case 3: Direct 'r2' key
            if 'r2' in value:
                return value['r2']
        
        # Case 4: Tuple/list format (assume first element is r2_score)
        elif isinstance(value, (tuple, list)) and len(value) > 0:
            first_elem = value[0]
            if isinstance(first_elem, (int, float)):
                return first_elem
        
        # Case 5: Direct numeric value
        elif isinstance(value, (int, float)):
            return value
        
        # Default: return 0.0 to allow comparison to continue
        return 0.0
'''


def create_fixed_cell_source(original_source):
    """Create fixed cell source with improved get_r2_score function."""
    lines = original_source.split("\n")

    # Find the function definition
    func_start = None
    func_end = None

    for i, line in enumerate(lines):
        if "def get_r2_score(item):" in line:
            func_start = i
        elif func_start is not None and func_end is None:
            # Find end of function (next def, or code at same indentation level)
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#") and not line.startswith("    "):
                func_end = i
                break
            # Also check for explicit function boundaries
            if i > func_start and "return 0" in line and line.startswith("        return"):
                func_end = i + 1
                break

    if func_start is None:
        print("ERROR: Could not find get_r2_score function")
        return None

    if func_end is None:
        func_end = func_start + 20  # Estimate

    # Replace the function
    new_lines = lines[:func_start] + [fix_get_r2_score_function()] + lines[func_end:]

    return "\n".join(new_lines)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Fixing TypeError in Phase 9.5: Tree-Based Models Comparison")
    print("=" * 80)

    # Load notebook
    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: Notebook not found: {NOTEBOOK_PATH}")
        return 1

    print(f"\n[*] Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find Phase 9.5 cell
    cell_idx, cell = find_phase95_cell(notebook["cells"])

    if cell is None:
        print("ERROR: Could not find Phase 9.5 cell")
        return 1

    print(f"[+] Found Phase 9.5 cell at index {cell_idx}")

    # Get original source
    original_source = "".join(cell.get("source", []))

    # Create fixed source
    print("\n[*] Creating fixed get_r2_score function...")
    fixed_source = create_fixed_cell_source(original_source)

    if fixed_source is None:
        return 1

    # Update cell
    notebook["cells"][cell_idx]["source"] = fixed_source.split("\n")

    # Create backup
    backup_path = NOTEBOOK_PATH.with_suffix(".ipynb.backup_phase95_fix")
    print(f"\n[*] Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(json.load(open(NOTEBOOK_PATH, "r", encoding="utf-8")), f, indent=1)

    # Save updated notebook
    print(f"[*] Saving fixed notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[SUCCESS] Fix Complete!")
    print("=" * 80)
    print("\nChanges made:")
    print("  - Enhanced get_r2_score() to handle standard training function format")
    print("  - Added support for 'metrics' sub-dict per code_guidelines.md Section 7.1")
    print("  - Improved error handling with proper default return values")
    print("  - Added comprehensive docstring")
    print("\nThe function now properly extracts R² scores from:")
    print("  1. Standard format: dict['metrics']['r2_score'] or dict['metrics']['r2']")
    print("  2. Legacy format: dict['r2_score']")
    print("  3. Tuple/list format: (key, dict) pairs from dict.items()")
    print("  4. Direct numeric values")
    print("\n[!] Before running the notebook:")
    print("  1. Review the changes in the Phase 9.5 cell")
    print("  2. Ensure training functions return the standard format")
    print(f"  3. Backup saved at: {backup_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
