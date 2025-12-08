#!/usr/bin/env python3
"""
Fix and refactor Phase 9.5 cell with proper syntax and code guidelines alignment.
Fixes: Syntax errors, indentation, and aligns with code_guidelines.md Section 7.1 and 8
"""

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("ml_finance_model_main2_0.ipynb")


def create_refactored_cell():
    """Generate properly formatted Phase 9.5 cell."""
    return '''# ============================================================================
# PHASE 9.5: Tree-Based Models Comparison
# ============================================================================
print("=" * 80)
print("PHASE 9.5: TREE-BASED MODELS COMPARISON")
print("=" * 80)

if 'X_train' not in dir() or 'y_train' not in dir():
    print("⚠️  Training data not prepared. Run Section 6.2 first.")
else:
    from finance_ml.ml_workflow.regression.models import (
        train_random_forest_regressor,
        train_extra_trees_regressor,
    )
    
    tree_results = {}
    
    
    def get_r2_score(item):
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
    
    
    # 1. Random Forest
    print("\\n🌲 Training Random Forest Regressor...")
    try:
        rf_result = train_random_forest_regressor(
            X_train, y_train,
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED
        )
        tree_results['RandomForest'] = rf_result
        score = get_r2_score(rf_result)
        print(f"   R² Score: {score:.4f}")
        if isinstance(rf_result, dict):
            # Check both locations for MAE
            mae = rf_result.get('mae')
            if mae is None and 'metrics' in rf_result:
                mae = rf_result['metrics'].get('mae')
            if mae is not None:
                print(f"   MAE: {mae:.4f}")
    except Exception as e:
        print(f"   ⚠️ RandomForest failed: {e}")
    
    # 2. Extra Trees
    print("\\n🌲 Training Extra Trees Regressor...")
    try:
        et_result = train_extra_trees_regressor(
            X_train, y_train,
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED
        )
        tree_results['ExtraTrees'] = et_result
        score = get_r2_score(et_result)
        print(f"   R² Score: {score:.4f}")
        if isinstance(et_result, dict):
            # Check both locations for MAE
            mae = et_result.get('mae')
            if mae is None and 'metrics' in et_result:
                mae = et_result['metrics'].get('mae')
            if mae is not None:
                print(f"   MAE: {mae:.4f}")
    except Exception as e:
        print(f"   ⚠️ ExtraTrees failed: {e}")
    
    # Feature importance from best tree model
    if tree_results:
        best_tree = max(tree_results.items(), key=get_r2_score)
        r2_value = get_r2_score(best_tree)
        print(f"\\n📊 Best Tree Model: {best_tree[0]} (R² = {r2_value:.4f})")
        
        best_tree_data = best_tree[1]
        if isinstance(best_tree_data, dict) and 'feature_importance' in best_tree_data:
            print("\\n📊 Top 10 Feature Importances:")
            fi = best_tree_data['feature_importance']
            if hasattr(fi, 'items'):
                sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                for feat, imp in sorted_fi:
                    print(f"   {feat:30s}: {imp:.4f}")
    
    print("\\n✓ Tree-based models comparison complete")
'''


def find_phase95_cell(cells):
    """Find the Phase 9.5 cell that needs fixing."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "PHASE 9.5: TREE-BASED MODELS COMPARISON" in source:
                return i, cell
    return None, None


def main():
    """Main execution function."""
    print("=" * 80)
    print("Refactoring Phase 9.5: Tree-Based Models Comparison")
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

    # Create refactored cell
    print("\n[*] Creating refactored cell with proper syntax and formatting...")
    refactored_source = create_refactored_cell()

    # Update cell source
    notebook["cells"][cell_idx]["source"] = refactored_source.split("\n")

    # Create backup
    backup_path = NOTEBOOK_PATH.with_suffix(".ipynb.backup_phase95_refactor")
    print(f"\n[*] Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(json.load(open(NOTEBOOK_PATH, "r", encoding="utf-8")), f, indent=1)

    # Save updated notebook
    print(f"[*] Saving refactored notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[SUCCESS] Refactoring Complete!")
    print("=" * 80)
    print("\nFixes applied:")
    print("  ✓ Fixed missing newline after 'else:'")
    print("  ✓ Properly formatted get_r2_score() function")
    print("  ✓ Fixed indentation throughout")
    print("  ✓ Separated function definition from code")
    print("  ✓ Improved MAE extraction logic (checks both dict locations)")
    print("  ✓ Added proper docstring formatting")
    print("  ✓ Aligned with code_guidelines.md Section 7.1 and 8")
    print("\nKey improvements:")
    print("  - Function handles standard {'metrics': {'r2': value}} format")
    print("  - Function handles legacy {'r2_score': value} format")
    print("  - MAE extraction checks both direct and nested locations")
    print("  - Proper PEP 8 compliant formatting")
    print("  - Clear separation of concerns")
    print(f"\n[!] Backup saved at: {backup_path}")
    print("\n[!] Next steps:")
    print("  1. Open ml_finance_model_main2_0.ipynb in Jupyter/PyCharm")
    print("  2. Navigate to Phase 9.5 cell")
    print("  3. Run the cell to verify no syntax errors")
    print("  4. Check that models train and best model is selected correctly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
