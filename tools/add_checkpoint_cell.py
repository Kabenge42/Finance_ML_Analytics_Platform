#!/usr/bin/env python3
"""
Add missing regression_complete checkpoint after Phase 9.5 in notebook.

This script inserts a new code cell after cell 140 that sets the regression_complete
checkpoint, which is required by Phase 9.5.1 (Model Optimization Enhancements).
"""
import json
import sys
from pathlib import Path


def add_checkpoint_cell(notebook_path: str, backup: bool = True):
    """
    Add checkpoint cell after Phase 9.5 completion.

    Args:
        notebook_path: Path to notebook file
        backup: Whether to create backup before modifying
    """
    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Create backup if requested
    if backup:
        backup_path = notebook_path.replace(".ipynb", "_backup_before_checkpoint.ipynb")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"[OK] Backup created: {backup_path}")

    # Define the checkpoint cell
    checkpoint_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "#%%\n",
            "# ============================================================================\n",
            "# CHECKPOINT: PHASE 9.5 REGRESSION COMPLETE\n",
            "# ============================================================================\n",
            "# Mark regression modeling complete for downstream phases\n",
            "# This checkpoint is required by:\n",
            "#   - Phase 9.5.1: Model Optimization Enhancements\n",
            "#   - Phase 9.6: Model Evaluation and Error Analysis  \n",
            "#   - Phase 9.7: Stock Valuation and Identification\n",
            "# ============================================================================\n",
            "\n",
            'checkpoint("regression_complete", requires=["classification_complete"])\n',
            'print("[OK] Checkpoint: regression_complete")\n',
            'print("  Phase 9.5 regression models ready for optimization and evaluation")\n',
        ],
    }

    # Insert after cell 140 (which is Phase 9.5 main implementation)
    insert_position = 141
    nb["cells"].insert(insert_position, checkpoint_cell)

    print(f"\n[OK] Checkpoint cell inserted at position {insert_position}")
    print(f"  Total cells before: {len(nb['cells']) - 1}")
    print(f"  Total cells after: {len(nb['cells'])}")

    # Write modified notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[OK] Notebook updated: {notebook_path}")
    print("\n" + "=" * 80)
    print("CHECKPOINT FIX APPLIED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Restart Jupyter kernel: Kernel -> Restart Kernel")
    print("2. Run cells sequentially from the beginning")
    print("3. Verify checkpoint appears: '[OK] Checkpoint: regression_complete'")
    print("4. Phase 9.5.1 (cell 143) should now execute without errors")

    return True


if __name__ == "__main__":
    notebook_path = "ml_finance_model_main_backup.ipynb"

    if not Path(notebook_path).exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        sys.exit(1)

    try:
        add_checkpoint_cell(notebook_path, backup=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
