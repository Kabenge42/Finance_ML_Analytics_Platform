#!/usr/bin/env python3
"""
Update ml_finance_model_main.ipynb cells 42-43 with refactored benchmarking
that analyzes all_stocks_features DataFrame instead of metadata JSON.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# File paths
notebook_path = Path("ml_finance_model_main.ipynb")
backup_path = Path(
    f"ml_finance_model_main.ipynb.backup_refactored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
markdown_path = Path("refactored_benchmarking_markdown.md")
code_path = Path("refactored_benchmarking_cell.py")

# Load markdown and code content
markdown_content = markdown_path.read_text(encoding="utf-8")
code_content = code_path.read_text(encoding="utf-8")

# Load notebook
print(f"Loading notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Backup original
print(f"Creating backup: {backup_path}")
shutil.copy2(notebook_path, backup_path)

# Update Cell 42 (markdown)
print("\nUpdating Cell 42 (markdown)...")
if len(nb["cells"]) > 42:
    cell_42 = nb["cells"][42]
    print(f"  Original cell type: {cell_42.get('cell_type', 'unknown')}")

    # Update to markdown cell with new content
    cell_42["cell_type"] = "markdown"
    cell_42["source"] = markdown_content
    cell_42["metadata"] = cell_42.get("metadata", {})

    print(f"  ✓ Cell 42 updated with refactored markdown")
    print(f"    Content preview: {markdown_content[:100]}...")
else:
    print(f"  ✗ ERROR: Cell 42 not found (notebook has {len(nb['cells'])} cells)")
    exit(1)

# Update Cell 43 (code)
print("\nUpdating Cell 43 (code)...")
if len(nb["cells"]) > 43:
    cell_43 = nb["cells"][43]
    print(f"  Original cell type: {cell_43.get('cell_type', 'unknown')}")

    # Update to code cell with new content
    cell_43["cell_type"] = "code"
    cell_43["source"] = code_content
    cell_43["metadata"] = cell_43.get("metadata", {})
    cell_43["outputs"] = []  # Clear outputs
    cell_43["execution_count"] = None  # Clear execution count

    print(f"  ✓ Cell 43 updated with refactored code")
    print(f"    Code length: {len(code_content)} characters")
    print(f"    Lines: {len(code_content.splitlines())}")
else:
    print(f"  ✗ ERROR: Cell 43 not found (notebook has {len(nb['cells'])} cells)")
    exit(1)

# Save updated notebook
print(f"\nSaving updated notebook: {notebook_path}")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "=" * 80)
print("✓ NOTEBOOK UPDATE COMPLETE")
print("=" * 80)
print(f"\nChanges made:")
print(f"  • Cell 42: Updated markdown to explain post-feature-engineering analysis")
print(f"  • Cell 43: Refactored code to analyze all_stocks_features DataFrame")
print(f"\nKey improvements:")
print(f"  1. Analyzes engineered features instead of preprocessed metadata")
print(f"  2. Uses phase93_categories module for accurate feature detection")
print(f"  3. Shows actual Phase 9.3 coverage after feature engineering")
print(f"  4. Validates DataFrame exists before running")
print(f"  5. Exports to phase93_benchmarking_post_engineering.json")
print(f"\nBackup saved to: {backup_path}")
print(f"\nNext steps:")
print(f"  1. Review updated cells 42-43 in the notebook")
print(f"  2. Execute Phase 9.3 feature engineering cells (33-40)")
print(f"  3. Execute updated benchmarking cells (42-43)")
print(f"  4. Verify proper Phase 9.3 coverage reporting")
print("=" * 80)
