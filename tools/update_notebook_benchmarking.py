"""
Update the benchmarking section in ml_finance_model_main.ipynb
"""

import json
from pathlib import Path
import shutil

# Backup original notebook
nb_path = Path("ml_finance_model_main.ipynb")
backup_path = nb_path.with_suffix(".ipynb.backup_benchmarking")
shutil.copy2(nb_path, backup_path)
print(f"✓ Created backup: {backup_path}")

# Load notebook
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Updated markdown for Cell 42
new_markdown = """### Phase 9.3 Enhanced EDA - Benchmarking Analysis

**Data Source:** `preprocessed_stocks_metadata.json` (Phase 9.1 output)

This section analyzes the preprocessed data and reports which Phase 9.3 feature categories are available. Note that most Phase 9.3 engineered features (momentum, quality scores, composite metrics, etc.) require running `build_comprehensive_features()` during the feature engineering step.

**Current Analysis:**
- Uses actual columns from preprocessed metadata catalog (351 columns)
- Reports accurate coverage by Phase 9.3 category
- Identifies which features are present vs. require feature engineering
- Aligns with code_guidelines.md Section 2.1 variable mapping standards
"""

# Load corrected benchmarking code
corrected_code_path = Path("corrected_benchmarking_cell.py")
with open(corrected_code_path, "r", encoding="utf-8") as f:
    corrected_code = f.read()

# Remove the docstring (first 4 lines)
lines = corrected_code.split("\n")
corrected_code_clean = "\n".join(lines[4:])

# Update Cell 42 (markdown)
if 42 < len(nb["cells"]) and nb["cells"][42]["cell_type"] == "markdown":
    nb["cells"][42]["source"] = new_markdown
    print(f"✓ Updated Cell 42 (markdown)")
else:
    print(f"⚠️  Cell 42 not found or not markdown type")

# Update Cell 43 (code)
if 43 < len(nb["cells"]) and nb["cells"][43]["cell_type"] == "code":
    # Convert code to list of lines for notebook format
    code_lines = corrected_code_clean.split("\n")
    # Add newline to each line except the last
    nb["cells"][43]["source"] = [
        line + "\n" if i < len(code_lines) - 1 else line for i, line in enumerate(code_lines)
    ]
    print(f"✓ Updated Cell 43 (code) with corrected benchmarking implementation")
else:
    print(f"⚠️  Cell 43 not found or not code type")

# Save updated notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Notebook updated: {nb_path}")
print(f"\nChanges made:")
print(f"  - Cell 42: Updated markdown description")
print(f"  - Cell 43: Replaced with corrected benchmarking code")
print(f"  - Uses preprocessed_stocks_metadata.json directly")
print(f"  - Reports accurate Phase 9.3 category coverage")
print(f"  - Aligns with code_guidelines.md standards")
print(f"\nBackup saved at: {backup_path}")
