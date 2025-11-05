"""
Script to insert Section 9.1.8 into ml_finance_model_main.ipynb
"""

import json
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

# Load notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Find Phase 9.2 cell index
phase92_idx = next(
    i
    for i, cell in enumerate(nb.cells)
    if cell.cell_type == "markdown" and "Phase 9.2" in cell.source
)

print(f"Found Phase 9.2 at index {phase92_idx}")

# Create markdown cell content
markdown_content = """### 9.1.8 Enhanced 4-Step Imputation Strategy (Phase 9.1 Complete)

Complete imputation pipeline ensuring zero missing values:
1. **Step 1: Zero Imputation** (48 columns) - Exceptional events (impairments, restructuring)
2. **Step 2: KNN Imputation** (148 columns) - Sector-aware financial metrics
3. **Step 3: Price Imputation** (5 columns) - Price targets from last_price
4. **Step 4: Median Imputation** (remaining) - Fallback for all other numerical columns"""

# Read code content from prepared file
with open("notebook_section_9_1_8.txt", "r", encoding="utf-8") as f:
    content = f.read()
    # Extract code after the marker
    code_content = content.split("---CODE_CELL---\n")[1].strip()

# Create new cells
markdown_cell = new_markdown_cell(markdown_content)
code_cell = new_code_cell(code_content)

# Insert cells before Phase 9.2
nb.cells.insert(phase92_idx, code_cell)
nb.cells.insert(phase92_idx, markdown_cell)

# Save notebook
with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✓ Section 9.1.8 inserted at index {phase92_idx}")
print(f"✓ Total cells: {len(nb.cells)}")
print("✓ Notebook saved successfully!")
