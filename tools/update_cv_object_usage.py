"""
Update Cell 74 to use cv_object from determine_cv_strategy().
Priority 2 - Task 2.2: Integrate determine_cv_strategy() in notebook
"""

import nbformat

# Read notebook
nb = nbformat.read("ml_finance_model_main.ipynb", as_version=4)

# Get Cell 74 source
cell_74_source = nb.cells[74]["source"]

# Replace hardcoded cv=5, stratify_by='sector' with cv_object
# The cross_validate_classifier function should accept cv parameter directly
updated_source = cell_74_source.replace(
    "cv=5, stratify_by='sector'",
    "cv=cv_object  # Use CV strategy from determine_cv_strategy() (Cell 63)",
)

# Add explanatory comment
updated_source = updated_source.replace(
    "    try:\n        cv_results = cross_validate_classifier(",
    "    try:\n        # Use cv_object from determine_cv_strategy() (Cell 63) for proper CV policy\n        cv_results = cross_validate_classifier(",
)

# Update the cell
nb.cells[74]["source"] = updated_source

# Write updated notebook
nbformat.write(nb, "ml_finance_model_main.ipynb")

print("✓ Updated Cell 74 to use cv_object from determine_cv_strategy()")
print("  Changed: cv=5, stratify_by='sector' → cv=cv_object")
print("  Added comment referencing Cell 63 CV strategy")
