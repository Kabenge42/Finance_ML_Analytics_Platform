"""
Script to add feature leakage validation cell to notebook.
Inserts validation after Section 6.2 (prepare_regression_data call).
"""

import nbformat

# Read notebook
nb = nbformat.read("ml_finance_model_main.ipynb", as_version=4)

# Create validation cell content
validation_code = """# Feature Leakage Prevention Check (Priority 1 - Task 1.1)
print("\\n" + "=" * 80)
print("🔍 Feature Leakage Prevention Check")
print("=" * 80)

# Verify no market cap leakage
leakage_cols = [
    col for col in X_train.columns 
    if 'market_cap' in col.lower()
]

if leakage_cols:
    raise ValueError(f"⚠️ FEATURE LEAKAGE DETECTED: {leakage_cols}")
else:
    print("✓ No market_cap feature leakage detected")

# Log feature statistics
print(f"\\n📊 Feature Statistics:")
print(f"  Total features: {X_train.shape[1]}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# Show sector interaction features added
sector_interaction_cols = [
    col for col in X_train.columns 
    if 'sector_' in col and '__x__' in col
]
print(f"  Sector interactions: {len(sector_interaction_cols)}")

# Verify debt_to_equity is included (replacement for market_cap)
debt_cols = [col for col in X_train.columns if 'debt_to_equity' in col.lower()]
if debt_cols:
    print(f"✓ debt_to_equity included ({len(debt_cols)} features)")
else:
    print("⚠️ Warning: debt_to_equity not found in features")

# Verify enterprise_value is allowed (forward-looking metric)
ev_cols = [col for col in X_train.columns if 'enterprise_value' in col.lower()]
if ev_cols:
    print(f"✓ enterprise_value allowed ({len(ev_cols)} features)")

print("\\n✓ Feature leakage validation passed")
"""

# Create new cell
new_cell = nbformat.v4.new_code_cell(source=validation_code)

# Insert after cell 82 (which contains prepare_regression_data call)
insertion_index = 83
nb.cells.insert(insertion_index, new_cell)

# Write updated notebook
nbformat.write(nb, "ml_finance_model_main.ipynb")

print(f"✓ Added feature leakage validation cell at position {insertion_index}")
print(f"  Total cells: {len(nb.cells)}")
print(f"  Validation cell checks for market_cap leakage after prepare_regression_data()")
