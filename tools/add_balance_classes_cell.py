"""
Add balance_classes() integration cell to notebook after Cell 59.
Priority 2 - Task 2.1: Integrate balance_classes() in notebook
"""

import nbformat

# Read notebook
nb = nbformat.read("ml_finance_model_main.ipynb", as_version=4)

# Create new cell with balance_classes integration
new_cell_code = """#%% Phase 9.4: Class Balance Analysis and Adjustment
print("\\n🔧 Phase 9.4: Class Balance Analysis and Adjustment")
print("=" * 80)

# Analyze class distribution
y_train_dist = pd.Series(y_train_cls).value_counts().sort_index()
print(f"\\n📊 Original Class Distribution:")
for cls, count in y_train_dist.items():
    pct = count / len(y_train_cls) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

# Calculate imbalance ratio
max_count = y_train_dist.max()
min_count = y_train_dist.min()
imbalance_ratio = max_count / min_count
print(f"\\n  Imbalance ratio: {imbalance_ratio:.2f}:1")

# Apply balancing (SMOTE for minority, undersample majority)
print(f"\\n  Applying balance_classes() with method='auto'...")
X_train_cls_balanced, y_train_cls_balanced = balance_classes(
    X_train_cls, 
    y_train_cls,
    method='auto',  # Auto-selects SMOTE or undersampling
    min_samples=MIN_SECTOR_SAMPLES,
    random_state=RANDOM_SEED
)

y_balanced_dist = pd.Series(y_train_cls_balanced).value_counts().sort_index()
print(f"\\n✓ Balanced Class Distribution:")
for cls, count in y_balanced_dist.items():
    pct = count / len(y_train_cls_balanced) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

# Calculate new imbalance ratio
max_count_after = y_balanced_dist.max()
min_count_after = y_balanced_dist.min()
imbalance_ratio_after = max_count_after / min_count_after

print(f"\\n  Resampling: {len(X_train_cls):,} → {len(X_train_cls_balanced):,} samples")
print(f"  Imbalance improvement: {imbalance_ratio:.2f}:1 → {imbalance_ratio_after:.2f}:1")
print(f"\\n✓ Training data balanced and ready for model training")
"""

# Create new code cell
new_cell = nbformat.v4.new_code_cell(new_cell_code)

# Insert after Cell 59 (classification data prep)
insert_position = 60
nb.cells.insert(insert_position, new_cell)

# Write updated notebook
nbformat.write(nb, "ml_finance_model_main.ipynb")

print(f"✓ Inserted balance_classes cell at position {insert_position}")
print(f"  Total cells: {len(nb.cells)}")
print(f"  Cell 59: Classification Data Preparation")
print(f"  Cell 60: Class Balance Analysis and Adjustment (NEW)")
print(f"  Cell 61: (previous Cell 60 content)")
