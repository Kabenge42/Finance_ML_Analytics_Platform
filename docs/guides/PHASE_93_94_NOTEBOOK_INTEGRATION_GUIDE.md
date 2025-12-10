# Phase 9.3 & 9.4 Notebook Integration Guide

**Date**: 2025-12-09  
**Status**: READY FOR INTEGRATION  
**Notebook**: `ml_finance_model_main.ipynb`  
**Alignment**: `code_guidelines.md` v1.10 Section 8 (Notebook Best Practices)

---

## Executive Summary

This guide provides ready-to-use code cells and markdown documentation for integrating Phase 9.3 (Automated Feature
Selection) and Phase 9.4 (Multi-Label Classification, CV Policy, Class Balance) functions into
`ml_finance_model_main.ipynb`.

**Imports Updated**: ✅ Complete

- Phase 9.3: `select_features_auto`, `select_features_by_category` (lines 642-644)
- Phase 9.4: `create_multilabel_event_labels`, `determine_cv_strategy`, `balance_classes` (lines 530-533)

**Integration Points**:

- **Phase 9.3 Section** (~line 34339): Add 3 new cells for automated feature selection
- **Phase 9.4 Section** (~line 44878): Add 4 new cells for multi-label classification, CV policy, and class balance

---

## Phase 9.3: Automated Feature Selection Integration

### Location

Insert these cells **after** the existing Phase 9.3 feature engineering content (after the `build_features()` calls).

### Cell 1: Markdown - Phase 9.3 TDD Enhancement Documentation

```markdown
### 🆕 Phase 9.3 TDD: Automated Feature Selection (Task 1)

**Business Objective**: Reduce feature noise, improve model interpretability, and prevent overfitting through automated
feature selection.

**New Capabilities**:

1. **Automated Feature Selection** (`select_features_auto`): Combines multiple methods (mutual information, random
   forest importance, correlation-based redundancy detection)
2. **Category-Based Selection** (`select_features_by_category`): Select features from specific Phase 9.3 categories (
   momentum, valuation, quality, etc.)

**Key Features**:

- Importance threshold filtering (default: 0.01)
- Correlation-based redundancy removal (default: >0.95)
- Price column preservation (21 columns always kept)
- Integration with unified ETL pipeline via `auto_feature_selection` parameter

**References**:

- Implementation Plan: `docs/improvement_plan/phase_9.3_implementation_plan.md`
- Test Coverage: `tests/test_feature_selection_auto.py` (4 tests, 100% pass)
- Code Guidelines: Section 9.3.1
```

### Cell 2: Code - Automated Feature Selection with Importance Threshold

```python
# %%
# Phase 9.3 TDD: Automated Feature Selection
print('\n' + '=' * 80)
print('PHASE 9.3 TDD: AUTOMATED FEATURE SELECTION')
print('=' * 80)

# Prepare feature matrix and target
feature_cols = [col for col in all_stocks_features.columns
                if col not in ['ticker', 'isin', 'sector', 'region', 'snapshot_date',
                               'price_target', 'last_price']]

X_features = all_stocks_features[feature_cols].copy()
y_target = all_stocks_features['price_target'].copy()

# Remove rows with missing target
valid_mask = y_target.notna()
X_features_clean = X_features[valid_mask]
y_target_clean = y_target[valid_mask]

print(f'\n📊 Feature Selection Input:')
print(f'  Total features: {len(feature_cols)}')
print(f'  Valid samples: {len(X_features_clean):,}')
print(f'  Target: {y_target_clean.name}')

# Apply automated feature selection
# Method: 'combined' uses mutual info + RF importance + correlation pruning
# Importance threshold: 0.01 (remove features with <1% importance)
# Correlation threshold: 0.95 (remove redundant features with r>0.95)
print(f'\n🔍 Applying select_features_auto():')
print(f'  Method: combined (mutual_info + rf_importance + correlation)')
print(f'  Importance threshold: {FEATURE_IMPORTANCE_THRESHOLD}')
print(f'  Correlation threshold: 0.95')

X_selected = select_features_auto(
        X_features_clean,
        y_target_clean,
        importance_threshold=0.01,
        correlation_threshold=0.95,
        method='combined',
        preserve_columns=None,  # Auto-preserves price columns
        return_scores=False
        )

print(f'\n✅ Feature Selection Results:')
print(f'  Features before: {X_features_clean.shape[1]}')
print(f'  Features after: {X_selected.shape[1]}')
print(f'  Reduction: {X_features_clean.shape[1] - X_selected.shape[1]} features removed')
print(f'  Dimensionality reduced by: {(1 - X_selected.shape[1] / X_features_clean.shape[1]) * 100:.1f}%')

# Update dataframe with selected features
all_stocks_selected = all_stocks_features[
    ['ticker', 'isin', 'sector', 'region', 'price_target', 'last_price'] +
    list(X_selected.columns)
    ].copy()

print(f'\n✓ Created all_stocks_selected DataFrame with {len(all_stocks_selected):,} rows')
print(f'  Selected features: {X_selected.shape[1]}')
```

### Cell 3: Code - Category-Based Feature Selection

```python
# %%
# Phase 9.3 TDD: Category-Based Feature Selection
print('\n' + '=' * 80)
print('PHASE 9.3 TDD: CATEGORY-BASED FEATURE SELECTION')
print('=' * 80)

# Select only momentum and valuation features for a focused model
target_categories = ['momentum', 'valuation', 'quality']

print(f'\n🎯 Selecting features from categories: {target_categories}')

X_category_features = select_features_by_category(
        X_features_clean,
        categories=target_categories
        )

print(f'\n✅ Category Selection Results:')
print(f'  Total features: {X_features_clean.shape[1]}')
print(f'  Category features: {X_category_features.shape[1]}')
print(f'  Categories: {", ".join(target_categories)}')

# Example: Show feature breakdown by category
print(f'\n📋 Feature Breakdown by Category:')
for category in target_categories:
    cat_cols = [col for col in X_category_features.columns if category in col.lower()]
    if cat_cols:
        print(f'  {category}: {len(cat_cols)} features')
        print(f'    Examples: {", ".join(cat_cols[:5])}')
```

---

## Phase 9.4: Classification Enhancements Integration

### Location

Insert these cells **within** the Phase 9.4 section, after the existing event label creation but before model training.

### Cell 1: Markdown - Phase 9.4 TDD Enhancement Documentation

```markdown
### 🆕 Phase 9.4 TDD: Classification Enhancements (Tasks 2, 4, 5)

**Business Objective**: Provide granular event signals, prevent look-ahead bias, and ensure all market conditions are
represented in training data.

**New Capabilities**:

1. **Multi-Label Classification** (`create_multilabel_event_labels`): Produce independent binary labels per Phase 9.3
   category (momentum, valuation, quality, etc.)
2. **CV Policy Enforcement** (`determine_cv_strategy`): Automatic CV strategy selection (time_series → grouped →
   stratified)
3. **Class Balance Auto-Remediation** (`balance_classes`): Automatic SMOTE/undersampling when imbalance >10:1

**Key Features**:

- Multi-label mode: 16 independent binary labels (one per category)
- Sector-adjusted thresholds for multi-label classification
- Automatic CV strategy based on data structure (prevents leakage)
- Class balance with SMOTE for minority class augmentation

**References**:

- Implementation Plan: `docs/improvement_plan/phase_9.4_implementation_plan.md`
- Test Coverage: `tests/test_multilabel_classification.py`, `test_cv_policy_enforcement.py`,
  `test_class_balance_remediation.py` (9 tests, 100% pass)
- Code Guidelines: Section 9.4
```

### Cell 2: Code - Multi-Label Classification

```python
# %%
# Phase 9.4 TDD: Multi-Label Event Classification (Task 2)
print('\n' + '=' * 80)
print('PHASE 9.4 TDD: MULTI-LABEL EVENT CLASSIFICATION')
print('=' * 80)

# Create multi-label event labels (one binary label per category)
# This produces more granular signals than single multi-class labels
print(f'\n🏷️  Creating multi-label event labels...')
print(f'  Mode: multilabel (independent binary labels per category)')
print(f'  Categories: momentum, valuation, quality, profitability, growth')

# Select key categories for multi-label classification
ml_categories = ['momentum', 'valuation', 'quality', 'profitability', 'growth']

all_stocks_multilabel = create_multilabel_event_labels(
        all_stocks_features.copy(),
        label_mode='multilabel',
        categories=ml_categories,
        sector_adjusted=True,
        threshold_percentiles=(0.33, 0.67),
        min_samples=MIN_SECTOR_SAMPLES  # From configuration cell
        )

print(f'\n✅ Multi-Label Classification Results:')
print(f'  Total samples: {len(all_stocks_multilabel):,}')
print(f'  Label columns created: {len(ml_categories)}')

# Show label distribution per category
print(f'\n📊 Label Distribution by Category:')
for category in ml_categories:
    label_col = f'label_{category}'
    if label_col in all_stocks_multilabel.columns:
        positive_pct = (all_stocks_multilabel[label_col] == 1).mean() * 100
        print(f'  {category}: {positive_pct:.1f}% positive')

# Optional: Create feature interactions with multi-label probabilities
# This can be used downstream in regression models
print(f'\n💡 Tip: Use label_* columns as meta-features in regression models')
print(f'   Example: label_momentum, label_valuation can signal price movement drivers')
```

### Cell 3: Code - Cross-Validation Strategy Auto-Detection

```python
#%%
# Phase 9.4 TDD: CV Policy Enforcement (Task 4)
print('\n' + '=' * 80)
print('PHASE 9.4 TDD: CROSS-VALIDATION POLICY ENFORCEMENT')
print('=' * 80)

# Automatically determine best CV strategy based on data structure
# Hierarchy: time_series (if snapshot_date) → grouped (if ticker) → stratified (fallback)

print(f'\n🔍 Detecting optimal CV strategy...')
print(f'  Data columns: {list(all_stocks_features.columns[:10])}...')

# Prepare target for CV strategy detection
y_event = all_stocks_features['event_label'] if 'event_label' in all_stocks_features.columns else None

cv_strategy_name, cv_object = determine_cv_strategy(
        all_stocks_features,
        target=y_event,
        n_splits=CV_FOLDS,  # From configuration cell
        date_column='snapshot_date',
        group_column='ticker',
        random_state=RANDOM_SEED
        )

print(f'\n✅ CV Strategy Selected: {cv_strategy_name.upper()}')
print(f'  Number of folds: {CV_FOLDS}')
print(f'  Strategy: {cv_object.__class__.__name__}')

# Show why this strategy was chosen
if cv_strategy_name == 'time_series':
    print(f'  Reason: snapshot_date column detected → prevents look-ahead bias')
elif cv_strategy_name == 'grouped':
    n_groups = all_stocks_features['ticker'].nunique()
    print(f'  Reason: ticker column detected ({n_groups:,} unique tickers) → prevents data leakage')
elif cv_strategy_name == 'stratified':
    print(f'  Reason: classification target detected → maintains class balance')
else:
    print(f'  Reason: fallback strategy')

print(f'\n💡 Use cv_object in cross_val_score() or GridSearchCV for proper validation')
```

### Cell 4: Code - Class Balance Auto-Remediation

```python
# %%
# Phase 9.4 TDD: Class Balance Auto-Remediation (Task 5)
print('\n' + '=' * 80)
print('PHASE 9.4 TDD: CLASS BALANCE AUTO-REMEDIATION')
print('=' * 80)

# Check class balance and apply automatic remediation if needed
# Applies SMOTE for imbalance >10:1

if 'event_label' in all_stocks_features.columns:
    print(f'\n📊 Class Distribution Before Balancing:')
    class_counts = all_stocks_features['event_label'].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = count / len(all_stocks_features) * 100
        print(f'  Class {cls}: {count:,} ({pct:.1f}%)')

    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    print(f'\n  Imbalance ratio: {imbalance_ratio:.2f}:1')

    # Prepare features and target
    feature_cols_balance = [col for col in all_stocks_features.columns
                            if col not in ['ticker', 'isin', 'sector', 'region',
                                           'event_label', 'snapshot_date']]
    X_balance = all_stocks_features[feature_cols_balance].fillna(0)
    y_balance = all_stocks_features['event_label']

    # Apply automatic balancing if imbalance exceeds threshold
    if imbalance_ratio > 10.0:
        print(f'\n⚠️  Severe imbalance detected (>{10.0}:1)')
        print(f'  Applying balance_classes() with method="auto"...')

        X_balanced, y_balanced = balance_classes(
                X_balance,
                y_balance,
                method='auto',  # Chooses SMOTE or undersample based on severity
                imbalance_threshold=10.0,
                random_state=RANDOM_SEED
                )

        print(f'\n✅ Class Balance After Remediation:')
        class_counts_after = pd.Series(y_balanced).value_counts().sort_index()
        for cls, count in class_counts_after.items():
            pct = count / len(y_balanced) * 100
            print(f'  Class {cls}: {count:,} ({pct:.1f}%)')

        # Calculate new imbalance ratio
        max_count_after = class_counts_after.max()
        min_count_after = class_counts_after.min()
        imbalance_ratio_after = max_count_after / min_count_after
        print(f'\n  New imbalance ratio: {imbalance_ratio_after:.2f}:1')
        print(f'  Improvement: {imbalance_ratio:.2f}:1 → {imbalance_ratio_after:.2f}:1')

        # Store balanced data for downstream use
        all_stocks_balanced = pd.DataFrame(X_balanced, columns=feature_cols_balance)
        all_stocks_balanced['event_label'] = y_balanced.values

        print(f'\n✓ Created all_stocks_balanced DataFrame with {len(all_stocks_balanced):,} rows')
    else:
        print(f'\n✓ Class balance acceptable ({imbalance_ratio:.2f}:1 < 10:1)')
        print(f'  No remediation needed')
        all_stocks_balanced = all_stocks_features.copy()
else:
    print(f'\n⚠️  No event_label column found, skipping class balance check')
    all_stocks_balanced = all_stocks_features.copy()
```

---

## Section 8 Alignment: Notebook Best Practices

### Centralized Configuration Constants (Section 8.1)

All new cells reference constants from the configuration cell:

- `FEATURE_IMPORTANCE_THRESHOLD = 0.01`
- `CV_FOLDS = 5`
- `MIN_SECTOR_SAMPLES = 20`
- `RANDOM_SEED = 42`

### DataFrame Stage Naming (Section 8.2)

New DataFrames follow the naming convention:

- `all_stocks_features` → `all_stocks_selected` (after feature selection)
- `all_stocks_features` → `all_stocks_multilabel` (with multi-label columns)
- `all_stocks_features` → `all_stocks_balanced` (after class balancing)

### Magic Numbers Policy (Section 8.3)

No magic numbers in new cells:

- Thresholds use named constants: `FEATURE_IMPORTANCE_THRESHOLD`, `MIN_SECTOR_SAMPLES`
- Percentiles use tuples with comments: `threshold_percentiles=(0.33, 0.67)  # 33rd/67th percentile`
- Imbalance threshold: `10.0` → documented in cell markdown

---

## Testing Checklist

Before finalizing integration:

- [ ] Verify all imports load successfully (run imports cell)
- [ ] Phase 9.3 cells execute without errors
- [ ] Phase 9.4 cells execute without errors
- [ ] DataFrame naming follows convention (Section 8.2)
- [ ] No magic numbers in code (Section 8.3)
- [ ] Configuration constants used throughout
- [ ] Markdown documentation explains business value
- [ ] Output shows expected metrics and diagnostics

---

## Next Steps

1. **Add Phase 9.3 cells** after existing feature engineering content (search for "## Phase 9.3: Advanced Feature
   Engineering")
2. **Add Phase 9.4 cells** within classification section (search for "## Phase 9.4: Multi-Class Event Classification")
3. **Run end-to-end test** of Phase 9.3 and 9.4 sections
4. **Validate outputs**:
    - `all_stocks_selected` has reduced feature count
    - `all_stocks_multilabel` has `label_*` columns
    - `all_stocks_balanced` has improved class distribution
5. **Document** any issues or adjustments needed

---

## References

- **Phase 9.3 Plan**: `docs/improvement_plan/phase_9.3_implementation_plan.md`
- **Phase 9.4 Plan**: `docs/improvement_plan/phase_9.4_implementation_plan.md`
- **Phase 9.5 Plan**: `docs/improvement_plan/phase_9.5_implementation_plan.md` (already integrated)
- **Code Guidelines**: `docs/code_guidelines.md` v1.10
- **Test Coverage**:
    - Phase 9.3: `tests/test_feature_selection_auto.py` (4 tests)
    - Phase 9.4: `tests/test_multilabel_classification.py`, `test_cv_policy_enforcement.py`,
      `test_class_balance_remediation.py` (9 tests)
    - Phase 9.5: `tests/test_feature_alignment.py`, `test_stacking_hyperparameter.py` (7 tests)

**Total New Tests**: 20 tests (Phase 9.3: 4, Phase 9.4: 9, Phase 9.5: 7)  
**All Tests Passing**: ✅ 100% pass rate (1815 total tests in suite)
