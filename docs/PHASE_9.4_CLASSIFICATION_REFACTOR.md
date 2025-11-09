# Phase 9.4 - Classification Improvements Implementation Summary

## Status: COMPLETED (Partial)

**Date:** 2025-11-08  
**Phase:** 9.4 - Classification subpackage refactor with Phase 9.3 feature integration

---

## What Was Implemented

### 1. Classification Subpackage Structure Created

```
finance_ml/ml_workflow/classification/
├── __init__.py           (42 lines)  - Public API exports
├── labels.py            (193 lines)  - Event label creation
└── tuning.py            (331 lines)  - Hyperparameter optimization & CV
```

### 2. Modules Extracted and Refactored

#### labels.py

- **Function:** `create_enhanced_event_labels`
- **Methods:** price_momentum, valuation, fundamental, volatility, analyst_rating, market_events
- **Features:**
    - 6 event detection methods
    - Sector-specific threshold adjustments
    - Comprehensive documentation

#### tuning.py

- **Functions:**
    - `optimize_classifier_hyperparameters` - Optuna-based Bayesian optimization
    - `cross_validate_with_sector_stratification` - Sector-aware CV using StratifiedGroupKFold
- **Supported Models:** XGBoost, LightGBM, CatBoost, Random Forest
- **Features:**
    - F1-macro optimization
    - Configurable n_trials, cv_folds
    - Optional dependency handling (Optuna)

### 3. Backward Compatibility Maintained

#### classification.py

- Added Phase 9.4 refactor notice (lines 99-114)
- Imported `create_enhanced_event_labels` from new location
- Added deprecation wrapper with warning message
- Old imports continue to work: `from finance_ml.ml_workflow.classification import create_enhanced_event_labels`

#### classification_enhanced.py

- Added Phase 9.4 refactor notice (lines 61-76)
- Imported functions from `classification.tuning`
- Functions remain in place (full extraction deferred)
- Old imports continue to work

#### finance_ml/__init__.py

- Added Phase 9.4 imports (lines 157-165):
    - `classification_create_enhanced_event_labels`
    - `classification_optimize_hyperparameters`
    - `classification_cross_validate_sector`
- Aliased to avoid conflicts with existing imports
- Both old and new import paths available

### 4. Testing and Verification

✓ All imports tested and working
✓ Direct imports from submodules functional
✓ Package-level imports functional
✓ Backward compatibility verified
✓ No breaking changes to existing code

---

## What Was Deferred

### 1. Complete Extraction (Future Phase 9.4.1)

The following remain in the parent `classification.py` and should be extracted in a future iteration:

#### models.py (to be created)

- `prepare_classification_data`
- `_prepare_categorical_features`
- `train_xgboost_classifier`
- `train_lightgbm_classifier`
- `train_catboost_classifier`
- `train_svm_classifier`
- `train_neural_network_classifier`
- `train_voting_classifier`
- `train_stacking_classifier`
- `compare_classifiers`
- Sampling functions: `apply_smote`, `apply_adasyn`, `apply_undersampling`, `apply_combined_sampling`
- `export_classification_features`
- `clean_extreme_values`
- `validate_data_quality`

#### evaluation.py (to be created)

- `evaluate_classification`
- `evaluate_classification_by_sector`
- `plot_confusion_matrices`
- `plot_learning_curves`
- `compute_shap_values`
- `analyze_per_class_feature_importance`
- `cross_validate_classifier`
- `compare_feature_importance`
- `analyze_calibration` (from classification_enhanced.py)

### 2. New fit_classifier Orchestrator

Per the improvement plan, a high-level `fit_classifier` function should be created:

```python
from finance_ml.ml_workflow.classification.models import fit_classifier

clf_res = fit_classifier(
    X_feat, labels, 
    model="lightgbm", 
    tuning={"n_trials": 50}, 
    cv={"sector_stratified": True}
)
```

This orchestrator should:

- Accept model type and parameters
- Optionally run hyperparameter optimization
- Optionally run sector-stratified CV
- Return standardized dict: `{model, metrics, y_pred, y_proba, artifacts}`

---

## Phase 9.3 Feature Integration Strategy

### Current Phase 9.3 Features Available

From `finance_ml.ml_workflow.features.advanced`:

1. **Analyst Quality Features** (`engineer_analyst_quality_features`)
    - `analyst_coverage` (# of analysts)
    - `analyst_consensus_strength` (std of ratings)
    - `price_target_spread` (high - low spread)
    - `rating_buy_ratio` (buy ratings / total ratings)
    - `rating_sell_ratio` (sell ratings / total ratings)

2. **Accounting Quality Features** (`engineer_accounting_quality_features`)
    - `exceptional_items_intensity` (exceptional items / revenue)
    - `goodwill_intensity` (goodwill / total assets)
    - `intangibles_ratio` (intangibles / total assets)
    - `accounting_quality_score` (composite score)

3. **Employee Productivity Features** (`engineer_employee_productivity_features`)
    - `revenue_per_employee`
    - `profit_per_employee`
    - `assets_per_employee`
    - `employee_growth_rate` (YoY change)

### Integration Approach

#### Step 1: Update prepare_classification_data

Add feature detection and categorization in `prepare_classification_data`:

```python
def prepare_classification_data(
    df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_groups: Optional[List[str]] = None,  # NEW
):
    """Prepare data for classification with Phase 9.3 feature support.
    
    Args:
        df: Input DataFrame with features
        labels: Classification labels
        test_size: Test set size
        random_state: Random seed
        feature_groups: Optional list of feature groups to include:
            - 'analyst_quality': Analyst coverage and rating features
            - 'accounting_quality': Accounting quality indicators
            - 'employee_productivity': Employee productivity metrics
            If None, includes all available features.
    """
    # Detect Phase 9.3 features
    analyst_cols = [c for c in df.columns if any(k in c for k in 
                    ['analyst_coverage', 'analyst_consensus', 'price_target_spread', 
                     'rating_buy_ratio', 'rating_sell_ratio'])]
    
    accounting_cols = [c for c in df.columns if any(k in c for k in 
                       ['exceptional_items_intensity', 'goodwill_intensity', 
                        'intangibles_ratio', 'accounting_quality_score'])]
    
    employee_cols = [c for c in df.columns if any(k in c for k in 
                     ['revenue_per_employee', 'profit_per_employee', 
                      'assets_per_employee', 'employee_growth_rate'])]
    
    # Feature selection based on groups
    if feature_groups is not None:
        selected_features = []
        if 'analyst_quality' in feature_groups:
            selected_features.extend(analyst_cols)
        if 'accounting_quality' in feature_groups:
            selected_features.extend(accounting_cols)
        if 'employee_productivity' in feature_groups:
            selected_features.extend(employee_cols)
        
        # Filter df to only include selected features
        keep_cols = [c for c in df.columns if c not in 
                     analyst_cols + accounting_cols + employee_cols or 
                     c in selected_features]
        df = df[keep_cols]
    
    # Continue with existing logic...
    # (train/test split, categorical encoding, etc.)
```

#### Step 2: Update Feature Importance Analysis

Modify feature importance functions to highlight Phase 9.3 features:

```python
def analyze_feature_importance_with_groups(importance_df: pd.DataFrame):
    """Analyze feature importance with Phase 9.3 feature grouping."""
    # Categorize features
    importance_df['feature_group'] = 'basic'
    
    analyst_keywords = ['analyst_coverage', 'analyst_consensus', 'price_target_spread', 
                        'rating_buy_ratio', 'rating_sell_ratio']
    accounting_keywords = ['exceptional_items', 'goodwill_intensity', 
                          'intangibles_ratio', 'accounting_quality']
    employee_keywords = ['revenue_per_employee', 'profit_per_employee', 
                        'assets_per_employee', 'employee_growth']
    
    for idx, row in importance_df.iterrows():
        feat = row['feature']
        if any(k in feat for k in analyst_keywords):
            importance_df.loc[idx, 'feature_group'] = 'analyst_quality'
        elif any(k in feat for k in accounting_keywords):
            importance_df.loc[idx, 'feature_group'] = 'accounting_quality'
        elif any(k in feat for k in employee_keywords):
            importance_df.loc[idx, 'feature_group'] = 'employee_productivity'
    
    # Group importance by feature group
    group_importance = importance_df.groupby('feature_group')['importance'].sum()
    
    return importance_df, group_importance
```

#### Step 3: Update Categorical Feature Handling

Ensure categorical columns from Phase 9.3 are properly handled:

```python
# In _prepare_categorical_features or prepare_classification_data

# Schema-based categorical columns
schema_categorical = [
    'sector', 'industry', 'region', 'country', 'trading_country',
    'style_class', 'size_class', 'flag',  # From equities schema
]

# Detect categorical columns
categorical_cols = [c for c in df.columns if 
                   c in schema_categorical or 
                   df[c].dtype == 'object']
```

#### Step 4: Notebook Integration Example

```python
# In ml_finance_model_main.ipynb

# Phase 9.3: Build comprehensive features with Phase 9.3 enhancements
from finance_ml.ml_workflow.features.advanced import (
    build_comprehensive_features,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    )

# Build features
all_stocks_features = build_comprehensive_features(
        all_stocks_scaled,
        include_interactions=True,
        include_relative_values=True,
        sector_col='sector'
        )

# Add Phase 9.3 features
all_stocks_features = engineer_analyst_quality_features(all_stocks_features)
all_stocks_features = engineer_accounting_quality_features(all_stocks_features)
all_stocks_features = engineer_employee_productivity_features(all_stocks_features)

# Phase 9.4: Create labels with new classification subpackage
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels

labels = create_enhanced_event_labels(
        all_stocks_features,
        method='price_momentum',
        threshold_positive=10.0,
        threshold_negative=-10.0,
        use_sector_adjustment=True
        )

# Prepare classification data with Phase 9.3 feature groups
from finance_ml.ml_workflow.classification import prepare_classification_data

X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_classification_data(
        all_stocks_features, labels,
        feature_groups=['analyst_quality', 'accounting_quality', 'employee_productivity'],
        test_size=0.2, random_state=42
        )

# Train with hyperparameter optimization
from finance_ml.ml_workflow.classification.tuning import optimize_classifier_hyperparameters

result = optimize_classifier_hyperparameters(
        X_train, y_train,
        classifier_type='lightgbm',
        n_trials=50,
        cv_folds=5,
        verbose=True
        )

print(f"Best F1 score: {result['best_score']:.4f}")
print(f"Best parameters: {result['best_params']}")
```

---

## Files Modified

1. `finance_ml/ml_workflow/classification/__init__.py` - Created (42 lines)
2. `finance_ml/ml_workflow/classification/labels.py` - Created (193 lines)
3. `finance_ml/ml_workflow/classification/tuning.py` - Created (331 lines)
4. `finance_ml/ml_workflow/classification.py` - Updated (added deprecation wrapper)
5. `finance_ml/ml_workflow/classification_enhanced.py` - Updated (added import block)
6. `finance_ml/__init__.py` - Updated (added Phase 9.4 imports)

## Total Lines Added/Modified

- **New code:** 566 lines (labels.py + tuning.py + __init__.py)
- **Modified code:** ~30 lines (deprecation wrappers and imports)
- **Total impact:** ~596 lines

---

## Next Steps (Phase 9.4.1 - Future Work)

1. **Extract models.py** (~800-1000 lines)
    - All train_* functions
    - compare_classifiers
    - Sampling methods
    - Data preparation helpers

2. **Extract evaluation.py** (~400-500 lines)
    - All evaluate_* functions
    - Plotting functions
    - SHAP analysis
    - Calibration analysis

3. **Create fit_classifier orchestrator**
    - High-level API wrapper
    - Integrated tuning and CV
    - Standardized return format

4. **Full Phase 9.3 integration**
    - Update prepare_classification_data with feature groups
    - Add feature importance grouping
    - Update notebook examples

5. **Update tests**
    - Test classification subpackage modules
    - Test backward compatibility
    - Test Phase 9.3 feature integration

---

## Benefits Achieved

✅ **Modularity:** Separated concerns (labels, tuning) into focused modules  
✅ **Maintainability:** Easier to locate and update specific functionality  
✅ **Backward Compatibility:** No breaking changes to existing code  
✅ **Documentation:** Comprehensive docstrings with examples  
✅ **Testability:** Isolated modules are easier to unit test  
✅ **Extensibility:** Clear structure for adding new functionality

---

## References

- Improvement Plan: `docs/improvement_plan/finance_ml_improvement_plan.md`
- Phase 9.3 Features: `finance_ml/ml_workflow/features/advanced.py`
- Schema: `create_equities_schema.sql`
- Notebook: `ml_finance_model_main.ipynb`
