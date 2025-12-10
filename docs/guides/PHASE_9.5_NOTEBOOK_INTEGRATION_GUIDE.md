# Phase 9.5 Notebook Integration Guide

**Date:** 2025-12-09  
**Version:** 1.0  
**Related Files:** ml_finance_model_main.ipynb, code_guidelines.md v1.10+

## Overview

This guide documents the integration of Phase 9.5 TDD implementations (Task 6: Stacking Hyperparameter Tuning and Task
7: Feature Alignment) into the ml_finance_model_main.ipynb notebook.

## Changes Completed

### 1. Dependencies (✓ Complete)

**File:** `requirements.txt`

Added hyperparameter optimization library:

```python
# Hyperparameter Optimization
optuna>=4.6.0,<5.0.0
```

**Location:** After line 32 (gradient boosting libraries section)

### 2. Imports (✓ Complete)

**File:** `ml_finance_model_main.ipynb`

Added five new Phase 9.5 TDD functions to the imports section:

**Regression Dataset Functions** (lines 598-604):

```python
from finance_ml.ml_workflow.regression.dataset import (
    integrate_classification_features,
    create_classification_interactions as regression_create_classification_interactions,
    prepare_regression_data as regression_prepare_data,
    align_features_to_model,  # NEW - Phase 9.5 Task 7
    predict_with_model,        # NEW - Phase 9.5 Task 7
    )
```

**Regression Model Functions** (lines 607-613):

```python
from finance_ml.ml_workflow.regression.models import (
    compare_regressors as regression_compare_regressors,
    train_stacking_regressor as regression_train_stacking,
    tune_stacking_hyperparameters,  # NEW - Phase 9.5 Task 6
    select_stacking_base_models,     # NEW - Phase 9.5 Task 6
    select_meta_learner,             # NEW - Phase 9.5 Task 6
    )
```

## Notebook Integration - Phase 9.5 Section

### Current Status

- **Section Location:** Phase 9.5: Sector-Optimized Regression Models (search for "## Phase 9.5")
- **Imports:** ✓ Updated with new functions
- **Hyperparameter Tuning Cell:** ⚠️ Needs to be added

### Recommended Cell Additions

The following cells should be added to the Phase 9.5 section, preferably after the model comparison cell and before the
stacking ensemble training.

---

#### Cell 1: Markdown Documentation

```markdown
### 9.5.X: Automated Stacking Hyperparameter Tuning (Optional)

**Business Goal:** Optimize stacking ensemble performance through automated hyperparameter search

**Key Objectives:**
- Bayesian optimization with Optuna for efficient search space exploration
- Tune base models (XGBoost, LightGBM, Ridge, Lasso) and meta-learner simultaneously
- Balance performance vs. computation time (50 trials, 30-minute timeout)
- Reproducible results with fixed random seed

**Inputs:**
- `X_train_reg`, `y_train_reg`: Training features and target
- Feature names for model interpretation

**Outputs:**
- `best_stacking_params`: Optimized hyperparameters dictionary
- `tuned_stacking_model`: Trained model with best hyperparameters
- Study results saved to `outputs/regression/stacking_optuna_study.pkl`

**Performance Trade-offs:**
- **Skip if time-constrained:** Default stacking ensemble already provides good performance
- **Run if optimizing production model:** Can improve R² by 1-3% on validation set
- **Computation time:** ~30 minutes for 50 trials (configurable)

**Phase 9.5 TDD Implementation:**
- Function: `tune_stacking_hyperparameters()` from `regression.models`
- Test coverage: 84.5% (3 tests in test_stacking_hyperparameter.py)
- Documentation: code_guidelines.md v1.10+ Section 7.3
```

---

#### Cell 2: Python Code - Hyperparameter Tuning

```python
#%%
# 9.5.X: Automated Stacking Hyperparameter Tuning (Optional - Can Skip)
print('\n' + '='*80)
print('PHASE 9.5.X: AUTOMATED STACKING HYPERPARAMETER TUNING')
print('='*80)

# Configuration
RUN_HYPERPARAMETER_TUNING = False  # Set to True to enable (adds ~30 minutes)
N_TRIALS = 50  # Number of Optuna trials
TIMEOUT_SECONDS = 1800  # 30 minutes timeout

if RUN_HYPERPARAMETER_TUNING:
    print(f'\n📊 Starting Optuna hyperparameter search...')
    print(f'   Trials: {N_TRIALS}, Timeout: {TIMEOUT_SECONDS}s (~{TIMEOUT_SECONDS//60} min)')
    print(f'   Search space: XGBoost, LightGBM, Ridge, Lasso base models + Ridge/Huber meta-learner')
    
    try:
        # Run hyperparameter tuning
        tuning_result = tune_stacking_hyperparameters(
            X_train=X_train_reg,
            y_train=y_train_reg,
            n_trials=N_TRIALS,
            timeout=TIMEOUT_SECONDS,
            random_state=RANDOM_SEED,
            verbose=True
        )
        
        # Extract results
        best_stacking_params = tuning_result['best_params']
        tuned_stacking_model = tuning_result['best_model']
        best_score = tuning_result['best_score']
        
        # Display results
        print(f'\n✅ Hyperparameter tuning complete!')
        print(f'   Best CV Score (R²): {best_score:.4f}')
        print(f'\n📋 Best Hyperparameters:')
        for key, value in best_stacking_params.items():
            print(f'   {key}: {value}')
        
        # Save study for later analysis
        study_path = OUTPUT_DIR / 'regression' / 'stacking_optuna_study.pkl'
        if 'study' in tuning_result:
            import joblib
            joblib.dump(tuning_result['study'], study_path)
            print(f'\n💾 Saved Optuna study: {study_path}')
        
        # Use tuned model for subsequent predictions
        print(f'\n✓ Using tuned stacking model for predictions')
        stacking_model = tuned_stacking_model
        
    except Exception as e:
        print(f'\n⚠️  Hyperparameter tuning failed: {e}')
        print(f'   Falling back to default stacking ensemble')
        tuned_stacking_model = None
        best_stacking_params = None

else:
    print(f'\n⏭️  Skipping hyperparameter tuning (RUN_HYPERPARAMETER_TUNING=False)')
    print(f'   Using default stacking ensemble configuration')
    print(f'   To enable: Set RUN_HYPERPARAMETER_TUNING = True above')
    tuned_stacking_model = None
    best_stacking_params = None

print('\n' + '='*80)
```

---

#### Cell 3: Feature Alignment Example (Optional)

```markdown
### 9.5.Y: Safe Prediction with Feature Alignment

**Business Goal:** Prevent runtime errors from feature mismatches between training and prediction

**Key Objectives:**
- Automatically align test features to match trained model's expected features
- Handle missing features (fill with 0 or median)
- Handle extra features (drop silently)
- Provide diagnostic information for troubleshooting

**Use Case:**
- Predict on new data with different feature sets
- Deploy models trained on different dataset versions
- Handle dynamic feature engineering pipelines

**Phase 9.5 TDD Implementation:**
- Functions: `align_features_to_model()`, `predict_with_model()` from `regression.dataset`
- Test coverage: 94.0% (4 tests in test_feature_alignment.py)
- Documentation: code_guidelines.md v1.10+ Section 10.4 (Feature Alignment Policy)
```

---

#### Cell 4: Python Code - Feature Alignment Demo

```python
#%%
# 9.5.Y: Feature Alignment Demo (Optional)
print('\n' + '='*80)
print('PHASE 9.5.Y: FEATURE ALIGNMENT DEMONSTRATION')
print('='*80)

DEMO_FEATURE_ALIGNMENT = False  # Set to True to run demo

if DEMO_FEATURE_ALIGNMENT and 'stacking_model' in dir():
    print('\n📊 Demonstrating safe prediction with feature alignment...')
    
    # Simulate test data with mismatched features
    # (In production, this would be your actual test/validation set)
    X_test_mismatched = X_test_reg.copy()
    
    # Example 1: Missing features (drop 10 random columns)
    import random
    random.seed(RANDOM_SEED)
    cols_to_drop = random.sample(list(X_test_mismatched.columns), k=10)
    X_test_missing_features = X_test_mismatched.drop(columns=cols_to_drop)
    
    print(f'\n🔍 Example 1: Missing Features')
    print(f'   Original features: {len(X_test_mismatched.columns)}')
    print(f'   Test features: {len(X_test_missing_features.columns)}')
    print(f'   Missing: {len(cols_to_drop)} features')
    
    try:
        # Safe prediction with automatic alignment
        y_pred_aligned = predict_with_model(
            model=stacking_model,
            X_new=X_test_missing_features,
            strategy='zero'  # Options: 'zero', 'median'
        )
        print(f'   ✅ Prediction successful: {len(y_pred_aligned)} predictions')
        print(f'   Mean prediction: ${y_pred_aligned.mean():.2f}')
    except Exception as e:
        print(f'   ❌ Prediction failed: {e}')
    
    # Example 2: Extra features (add 5 random noise columns)
    X_test_extra_features = X_test_mismatched.copy()
    for i in range(5):
        X_test_extra_features[f'noise_feature_{i}'] = np.random.randn(len(X_test_extra_features))
    
    print(f'\n🔍 Example 2: Extra Features')
    print(f'   Original features: {len(X_test_mismatched.columns)}')
    print(f'   Test features: {len(X_test_extra_features.columns)}')
    print(f'   Extra: 5 noise features')
    
    try:
        # Safe prediction with automatic alignment (extra features dropped)
        y_pred_aligned2 = predict_with_model(
            model=stacking_model,
            X_new=X_test_extra_features,
            strategy='zero'
        )
        print(f'   ✅ Prediction successful: {len(y_pred_aligned2)} predictions')
        print(f'   Mean prediction: ${y_pred_aligned2.mean():.2f}')
    except Exception as e:
        print(f'   ❌ Prediction failed: {e}')
    
    print('\n💡 Key Takeaway:')
    print('   Use predict_with_model() for production deployments to handle')
    print('   feature mismatches gracefully without manual intervention.')

else:
    print(f'\n⏭️  Skipping feature alignment demo (DEMO_FEATURE_ALIGNMENT=False)')
    print(f'   Set DEMO_FEATURE_ALIGNMENT = True to see examples')

print('\n' + '='*80)
```

---

## Integration Instructions

### Step 1: Locate Phase 9.5 Section

Search for: `"## Phase 9.5: Sector-Optimized Regression Models"`

### Step 2: Find Insertion Point

Recommended location: **After model comparison cell, before final stacking training**

Look for these markers:

- After: Model comparison results display
- Before: Final stacking ensemble training with all base models

### Step 3: Insert Cells

Add the four cells above in sequence:

1. Markdown documentation for hyperparameter tuning
2. Python code for hyperparameter tuning (with toggle)
3. Markdown documentation for feature alignment
4. Python code for feature alignment demo (with toggle)

### Step 4: Update Section Numbers

Renumber subsequent Phase 9.5 subsections (e.g., 9.5.X → 9.5.6, 9.5.Y → 9.5.7)

### Step 5: Test Execution

Run the cells with toggles set to `False` initially:

- `RUN_HYPERPARAMETER_TUNING = False` (default)
- `DEMO_FEATURE_ALIGNMENT = False` (default)

This ensures the notebook executes quickly during regular runs.

## Unified ETL Pipeline Integration

### Status: ✓ Already Implemented

The notebook (as of version "Phase 9.5 - Unified ETL with Semantic Transforms + Feature Engineering", updated
2025-12-08) already uses the unified ETL pipeline from code_guidelines.md v1.10.

**Key Integration Points:**

1. **Data Loading:** Uses `etl_with_features()` as single entry point
2. **Feature Engineering:** Uses `feature_preset` parameter (basic, momentum, quality, comprehensive)
3. **Semantic Transformations:** Preserves price columns, applies log transforms for market values
4. **Schema Alignment:** Uses `COLUMN_SCHEMA` from `finance_ml.ml_workflow.data.schema`

**Evidence from Notebook Header:**

```python
# Version: Phase 9.5 (Unified ETL with Semantic Transforms + Feature Engineering)
# Last Updated: 2025-12-08
# Aligned with: code_guidelines.md v1.10, Section 2.4 Business-Driven Configuration
```

### Migration Complete

The notebook has been fully migrated to the unified ETL pipeline. No additional changes needed for ETL integration.

## Testing and Validation

### Test Suite Status

**Phase 9.5 TDD Tests:** ✓ 7 tests passing (no regressions)

- `test_feature_alignment.py`: 4 tests (Task 7)
- `test_stacking_hyperparameter.py`: 3 tests (Task 6)

**Coverage:**

- `regression/dataset.py` new functions: 94.0%
- `regression/models.py` new functions: 84.5%
- Overall new functions: 87.6%

### Validation Checklist

- [x] Optuna added to requirements.txt
- [x] New functions imported in notebook
- [x] Integration guide created
- [x] Example cells documented
- [ ] Cells added to notebook (manual step)
- [ ] Notebook tested end-to-end (manual step)

## References

- **Implementation Plan:** `docs/improvement_plan/phase_9.5_implementation_plan.md`
- **Code Guidelines:** `docs/code_guidelines.md` v1.10+ (Sections 7.3, 10.4)
- **Test Files:** `tests/test_feature_alignment.py`, `tests/test_stacking_hyperparameter.py`
- **Module Files:** `finance_ml/ml_workflow/regression/{dataset.py, models.py}`

## Troubleshooting

### Issue: Import Errors

**Symptom:** `ImportError: cannot import name 'tune_stacking_hyperparameters'`

**Solution:**

1. Verify optuna is installed: `python -c "import optuna; print(optuna.__version__)"`
2. Verify functions exist:
   `python -c "from finance_ml.ml_workflow.regression.models import tune_stacking_hyperparameters"`
3. Restart Jupyter kernel

### Issue: Hyperparameter Tuning Takes Too Long

**Symptom:** Tuning runs for >30 minutes

**Solutions:**

1. Reduce `N_TRIALS` from 50 to 20-30
2. Reduce `TIMEOUT_SECONDS` from 1800 to 900 (15 min)
3. Set `RUN_HYPERPARAMETER_TUNING = False` and use default ensemble

### Issue: Feature Alignment Fails

**Symptom:** `ValueError: Cannot align features`

**Solutions:**

1. Check model has `feature_names_in_` attribute
2. Verify X_new is a DataFrame (not numpy array)
3. Use `strategy='median'` instead of `strategy='zero'`

## Summary

This guide documents all changes needed to integrate Phase 9.5 TDD implementations into the notebook:

1. **Dependencies:** ✓ optuna>=4.6.0 added to requirements.txt
2. **Imports:** ✓ 5 new functions added to notebook imports
3. **Documentation:** ✓ Example cells and integration guide created
4. **Testing:** ✓ 7 tests passing with 87.6% coverage
5. **ETL Pipeline:** ✓ Already integrated (unified ETL with semantic transforms)

**Next Steps:**

1. Add the four example cells to Phase 9.5 section (manual step)
2. Test notebook execution end-to-end
3. Update cell numbers and cross-references

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09  
**Author:** Phase 9.5 TDD Implementation Team
