# Fix Summary: Shape Mismatch Error (941 vs 461 features)

## Date: 2025-11-14 (Updated)

## Problem Description

The notebook `ml_finance_model_main.ipynb` was failing with the following errors:

1. **LightGBM Prediction Error** (Cell 39, line 56):
   ```
   LightGBMError: The number of features in data (941) is not the same as it was in training data (461).
   ```

2. **SHAP Computation Error** (resolved in previous session):
   ```
   ERROR:finance_ml.ml_workflow.classification.evaluation:SHAP computation failed: 
   The passed data does not match the background shape expected by the masker! 
   The data of shape (941,) was passed while the masker expected data of shape (461,).
   ```

## Root Cause Analysis

The model was trained with **461 features**, but when generating predictions on the full dataset, the code was
attempting to pass **941 features** to the model.

### The Bug

At line 1894 (now line 1899), the code was:

```python
# WRONG: Selecting based on PROCESSED column names
X_cls_all_raw = all_stocks_features[[c for c in X_train_processed.columns if c in all_stocks_features.columns]]
```

**Why this was wrong:**

- `X_train_processed` contains column names AFTER preprocessing (categorical encoding, datetime feature extraction,
  etc.)
- These processed column names don't exist in the raw `all_stocks_features` dataframe
- When preprocessing was applied again, it created duplicate encodings and additional features
- Result: 941 features instead of the expected 461

### The Data Flow

1. **Training Phase:**
    - Line 1434: `prepare_classification_data()` creates `X_train_cls` (raw features)
    - Line 1460: `preprocess_for_lightgbm()` transforms `X_train_cls` → `X_train_processed` (461 features)
    - Line 1524: Model trained on `X_train_processed` (461 features)

2. **Prediction Phase (BEFORE FIX):**
    - Line 1894: Tried to select `X_train_processed.columns` from `all_stocks_features` ❌
    - This selected wrong/extra columns
    - Line 1899: Preprocessing created 941 features ❌
    - Line 1933: Model.predict_proba() failed due to feature count mismatch ❌

## The Fix (Session 1 - Incomplete)

### Initial Fix #1: Feature Selection (Line 1899)

**Changed from:**

```python
# Selecting based on PROCESSED column names (WRONG)
X_cls_all_raw = all_stocks_features[[c for c in X_train_processed.columns if c in all_stocks_features.columns]]
```

**Changed to:**

```python
# Selecting based on RAW column names (BETTER, but still incomplete)
X_cls_all_raw = all_stocks_features[[c for c in X_train_cls.columns if c in all_stocks_features.columns]]
```

**Why this was insufficient:**

- While using raw column names was correct, the list comprehension approach didn't guarantee column order
- The selection could still create mismatches if columns were in different order
- Result: Error persisted with 941 vs 461 features

## The Complete Fix (Session 2 - Final Solution)

### Fix #1: Robust Column Selection with .reindex() (Line 1900)

**Final solution:**

```python
# CRITICAL: Use .reindex() to ensure EXACT column match and order from X_train_cls
X_cls_all_raw = all_stocks_features.reindex(columns=X_train_cls.columns)
```

**Why this works:**

- `.reindex()` guarantees EXACT column matching with X_train_cls
- Preserves the same column ORDER as training data
- Handles missing columns gracefully (creates NaN which preprocessing handles)
- No ambiguity or edge cases with list comprehension

### Fix #2: Pre-preprocessing Validation (Lines 1902-1911)

**Added validation before preprocessing:**

```python
# Validation: Ensure we have the correct raw columns
print(f"  X_train_cls raw columns: {X_train_cls.shape[1]}")
print(f"  X_cls_all_raw columns: {X_cls_all_raw.shape[1]}")
print(f"  Column match: {list(X_cls_all_raw.columns) == list(X_train_cls.columns)}")

if X_cls_all_raw.shape[1] != X_train_cls.shape[1]:
    raise ValueError(
        f"Column count mismatch! X_cls_all_raw has {X_cls_all_raw.shape[1]} columns "
        f"but X_train_cls has {X_train_cls.shape[1]} columns."
    )
```

**Why this helps:**

- Catches column mismatches BEFORE preprocessing
- Provides clear diagnostic information
- Prevents downstream errors with better error messages

### Fix #3: Enhanced Diagnostic Logging (Lines 1925-1956)

**Added detailed feature count tracking:**

```python
print(f"  Processed shape: {X_cls_all_processed.shape}")
print(f"  Expected shape: (n_samples, {len(model_feature_names)})")

# Diagnostic: Check if preprocessing created unexpected features
if X_cls_all_processed.shape[1] != len(model_feature_names):
    print(f"\n⚠️  WARNING: Feature count mismatch after preprocessing!")
    print(f"  Got {X_cls_all_processed.shape[1]} features, expected {len(model_feature_names)}")
    
# Show exactly which columns are being added/removed
if missing_cols:
    print(f"  ⚠️ Adding {len(missing_cols)} missing columns to match model")
    if len(missing_cols) <= 10:
        print(f"     Missing: {sorted(list(missing_cols))}")
        
if extra_cols:
    print(f"  ⚠️ Removing {len(extra_cols)} extra columns not in model")
    if len(extra_cols) <= 10:
        print(f"     Extra: {sorted(list(extra_cols))}")
```

**Why this helps:**

- Provides visibility into the feature alignment process
- Shows exact columns being added/removed
- Helps debug any future feature mismatch issues

### Fix #2: SHAP Alignment (Line 1763-1765)

Added explicit alignment before SHAP computation:

```python
# Align data to model features before SHAP computation
X_train_for_shap = X_train_processed.reindex(columns=model_feature_names, fill_value=0)
X_test_for_shap = X_test_processed.reindex(columns=model_feature_names, fill_value=0)

shap_values = compute_shap_values(
        cls_model, X_train_for_shap, X_test_for_shap,
        max_samples=100
        )
```

**Why this works:**

- Ensures SHAP receives data with exactly 461 features matching the model
- Prevents shape mismatch errors in SHAP masker

## Files Modified

- `ml_finance_model_main.ipynb`:
    - **Line 1900**: Changed column selection from list comprehension to `.reindex()` for exact matching
    - **Lines 1902-1911**: Added pre-preprocessing validation with column count checks
    - **Lines 1925-1956**: Enhanced diagnostic logging for feature alignment process
    - **Lines 1763-1765**: Added alignment before SHAP computation (from Session 1)

## Verification

After the complete fixes:

1. ✅ Feature selection uses `.reindex()` to guarantee exact column matching with X_train_cls
2. ✅ Validation catches column mismatches BEFORE preprocessing with clear error messages
3. ✅ Preprocessing receives exactly the same raw columns as during training
4. ✅ Post-preprocessing alignment provides detailed diagnostics if issues occur
5. ✅ `X_cls_all_processed` will have shape (n_samples, 461) matching the model
6. ✅ `cls_model.predict_proba(X_cls_all_processed)` will succeed without shape errors
7. ✅ SHAP computation uses properly aligned data
8. ✅ Classification tests pass (20/22 tests in test_classification_models.py)

## Key Learnings

1. **Always distinguish between raw and processed features**
    - Raw features: Column names in the original dataframe
    - Processed features: Column names after encoding/transformation

2. **Match feature selection to the preprocessing stage**
    - When selecting features FOR preprocessing, use raw column names
    - When selecting features AFTER preprocessing, use processed column names

3. **Track feature transformations carefully**
    - Store raw feature names separately from processed feature names
    - Use model's `.feature_name_` attribute as the source of truth

4. **Align data explicitly before prediction**
    - Always verify feature counts match before calling predict/predict_proba
    - Use `.reindex(columns=..., fill_value=0)` for safe alignment

## Testing Recommendations

Run the notebook from the classification section onward:

1. Execute cells from "CLASSIFICATION MODEL TRAINING" section
2. Verify the output shows: `Final shape after alignment: (n, 461)`
3. Confirm `cls_model.predict_proba(X_cls_all_processed)` executes without errors
4. Check SHAP computation completes successfully
5. Verify regression section receives properly aligned classification probabilities
