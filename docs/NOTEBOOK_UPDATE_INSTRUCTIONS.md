# Notebook Update Instructions for Zero Predictions Fix

## Overview

This document provides exact code changes to apply the zero predictions fix to `ml_finance_model_main.ipynb`.

The fix replaces hard zero lower-bound clipping with percentile-based adaptive clipping using the new
`adaptive_clip_predictions()` function.

## Changes Required

### 1. Add Import Statement (Cell at beginning of notebook)

Add to the imports section:

```python
from finance_ml.ml_workflow.regression.robust import adaptive_clip_predictions
```

### 2. Update Stacking Ensemble (Lines ~1897-1931)

**OLD CODE** (to be replaced):

```python
# Calculate adaptive upper bound
train_p995 = np.percentile(y_train, 99.5)
upper_bound = train_p995 * 1.5

# Apply clipping with diagnostic logging
y_pred_stacking = np.clip(y_pred_stacking, 0, upper_bound)

print(f"  Clipping Strategy: upper_bound = {upper_bound:.2f}")
```

**NEW CODE** (replacement):

```python
# Apply adaptive clipping with percentile-based bounds
# This eliminates zero predictions while preserving low-value predictions
clip_result = adaptive_clip_predictions(y_pred_stacking, y_train)
y_pred_stacking = clip_result['clipped_predictions']

# Log clipping diagnostics
print(f"  Clipping Strategy:")
print(f"    Lower bound: ${clip_result['lower_bound']:.2f} (0.5 × p0.5, min $0.10)")
print(f"    Upper bound: ${clip_result['upper_bound']:.2f} (1.5 × p99.5)")
print(f"    Clipped to lower bound: {clip_result['n_clipped_lower']} ({clip_result['pct_clipped_lower']:.1f}%)")
print(f"    Clipped to upper bound: {clip_result['n_clipped_upper']} ({clip_result['pct_clipped_upper']:.1f}%)")

# Verify zero elimination
n_zeros = np.sum(y_pred_stacking == 0.0)
print(f"    Zero predictions: {n_zeros} (should be 0)")
```

### 3. Update Time-Series Cross-Validation Loop (Lines ~2173-2180)

**OLD CODE** (to be replaced):

```python
# Calculate upper bound from training fold
train_p995 = np.percentile(y_tr, 99.5)
upper_bound = train_p995 * 1.5

# Clip predictions
y_hat = np.clip(fold_pred, 0, upper_bound)
```

**NEW CODE** (replacement):

```python
# Apply adaptive clipping with percentile-based bounds
clip_result_fold = adaptive_clip_predictions(fold_pred, y_tr)
y_hat = clip_result_fold['clipped_predictions']

# Optional: log clipping stats for first fold
if fold_idx == 0:
    print(f"  Fold clipping: lower=${clip_result_fold['lower_bound']:.2f}, "
          f"upper=${clip_result_fold['upper_bound']:.2f}")
```

## Expected Results

After applying these changes:

1. **Zero Predictions Eliminated**: Count should drop from ~350 to 0
2. **Lower Bound Range**: Typically $0.10 - $2.00 (depends on training data)
3. **Clipping Statistics**: 1-5% clipped to lower bound, <1% to upper bound
4. **MAE Impact**: Minimal change (<1%), slight improvement for low-value stocks
5. **Distribution**: Natural prediction distribution restored (no spike at zero)

## Validation

After updating the notebook:

1. Run cells 6.4 (Stacking Ensemble) and 6.5 (Time-Series CV)
2. Check console output for clipping diagnostics
3. Verify "Zero predictions: 0"
4. Compare predictions.csv before/after:
   ```python
   import pandas as pd
   import numpy as np
   
   df = pd.read_csv('../outputs/regression/regression_predictions_detailed.csv')
   n_zeros = np.sum(df['y_pred'] == 0)
   print(f"Zero predictions: {n_zeros} (expected: 0)")
   ```

## Testing

Run the validation script to confirm the fix:

```bash
python validate_zero_predictions_fix.py
```

Expected output: `[SUCCESS] ALL VALIDATION CHECKS PASSED`

## References

- **Issue**: ZERO_PREDICTIONS_FIX.md
- **Function**: `finance_ml.ml_workflow.regression.robust.adaptive_clip_predictions()`
- **Tests**: `tests/test_robust_outlier_safety.py::TestAdaptiveClipPredictions`
- **Validation**: `validate_zero_predictions_fix.py`
