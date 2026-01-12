# Phase 9.3 API Migration Guide

**Date:** 2025-11-19  
**Purpose:** Migrate notebook from legacy `features_build_comprehensive()` to modern `build_features()` API  
**Reference:** CODEBASE_GAP_ANALYSIS.md (Priority 1 Fix)  
**Alignment:** code_guidelines.md v1.3+, Section 1.3 Phase 9.3 API

## Background

The notebook currently uses the legacy `features_build_comprehensive()` function, which is a backward-compatible alias.
The modern API is `build_features()` from `finance_ml.ml_workflow.features.api`, which provides preset-based
configuration and is the recommended approach per code_guidelines.md v1.3+.

## Notebook Structure Issue

**IMPORTANT:** The notebook has duplicate import sections:

- First section: lines 222-238
- Second section: lines 6338-6354

These sections are **completely identical**, suggesting the notebook may have:

- Duplicate workflow sections
- Multiple execution paths
- Export/conversion artifacts

**Recommendation:** Review notebook structure after applying fixes. Consider splitting into separate notebooks if
workflows are independent.

## Migration Steps

### Step 1: Update Import Sections

**Location 1:** Lines 228-238  
**Location 2:** Lines 6344-6354

**Current code:**

```python
# Phase 9.3: Feature engineering
# Direct import from features subpackage (Phase 9.3 refactor)
from finance_ml import (
    # Phase 9.3 API with presets
    features_build_comprehensive,
    features_importance_rf,
    engineer_valuation_ratios,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    )
```

**Updated code:**

```python
# Phase 9.3: Feature engineering
# Direct import from features subpackage (Phase 9.3 refactor)
from finance_ml import (
    # Phase 9.3 legacy functions (for individual feature engineering)
    features_build_comprehensive,  # Legacy alias - use build_features() for new code
    features_importance_rf,
    engineer_valuation_ratios,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    )
# Phase 9.3 API with presets (code_guidelines.md v1.3+)
from finance_ml.ml_workflow.features.api import build_features
```

**Changes:**

1. Update comment to clarify features_build_comprehensive is legacy
2. Add new import line for build_features from features.api

**Apply to BOTH locations.**

### Step 2: Update Feature Engineering Call (First Occurrence)

**Location:** Line 1570

**Current code:**

```python
# Build comprehensive features
all_stocks_features = features_build_comprehensive(
        all_stocks_scaled,
        include_interactions=True,
        include_relative_values=True,
        sector_col='sector'
        )
```

**Updated code:**

```python
# Build comprehensive features using Phase 9.3 API (code_guidelines.md v1.3+)
all_stocks_features = build_features(
        all_stocks_scaled,
        preset="comprehensive",
        include_interactions=True,
        include_relative=True,  # Parameter name changed from include_relative_values
        sector_col='sector'
        )
```

**Changes:**

1. Replace `features_build_comprehensive` with `build_features`
2. Add `preset="comprehensive"` parameter
3. Change `include_relative_values=True` to `include_relative=True`
4. Update comment to reference code_guidelines.md v1.3+

### Step 3: Update Feature Engineering Call (Second Occurrence)

**Location:** Line 7692 (approximate - in second duplicate section)

Apply the same changes as Step 2 to the second occurrence of the feature engineering call.

### Step 4: Update Phase 9.3 Markdown Documentation

**Location:** Lines 1507-1566

**Current status:** build_features() API examples are commented out

**Action:** Uncomment the examples to show active usage:

**Current:**

```python
### Phase 9.3 API
# Example API usage (not executable in this context):
# from finance_ml.ml_workflow.features.api import build_features
# all_stocks_features = build_features(
#     all_stocks_preprocessed, 
#     preset="comprehensive",
#     include_interactions=True,
#     include_relative=True
# )
```

**Updated:**

```python
### Phase 9.3 API

The
modern
API is now
actively
used in this
notebook(code_guidelines.md
v1
.3 +):

```python
from finance_ml.ml_workflow.features.api import build_features

all_stocks_features = build_features(
        all_stocks_preprocessed,
        preset="comprehensive",
        include_interactions=True,
        include_relative=True,
        sector_col='sector'
        )
```

**Available Presets:**

- `"basic"`: Core ratios, margins, volatility, revenue CAGR
- `"momentum"`: Price momentum and technical indicators
- `"quality"`: Accounting quality and financial distress signals
- `"standard"`: Balanced feature set (valuation, profitability, growth, sentiment)
- `"earnings_analytics"`: Earnings surprises, GAAP vs adjusted, quality flags
- `"technical_plus"`: Technical analysis, valuation timeseries, market sentiment
- `"dividend_focus"`: Dividend reliability, capital allocation, FCF coverage
- `"employment_analytics"`: Employment dynamics, productivity trends
- `"revenue_forecasting"`: Revenue forecasts, growth metrics, analyst quality
- `"balance_sheet"`: Balance sheet trends, leverage, liquidity
- `"comprehensive"`: Full advanced feature set (350 features)

```

### Step 5: Validation

After applying changes:

1. **Import validation:**
   ```python
   # Check import works
   from finance_ml.ml_workflow.features.api import build_features
   print(f"build_features imported: {build_features}")
   ```

2. **API compatibility check:**
   ```python
   # Verify function signature
   import inspect
   sig = inspect.signature(build_features)
   print(f"build_features signature: {sig}")
   ```

3. **DataFrame shape verification:**
   ```python
   # Before and after should have similar feature counts
   print(f"Features shape: {all_stocks_features.shape}")
   # Should be similar to legacy function output
   ```

4. **Downstream phase check:**
    - Ensure Phase 9.4 (Classification) runs without errors
    - Ensure Phase 9.5 (Regression) runs without errors
    - Verify feature columns are available

5. **Run tests:**
   ```bash
   # Run Phase 9.3 tests
   python -m unittest tests.test_features -v
   python -m unittest tests.test_advanced_features -v
   ```

## Expected Behavior

### Before Migration

- Uses legacy alias `features_build_comprehensive()`
- No preset parameter
- Parameter: `include_relative_values=True`

### After Migration

- Uses modern API `build_features()`
- Explicit preset: `preset="comprehensive"`
- Parameter: `include_relative=True`
- Preset flexibility available (can switch to "basic", "momentum", "quality")

### Output Compatibility

- Feature count should be **identical** (comprehensive preset wraps same logic)
- Feature names should be **identical**
- DataFrame shape: `(n_stocks, ~400+ features)` depending on input data

## Troubleshooting

### Issue: Import Error

```python
ImportError: cannot
import name

'build_features'
from

'finance_ml.ml_workflow.features.api'
```

**Solution:**

1. Verify finance_ml package is up to date
2. Check that `finance_ml/ml_workflow/features/api.py` exists
3. Restart Jupyter kernel

### Issue: Different Feature Count

**Diagnosis:**

```python
# Compare outputs
legacy_features = features_build_comprehensive(df, ...)
modern_features = build_features(df, preset="comprehensive", ...)

print(f"Legacy: {legacy_features.shape}")
print(f"Modern: {modern_features.shape}")
print(f"Diff: {set(legacy_features.columns) - set(modern_features.columns)}")
```

**Expected:** Should be identical. If not, report as bug.

### Issue: Parameter Name Error

```python
TypeError: build_features()
got
an
unexpected
keyword
argument
'include_relative_values'
```

**Solution:** Change parameter name to `include_relative=True`

### Issue: Downstream Phase Errors

**Diagnosis:** Check which features are missing:

```python
# In Phase 9.4 or 9.5, if feature missing:
required_features = ['p_e_ratio', 'roe', ...]
missing = [f for f in required_features if f not in all_stocks_features.columns]
print(f"Missing features: {missing}")
```

**Solution:** Verify comprehensive preset includes all required features.

## Rollback Plan

If issues occur:

1. **Revert imports:**
   ```python
   # Remove:
   from finance_ml.ml_workflow.features.api import build_features
   
   # Keep only:
   from finance_ml import features_build_comprehensive
   ```

2. **Revert function calls:**
   ```python
   # Change back to:
   all_stocks_features = features_build_comprehensive(
       all_stocks_scaled,
       include_interactions=True,
       include_relative_values=True,
       sector_col='sector'
   )
   ```

3. **Document issue:**
    - Capture error messages
    - Note which phase failed
    - Report to development team

## Benefits of Migration

1. **Standards Compliance:** Aligns with code_guidelines.md v1.3+
2. **Preset Flexibility:** Can easily switch between feature sets
3. **Maintainability:** Uses forward-facing API, not legacy alias
4. **Documentation:** Clear preset names improve readability
5. **Testing:** Preset-based approach is better tested
6. **Future-Proof:** As new presets are added, easy to adopt

## Next Steps After Migration

1. **Test Coverage:** Run full test suite to ensure no regressions
2. **Performance:** Compare execution time (should be identical)
3. **Experiment with Presets:**
   ```python
   # Try different presets for comparison
   basic_features = build_features(df, preset="basic")
   momentum_features = build_features(df, preset="momentum")
   quality_features = build_features(df, preset="quality")
   ```
4. **Documentation:** Update any external documentation referencing the old function
5. **Notebook Structure:** Review duplicate sections issue (Priority 2 from gap analysis)

## References

- **Gap Analysis:** CODEBASE_GAP_ANALYSIS.md
- **Code Guidelines:** docs/code_guidelines.md v1.3+, Section 1.3 (Phase 9.3)
- **API Documentation:** finance_ml/ml_workflow/features/api.py docstrings
- **Test Examples:** tests/test_features.py

## Change Summary

| Aspect      | Before                                                | After                                                            |
|-------------|-------------------------------------------------------|------------------------------------------------------------------|
| Function    | `features_build_comprehensive()`                      | `build_features()`                                               |
| Import      | `from finance_ml import features_build_comprehensive` | `from finance_ml.ml_workflow.features.api import build_features` |
| Preset      | None (implicit comprehensive)                         | `preset="comprehensive"` (explicit)                              |
| Parameter   | `include_relative_values=True`                        | `include_relative=True`                                          |
| Alignment   | Legacy alias                                          | code_guidelines.md v1.3+                                         |
| Flexibility | Single mode                                           | 4 presets available                                              |

---

**Migration Priority:** HIGH  
**Risk Level:** LOW (API is compatible wrapper)  
**Estimated Time:** 15-30 minutes  
**Testing Time:** 15 minutes  
**Total Effort:** ~1 hour including validation
